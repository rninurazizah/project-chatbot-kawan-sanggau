"""
Ekstraksi tabel PDF BPS dengan PaddleOCR PPStructureV3 (hybrid).

Alur:
1. Baca hasil deteksi tabel dari pipeline lama (tables.jsonl) untuk
   mendapatkan halaman + bbox tabel.
2. Render halaman PDF, hapus watermark diagonal (overlay hitam 7%),
   lalu crop area tabel.
3. Jalankan PPStructureV3 pada crop tabel. Hasil HTML tabel disimpan
   sebagai raw JSON per tabel sehingga proses bisa dilanjutkan (resume).
4. Tulis tables_paddle.jsonl dengan kontrak yang sama seperti
   pipeline lama (raw_rows, clean_rows, bbox, caption, notes, dst).

Narasi TIDAK diambil dari sini. Narasi dibangun oleh paddle_build.py
dari text layer PDF yang sudah dibersihkan dari watermark.

Contoh:
    python paddle_extract.py \
        --pdf ../knowledge/pdf_6623babcb5fea80d742728f6.pdf \
        --detection ../extracted/pdf_6623babcb5fea80d742728f6/lattice_all/tables.jsonl \
        --out ../extracted_paddle/pdf_6623babcb5fea80d742728f6 \
        --dpi 200 --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from PIL import Image


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DPI = 200
DEFAULT_PADDING = 3.5  # poin PDF

# Watermark BPS = overlay hitam tipis (alpha berbeda per halaman, sekitar
# 4-8%) di atas warna latar. Dibuat peta warna hasil overlay -> warna asli
# untuk warna latar yang dipakai layout BPS.
WATERMARK_BACKGROUNDS = [
    (255, 255, 255),  # putih
    (255, 247, 231),  # krem isi tabel
    (255, 204, 48),   # kuning header tabel
    (255, 222, 137),  # kuning muda
    (255, 216, 107),  # kuning sedang
    (255, 231, 174),  # kuning sangat muda
]


def build_watermark_map() -> dict[tuple[int, int, int], tuple[int, int, int]]:
    mapping: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    for background in WATERMARK_BACKGROUNDS:
        for step in range(20, 121):
            alpha = step / 1000.0
            overlay = tuple(
                int(round(channel * (1.0 - alpha)))
                for channel in background
            )
            mapping.setdefault(overlay, background)

    return mapping


WATERMARK_MAP = build_watermark_map()

ZERO_WIDTH = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u2060\ufeff"),
    None,
)


# ============================================================
# UTILITAS TEKS
# ============================================================

def normalize_cell(value: Any) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFC", str(value))

    text = text.translate(ZERO_WIDTH)
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufffd", "")
    text = re.sub(r"\s+", " ", text)

    # Perbaikan satuan yang sering rusak karena font PDF.
    text = re.sub(r"\bkm\s*[²2]?\b", "km2", text)

    return text.strip()


def clean_html_text(value: str) -> str:
    text = normalize_cell(value)
    text = text.replace("|", "\\|")
    return text


# ============================================================
# PARSER HTML TABEL
# ============================================================

class TableHTMLParser(HTMLParser):
    """Mengubah HTML tabel PaddleOCR menjadi list baris + cell."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self._row: list[dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        values = dict(attrs)

        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = {
                "text": "",
                "colspan": self._int_attr(values.get("colspan")),
                "rowspan": self._int_attr(values.get("rowspan")),
            }
            self._buffer = []
        elif tag == "br" and self._cell is not None:
            self._buffer.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in ("td", "th") and self._cell is not None:
            self._cell["text"] = "".join(self._buffer)
            if self._row is not None:
                self._row.append(self._cell)
            self._cell = None
            self._buffer = []

        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    @staticmethod
    def _int_attr(value: str | None) -> int:
        try:
            return max(1, int(value)) if value else 1
        except (TypeError, ValueError):
            return 1


def html_to_grid(html: str) -> list[list[str]]:
    """HTML -> grid 2D dengan colspan/rowspan sudah diekspansi."""

    parser = TableHTMLParser()
    parser.feed(html or "")

    raw_rows = parser.rows

    if not raw_rows:
        return []

    occupied: dict[tuple[int, int], str] = {}
    width = 0

    for row_index, row in enumerate(raw_rows):
        col_index = 0

        for cell in row:
            while (row_index, col_index) in occupied:
                col_index += 1

            text = normalize_cell(cell["text"])
            colspan = cell["colspan"]
            rowspan = cell["rowspan"]

            for dr in range(rowspan):
                for dc in range(colspan):
                    occupied[(row_index + dr, col_index + dc)] = text

            col_index += colspan

        width = max(width, col_index)

    if width == 0:
        return []

    height = max(r for r, _ in occupied) + 1

    return [
        [occupied.get((row, col), "") for col in range(width)]
        for row in range(height)
    ]


def row_is_empty(row: list[str]) -> bool:
    return not any(normalize_cell(cell) for cell in row)


def grid_quality(rows: list[list[str]]) -> dict[str, Any]:
    filled = [
        cell
        for row in rows
        for cell in row
        if normalize_cell(cell)
    ]

    counts = [len(row) for row in rows]
    width = max(counts, default=0)

    return {
        "row_count": len(rows),
        "column_count": width,
        "nonempty_cells": len(filled),
        "uneven_rows": sum(count != width for count in counts),
        "single_character_cells": sum(
            len(cell) == 1 for cell in filled
        ),
    }


# ============================================================
# RENDER + WATERMARK
# ============================================================

def correct_watermark(image: Image.Image) -> Image.Image:
    """Mengembalikan warna latar yang tertutup overlay watermark BPS.

    Watermark dibuat sebagai overlay hitam dengan alpha tetap. Untuk setiap
    warna latar yang dikenal, warna hasil overlay dihitung lalu dipetakan
    kembali ke warna aslinya. Pendekatan ini menghapus watermark tanpa
    mengubah teks hitam.
    """

    array = np.array(image.convert("RGB"))

    colors, counts = np.unique(
        array.reshape(-1, 3),
        axis=0,
        return_counts=True,
    )

    for color, count in zip(colors, counts):
        key = (int(color[0]), int(color[1]), int(color[2]))
        background = WATERMARK_MAP.get(key)

        if background is None:
            continue

        mask = np.all(array == color, axis=2)
        array[mask] = background

    return Image.fromarray(array)


def render_page(
    page: pymupdf.Page,
    dpi: int,
    fix_watermark: bool = True,
) -> Image.Image:
    matrix = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    if fix_watermark:
        image = correct_watermark(image)

    return image


def crop_bbox(
    image: Image.Image,
    bbox_points: list[float],
    dpi: int,
    padding_points: float,
) -> Image.Image:
    scale = dpi / 72.0

    x0, y0, x1, y1 = bbox_points

    left = max(0, int((x0 - padding_points) * scale))
    top = max(0, int((y0 - padding_points) * scale))
    right = min(image.width, int((x1 + padding_points) * scale))
    bottom = min(image.height, int((y1 + padding_points) * scale))

    if right <= left or bottom <= top:
        raise ValueError(f"BBox tabel tidak valid: {bbox_points}")

    return image.crop((left, top, right, bottom))


# ============================================================
# HASIL PPSTRUCTURE
# ============================================================

def unwrap_result(result: Any) -> dict[str, Any]:
    data = result.json

    if isinstance(data, dict) and "res" in data:
        return data["res"] or {}

    return data or {}


def extract_best_table(
    result: Any,
) -> tuple[str, list[list[str]], dict[str, Any]]:
    """
    Mengambil HTML tabel terbaik dari hasil PPStructureV3.

    Urutan prioritas:
    1. table_res_list[*].pred_html
    2. parsing_res_list block_label == "table"
    """

    data = unwrap_result(result)

    candidates: list[tuple[str, list[list[str]], dict[str, Any]]] = []

    for item in data.get("table_res_list") or []:
        html = item.get("pred_html") or ""
        rows = html_to_grid(html)

        if rows and not all(row_is_empty(row) for row in rows):
            candidates.append((
                html,
                rows,
                grid_quality(rows),
            ))

    if not candidates:
        for block in data.get("parsing_res_list") or []:
            if block.get("block_label") != "table":
                continue

            html = block.get("block_content") or ""
            rows = html_to_grid(html)

            if rows:
                candidates.append((
                    html,
                    rows,
                    grid_quality(rows),
                ))

    if not candidates:
        return "", [], grid_quality([])

    # Pilih tabel dengan isi sel terbanyak.
    return max(
        candidates,
        key=lambda item: (
            item[2]["nonempty_cells"],
            item[2]["row_count"],
            item[2]["column_count"],
        ),
    )


# ============================================================
# IO
# ============================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON tidak valid: {path}, baris {line_number}: {error}"
                )

    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def raw_key(page: int, table_id: int) -> str:
    return f"page_{page:03d}_table_{table_id:03d}"


def detection_targets(
    records: list[dict[str, Any]],
    pages: set[int] | None,
    limit: int,
) -> list[dict[str, Any]]:
    selected = []

    for record in records:
        if record.get("status") not in ("accepted", "needs_review"):
            continue

        if pages and record.get("page") not in pages:
            continue

        bbox = record.get("bbox_pymupdf") or record.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        selected.append(record)

    selected.sort(key=lambda item: (item.get("page", 0), item.get("table_id", 0)))

    if limit > 0:
        selected = selected[:limit]

    return selected


# ============================================================
# PROSES UTAMA
# ============================================================

def process(
    pdf_path: Path,
    detection_path: Path,
    output_dir: Path,
    dpi: int,
    padding: float,
    limit: int,
    pages: set[int] | None,
    force: bool,
    save_images: bool,
    fix_watermark: bool,
) -> None:
    from paddleocr import PPStructureV3

    detections = read_jsonl(detection_path)
    targets = detection_targets(detections, pages, limit)

    if not targets:
        raise ValueError("Tidak ada tabel yang dipilih.")

    raw_dir = output_dir / "paddle_raw"
    image_dir = output_dir / "paddle_images"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    print(f"PDF        : {pdf_path}")
    print(f"Deteksi    : {detection_path}")
    print(f"Output     : {output_dir}")
    print(f"Tabel      : {len(targets)}")
    print(f"DPI        : {dpi}")
    print("Memuat PPStructureV3...", flush=True)

    init_start = time.time()

    pipeline = PPStructureV3(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_seal_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_region_detection=False,
        cpu_threads=8,
    )

    print(
        f"Pipeline siap dalam {time.time() - init_start:.1f} detik.",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    processed = 0
    failed = 0

    with pymupdf.open(pdf_path) as document:
        page_cache: dict[int, Image.Image] = {}

        for index, detection in enumerate(targets, start=1):
            page_no = int(detection["page"])
            table_id = int(detection["table_id"])

            key = raw_key(page_no, table_id)
            raw_path = raw_dir / f"{key}.json"

            if raw_path.exists() and not force:
                try:
                    cached = json.loads(raw_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    cached = None

                if cached and cached.get("rows"):
                    results.append(cached["record"])
                    print(
                        f"[{index}/{len(targets)}] {key} -> cache",
                        flush=True,
                    )
                    continue

            print(
                f"[{index}/{len(targets)}] {key} ...",
                flush=True,
            )

            start = time.time()

            try:
                if page_no not in page_cache:
                    page_cache[page_no] = render_page(
                        document[page_no - 1],
                        dpi=dpi,
                        fix_watermark=fix_watermark,
                    )

                page_image = page_cache[page_no]

                bbox = (
                    detection.get("bbox_pymupdf")
                    or detection.get("bbox")
                )

                crop = crop_bbox(
                    page_image,
                    [float(value) for value in bbox],
                    dpi=dpi,
                    padding_points=padding,
                )

                if save_images:
                    crop.save(image_dir / f"{key}.png")

                # PaddleX image reader memakai urutan BGR.
                array = np.array(crop.convert("RGB"))[:, :, ::-1]

                output = pipeline.predict(
                    array,
                    use_region_detection=False,
                    # Harus True: PaddleX 3.7.2 gagal menyiapkan OCR
                    # internal tabel jika orientasi dimatikan.
                    use_table_orientation_classify=True,
                )

                html = ""
                rows: list[list[str]] = []
                quality: dict[str, Any] = grid_quality([])

                for result in output:
                    html, rows, quality = extract_best_table(result)
                    if rows:
                        break

                elapsed = time.time() - start

                status = "accepted"

                if not rows:
                    status = "needs_review"
                    failed += 1
                elif quality["uneven_rows"] or quality["row_count"] < 2:
                    status = "needs_review"

                record = {
                    "table_id": table_id,
                    "page": page_no,
                    "page_end": detection.get("page_end", page_no),
                    "engine": "paddleocr",
                    "flavor": "ppstructure_v3",
                    "bbox": detection.get("bbox"),
                    "bbox_pymupdf": detection.get("bbox_pymupdf"),
                    "bbox_expanded": detection.get("bbox_expanded"),
                    "raw_rows": rows,
                    "clean_rows": rows,
                    "cleaning": {
                        "removed_empty_rows": 0,
                        "removed_empty_columns": 0,
                        "removed_repeated_rows": 0,
                    },
                    "quality": {
                        "accepted": bool(rows),
                        "needs_review": status == "needs_review",
                        "rejection_reasons": (
                            [] if rows else ["paddle_tidak_menghasilkan_tabel"]
                        ),
                        **quality,
                    },
                    "classification": {
                        "kind": "data_table",
                        "confidence": "high" if rows else "low",
                        "reason": "ppstructure_v3",
                    },
                    "accuracy": None,
                    "whitespace": None,
                    "content_type": "table",
                    "caption": detection.get("caption"),
                    "notes": detection.get("notes"),
                    "table_number": detection.get("table_number"),
                    "is_continuation": bool(detection.get("is_continuation")),
                    "status": status,
                    "html": html,
                    "paddle": {
                        "model": "PPStructureV3",
                        "text_models": "PP-OCRv5_mobile",
                        "dpi": dpi,
                        "padding_points": padding,
                        "elapsed_seconds": round(elapsed, 2),
                    },
                }

                payload = {
                    "key": key,
                    "page": page_no,
                    "table_id": table_id,
                    "detection": detection,
                    "rows": rows,
                    "html": html,
                    "quality": quality,
                    "elapsed_seconds": round(elapsed, 2),
                    "record": record,
                }

                write_json(raw_path, payload)
                results.append(record)
                processed += 1

                print(
                    f"    selesai {elapsed:.1f}s, "
                    f"{len(rows)} baris x {quality['column_count']} kolom",
                    flush=True,
                )

            except Exception as error:
                failed += 1

                print(
                    f"    GAGAL: {type(error).__name__}: {error}",
                    flush=True,
                )

                results.append({
                    "table_id": table_id,
                    "page": page_no,
                    "engine": "paddleocr",
                    "flavor": "ppstructure_v3",
                    "bbox": detection.get("bbox"),
                    "bbox_pymupdf": detection.get("bbox_pymupdf"),
                    "raw_rows": [],
                    "clean_rows": [],
                    "quality": {
                        "accepted": False,
                        "needs_review": True,
                        "rejection_reasons": [
                            f"{type(error).__name__}: {error}"
                        ],
                    },
                    "classification": {
                        "kind": "data_table",
                        "confidence": "low",
                        "reason": "ppstructure_error",
                    },
                    "caption": detection.get("caption"),
                    "notes": detection.get("notes"),
                    "table_number": detection.get("table_number"),
                    "is_continuation": bool(detection.get("is_continuation")),
                    "status": "needs_review",
                    "html": "",
                })

    results.sort(key=lambda item: (item["page"], item["table_id"]))

    write_jsonl(output_dir / "tables_paddle.jsonl", results)

    summary = {
        "pdf": str(pdf_path),
        "detection": str(detection_path),
        "output": str(output_dir),
        "dpi": dpi,
        "padding_points": padding,
        "requested": len(targets),
        "processed_this_run": processed,
        "failed": failed,
        "total_records": len(results),
        "accepted": sum(
            item.get("status") == "accepted" for item in results
        ),
        "needs_review": sum(
            item.get("status") == "needs_review" for item in results
        ),
    }

    write_json(output_dir / "paddle_summary.json", summary)

    print("\n=== SELESAI ===")
    for key, value in summary.items():
        print(f"{key:22}: {value}")


# ============================================================
# CLI
# ============================================================

def parse_pages(value: str) -> set[int]:
    pages: set[int] = set()

    for part in filter(None, (item.strip() for item in value.split(","))):
        if "-" in part:
            start, end = part.split("-", 1)
            pages.update(range(int(start), int(end) + 1))
        else:
            pages.add(int(part))

    return pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ekstraksi tabel PDF dengan PaddleOCR PPStructureV3 "
            "menggunakan bbox dari pipeline deteksi lama."
        )
    )

    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument(
        "--detection",
        type=Path,
        required=True,
        help="tables.jsonl hasil deteksi lama",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING,
        help="Padding crop tabel dalam poin PDF",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pages", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument(
        "--no-watermark-fix",
        action="store_true",
        help="Jangan koreksi warna watermark BPS",
    )

    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)

    if not args.detection.exists():
        raise FileNotFoundError(args.detection)

    if args.dpi < 100:
        parser.error("--dpi minimal 100")

    process(
        pdf_path=args.pdf,
        detection_path=args.detection,
        output_dir=args.out,
        dpi=args.dpi,
        padding=args.padding,
        limit=args.limit,
        pages=parse_pages(args.pages) if args.pages else None,
        force=args.force,
        save_images=args.save_images,
        fix_watermark=not args.no_watermark_fix,
    )


if __name__ == "__main__":
    main()
