"""
Membangun narrative.jsonl + tables.jsonl dari:
- text layer PDF yang sudah dibersihkan dari watermark/header/footer
- hasil OCR tabel PaddleOCR (tables_paddle.jsonl)

Output memakai kontrak yang sama dengan pipeline lama sehingga
cleaner.py, normalizer, dan chunker dapat memprosesnya.

Contoh:
    python paddle_build.py \
        --pdf ../knowledge/pdf_6623babcb5fea80d742728f6.pdf \
        --detection ../extracted/pdf_6623babcb5fea80d742728f6/lattice_all/tables.jsonl \
        --tables ../extracted_paddle/pdf_6623babcb5fea80d742728f6/tables_paddle.jsonl \
        --out ../extracted_paddle/pdf_6623babcb5fea80d742728f6
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf
import pdfplumber

from pipeline import detect_running_rects


# ============================================================
# KONFIGURASI
# ============================================================

HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?)\s+[A-ZÀ-ÿ]"
)

UPPER_HEADING_RE = re.compile(
    r"^[A-Z][A-Z\s\-–—/&]{4,120}$"
)

TITLE_HEADING_RE = re.compile(
    r"^[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+){0,5}$"
)

GLOSSARY_RE = re.compile(
    r"\b(PENJELASAN UMUM|SINGKATAN|ISTILAH DAN DEFINISI|"
    r"ISTILAH DAN KODE WILAYAH)\b",
    re.IGNORECASE,
)

URL_RE = re.compile(
    r"(https?://|www\.|sanggaukab\.bps\.go\.id|bps\.go\.id)",
    re.IGNORECASE,
)

TABLE_NUMBER_RE = re.compile(
    r"(?:Tabel|Table)\s+(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

PAGE_NUMBER_RE = re.compile(
    r"^(?:halaman\s*)?[-–\s]*(\d{1,4}|[ivxlcdm]{1,6})[-–\s]*$",
    re.IGNORECASE,
)

LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

ZERO_WIDTH = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u2060\ufeff"),
    None,
)


# ============================================================
# TEKS
# ============================================================

def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFC", str(value))

    for source, target in LIGATURES.items():
        text = text.replace(source, target)

    text = text.translate(ZERO_WIDTH)
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufffd", "")
    text = text.replace("\u00b2", "2").replace("\u00b3", "3")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)

    return text.strip()


def is_heading(text: str) -> bool:
    text = text.strip()

    if not text or len(text) > 160:
        return False

    if HEADING_RE.match(text):
        return True

    if UPPER_HEADING_RE.match(text) and len(text.split()) <= 12:
        return True

    # Subjudul title case hasil rekonstruksi paragraf, mis. "Keadaan Geografis".
    if TITLE_HEADING_RE.match(text):
        return True

    return False


def is_glossary_title(text: str) -> bool:
    return bool(GLOSSARY_RE.search(text or ""))


def line_is_diagonal(line: dict[str, Any]) -> bool:
    direction = line.get("dir") or (1.0, 0.0)
    dx, dy = float(direction[0]), float(direction[1])

    # Watermark BPS miring ~45 derajat. Garis horizontal/vertikal
    # tetap dipertahankan agar tabel landscape tidak ikut hilang.
    return abs(dx) > 0.01 and abs(dy) > 0.01


def rect_contains(rect: tuple[float, float, float, float], x: float, y: float) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def word_in_rects(
    rects: list[tuple[float, float, float, float]],
    x: float,
    y: float,
) -> bool:
    return any(rect_contains(rect, x, y) for rect in rects)


# ============================================================
# METADATA
# ============================================================

def extract_metadata(pdf_path: Path) -> dict[str, Any]:
    text_parts: list[str] = []

    with pymupdf.open(pdf_path) as document:
        for page_index in range(min(8, len(document))):
            text_parts.append(document[page_index].get_text())

    text = "\n".join(text_parts)

    kecamatan = None
    tahun = None

    match = re.search(
        r"KECAMATAN\s+(.+?)\s*\n?\s*DALAM\s+ANGKA",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if match:
        raw = re.sub(r"\s+", " ", match.group(1)).strip()
        raw = raw.title()
        raw = re.sub(r"^Kecamatan\s+", "", raw, flags=re.IGNORECASE)
        kecamatan = raw

    year_match = re.search(
        r"DALAM\s+ANGKA\s*\n?\s*(20\d{2})",
        text,
        re.IGNORECASE,
    )

    if year_match:
        tahun = int(year_match.group(1))
    else:
        fallback = re.search(r"\b(20\d{2})\b", text)
        if fallback:
            tahun = int(fallback.group(1))

    catalog = None
    catalog_match = re.search(r"Katalog\s*:\s*([\d.]+)", text)
    if catalog_match:
        catalog = catalog_match.group(1)

    # Kabupaten dibaca dari halaman judul agar pipeline tidak terikat
    # pada satu kabupaten. Fallback lama dipertahankan.
    kabupaten = "Sanggau"
    kabupaten_match = re.search(
        r"KABUPATEN\s+([A-ZÀ-Ý][A-ZÀ-Ý .'-]{2,40})",
        text,
    )
    if kabupaten_match:
        raw_kabupaten = re.split(
            r"\b(?:PROVINSI|KECAMATAN|DALAM|ANGKA)\b",
            kabupaten_match.group(1),
        )[0]
        raw_kabupaten = re.sub(r"\s+", " ", raw_kabupaten).strip()
        if raw_kabupaten:
            kabupaten = raw_kabupaten.title()

    if kecamatan and tahun:
        title = f"Kecamatan {kecamatan} Dalam Angka {tahun}"
    elif kecamatan:
        title = f"Kecamatan {kecamatan} Dalam Angka"
    else:
        title = pdf_path.stem

    return {
        "kecamatan": kecamatan,
        "kabupaten": kabupaten,
        "tahun": tahun,
        "document_type": "Kecamatan Dalam Angka",
        "document": title,
        "document_title": title,
        "document_id": pdf_path.stem,
        "source": pdf_path.name,
        "catalog": catalog,
    }


# ============================================================
# NARASI
# ============================================================

def read_page_paragraphs(
    page: pymupdf.Page,
    skip_rects: list[tuple[float, float, float, float]],
    running_rects: list[tuple[float, float, float, float]],
) -> list[str]:
    """
    Menggabungkan baris teks menjadi paragraf.

    PDF BPS sering menyimpan setiap baris sebagai blok terpisah, sehingga
    pengelompokan per blok menghasilkan satu paragraf per baris. Di sini baris
    dikelompokkan berdasarkan posisi vertikal dan ketebalan font:
    - fragmen dengan baseline sama digabung menjadi satu baris visual
      (penting untuk teks justify yang terpecah per kata),
    - baris visual yang berdekatan digabung menjadi paragraf,
    - baris bold dianggap judul dan selalu menjadi paragraf sendiri.
    """

    data = page.get_text("dict")
    page_height = page.rect.height

    lines: list[dict[str, Any]] = []

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            if line_is_diagonal(line):
                continue

            spans = line.get("spans", [])
            text = normalize_text(
                "".join(span.get("text", "") for span in spans)
            )

            if not text:
                continue

            if URL_RE.search(text):
                continue

            x0, y0, x1, y1 = line.get("bbox", (0, 0, 0, 0))

            if PAGE_NUMBER_RE.match(text):
                continue

            if skip_rects and word_in_rects(skip_rects, x0, y0):
                continue

            if running_rects and word_in_rects(running_rects, x0, y0):
                continue

            # Buang nomor halaman/footer yang tertinggal di tepi.
            if y0 > page_height * 0.94 and len(text) <= 6:
                continue

            char_count = sum(len(span.get("text", "")) for span in spans) or 1
            bold_count = sum(
                len(span.get("text", ""))
                for span in spans
                if int(span.get("flags", 0)) & 16
            )

            lines.append({
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "text": text,
                "chars": char_count,
                "bold": bold_count,
            })

    if not lines:
        return []

    lines.sort(key=lambda item: (item["y0"], item["x0"]))

    # 1. Gabungkan fragmen dengan baseline yang sama.
    visual: list[dict[str, Any]] = []

    for line in lines:
        if visual and abs(line["y0"] - visual[-1]["y0"]) <= 2.0:
            previous = visual[-1]
            joiner = "" if line["x0"] - previous["x1"] < 1.0 else " "
            previous["text"] = normalize_text(
                f"{previous['text']}{joiner}{line['text']}"
            )
            previous["x1"] = max(previous["x1"], line["x1"])
            previous["y1"] = max(previous["y1"], line["y1"])
            previous["chars"] += line["chars"]
            previous["bold"] += line["bold"]
            continue

        visual.append(dict(line))

    # 2. Gabungkan baris visual menjadi paragraf.
    for line in visual:
        line["heading"] = (
            line["bold"] / line["chars"] >= 0.6
            and len(line["text"]) <= 160
        )

    paragraphs: list[str] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current

        if current and current["text"]:
            paragraphs.append(normalize_text(current["text"]))

        current = None

    for line in visual:
        if current is None:
            current = dict(line)
            continue

        gap = line["y0"] - current["y1"]
        overlap = min(current["x1"], line["x1"]) - max(current["x0"], line["x0"])
        width = min(
            current["x1"] - current["x0"],
            line["x1"] - line["x0"],
        ) or 1.0

        same_paragraph = (
            not line["heading"]
            and not current["heading"]
            and -3.0 <= gap <= 6.5
            and overlap / width >= 0.3
        )

        if same_paragraph:
            current["text"] = normalize_text(
                f"{current['text']} {line['text']}"
            )
            current["x0"] = min(current["x0"], line["x0"])
            current["x1"] = max(current["x1"], line["x1"])
            current["y1"] = max(current["y1"], line["y1"])
            current["chars"] += line["chars"]
            current["bold"] += line["bold"]
        else:
            flush()
            current = dict(line)

    flush()

    return paragraphs


def build_narrative(
    pdf_path: Path,
    table_rects: dict[int, list[tuple[float, float, float, float]]],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    running = detect_running_rects(pdf_path)

    narratives: list[dict[str, Any]] = []
    glossaries: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []

    current_title = "Narasi"
    section_index = 0
    section_paragraphs: list[str] = []
    section_page: int | None = None
    section_end: int | None = None
    glossary_active: str | None = None

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_no = page_index + 1

            paragraphs = read_page_paragraphs(
                page,
                skip_rects=table_rects.get(page_no, []),
                running_rects=running.get(page_no, []),
            )

            page_text = "\n\n".join(paragraphs).strip()

            raw_pages.append({
                "page": page_no,
                "text": page_text,
                "word_count": len(page_text.split()),
                **metadata,
            })

            if not paragraphs:
                continue

            # Daftar isi / daftar tabel / daftar gambar: banyak titik
            # penuntun di halaman yang sama. Tidak perlu masuk RAG.
            leader_count = sum(
                len(re.findall(r"(?:\.\s?){4,}|…{2,}", paragraph))
                for paragraph in paragraphs
            )

            if leader_count >= 4:
                continue

            joined = "\n".join(paragraphs)

            # Glosarium: halaman awal setelah daftar isi. Halaman lanjutan
            # tidak diawali judul bab baru.
            if 10 <= page_no <= 60:
                glossary_match = GLOSSARY_RE.search(joined)

                if glossary_match:
                    glossary_active = " ".join(
                        word.capitalize()
                        for word in glossary_match.group(1).split()
                    )
                elif glossary_active and not is_heading(paragraphs[0]):
                    pass
                else:
                    glossary_active = None

                if glossary_active:
                    title = glossary_active

                    glossaries.append({
                        "page": page_no,
                        "title": title,
                        "header": title,
                        "content": joined,
                        "source": metadata["document"],
                        "document": metadata["document"],
                        "type": "glossary",
                        **metadata,
                    })
                    continue

            def flush() -> None:
                nonlocal section_index, section_paragraphs, section_page, section_end

                content = "\n\n".join(section_paragraphs).strip()

                if len(content) < 30:
                    section_paragraphs = []
                    section_page = None
                    section_end = None
                    return

                section_index += 1
                page_start = section_page or page_no
                page_finish = section_end or page_no

                narratives.append({
                    "type": "narrative",
                    "title": current_title,
                    "section": current_title,
                    "page": page_start,
                    "page_end": page_finish,
                    "section_index": section_index,
                    "content": content,
                    "paragraphs": list(section_paragraphs),
                    "word_count": len(content.split()),
                    "source": metadata["document"],
                    "document": metadata["document"],
                    "metadata": {
                        "normalization_method": "paddle_hybrid",
                    },
                    **metadata,
                })

                section_paragraphs = []
                section_page = None
                section_end = None

            for paragraph in paragraphs:
                if is_heading(paragraph):
                    flush()
                    current_title = paragraph
                else:
                    if not section_paragraphs:
                        section_page = page_no
                    section_end = page_no
                    section_paragraphs.append(paragraph)

            flush()

    return narratives, glossaries, raw_pages


# ============================================================
# TABEL
# ============================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


CAPTION_BAND = 130.0


def recover_caption(
    document: pymupdf.Document | None,
    record: dict[str, Any],
    table_number: str | None,
) -> str:
    """
    Ambil ulang judul tabel dari text layer di atas bbox tabel.

    Caption hasil deteksi lama sering hanya menangkap sebagian judul
    (mis. "Tabel 2.2.1 2025"). Semua baris pada satu baseline digabung
    lebih dulu agar judul yang terpecah antarfragmen tetap utuh.
    """

    if document is None:
        return ""

    try:
        page = document[int(record["page"]) - 1]
    except (KeyError, IndexError, ValueError):
        return ""

    bbox = record.get("bbox_pymupdf") or record.get("bbox")

    if not bbox or len(bbox) != 4:
        return ""

    top = float(bbox[1])
    raw: list[dict[str, Any]] = []

    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            if line_is_diagonal(line):
                continue

            text = normalize_text(
                "".join(span.get("text", "") for span in line.get("spans", []))
            )

            if not text:
                continue

            x0, y0, x1, y1 = line.get("bbox", (0, 0, 0, 0))

            if y1 <= top + 1 and y0 >= top - CAPTION_BAND:
                raw.append({
                    "x0": float(x0),
                    "y0": float(y0),
                    "x1": float(x1),
                    "y1": float(y1),
                    "text": text,
                })

    if not raw:
        return ""

    raw.sort(key=lambda item: (item["y0"], item["x0"]))

    grouped: list[dict[str, Any]] = []

    for item in raw:
        if grouped and abs(item["y0"] - grouped[-1]["y0"]) <= 2.0:
            grouped[-1]["items"].append(item)
        else:
            grouped.append({
                "y0": item["y0"],
                "items": [item],
            })

    groups: list[dict[str, Any]] = []

    for group in grouped:
        # Urutkan berdasarkan x agar "Tabel 4.1.1" tidak didahului
        # angka superscript dari baris yang sama.
        items = sorted(group["items"], key=lambda item: item["x0"])

        groups.append({
            "x0": min(item["x0"] for item in items),
            "y0": group["y0"],
            "x1": max(item["x1"] for item in items),
            "y1": max(item["y1"] for item in items),
            "text": normalize_text(
                " ".join(item["text"] for item in items)
            ),
        })

    caption_re = re.compile(r"^(?:lanjutan\s+)?tabel\s+", re.IGNORECASE)
    start = None

    # Ambil caption terdekat di atas tabel, bukan yang paling atas.
    for index in range(len(groups) - 1, -1, -1):
        if caption_re.match(groups[index]["text"]):
            start = index
            break

    if start is None:
        return ""

    merged = groups[start]["text"]
    last_y1 = groups[start]["y1"]

    for group in groups[start + 1:]:
        if group["y0"] - last_y1 > 6.0:
            break

        if re.match(r"^(sumber|catatan|keterangan)\b", group["text"], re.IGNORECASE):
            break

        merged = normalize_text(f"{merged} {group['text']}")
        last_y1 = group["y1"]

    # "Tabel 4.1 . 1" -> "Tabel 4.1.1"
    merged = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", merged)
    merged = re.sub(r"\s+([,.;:])", r"\1", merged)

    return merged


def _is_upright_object(obj: dict[str, Any]) -> bool:
    """Buang teks watermark yang diputar (matrix b/c != 0)."""

    matrix = obj.get("matrix")

    if not matrix:
        return True

    try:
        return abs(float(matrix[1])) < 1e-3 and abs(float(matrix[2])) < 1e-3
    except (TypeError, ValueError, IndexError):
        return True


def _clean_table_cell(value: Any) -> str:
    """Normalisasi isi sel hasil text layer."""

    if value is None:
        return ""

    text = normalize_text(str(value))

    # BPS memakai glyph ini untuk "tidak ada atau nol".
    text = text.replace("\ufffd", "0")

    # Fragmen tahun akibat text layer terpotong: "2024/20 25" -> "2024/2025".
    text = re.sub(r"((?:19|20)\d{2})\s*/\s*(\d{2,4})", r"\1/\2", text)
    text = re.sub(r"((?:19|20)\d{2}/\d{2})\s+(\d{2})\b", r"\1\2", text)

    return re.sub(r"\s+", " ", text).strip()


def _cluster_positions(values: list[float], tolerance: float = 1.0) -> list[list[float]]:
    """Kelompokkan koordinat yang berdekatan (garis ganda akibat rect bertumpuk)."""

    groups: list[list[float]] = []

    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)

    return groups


def _outer_cell_rects(rects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ambil rect sel terluar.

    PDF BPS sering menggambar setiap sel sebagai rect terisi, lalu menambahkan
    rect latar yang lebih kecil di dalamnya. Rect dalam itu bukan sel baru dan
    harus dibuang agar grid kolom tidak terpecah.
    """

    valid = [
        rect for rect in rects
        if (rect["x1"] - rect["x0"]) > 1.0 and (rect["bottom"] - rect["top"]) > 1.0
    ]

    keep: list[dict[str, Any]] = []

    for rect in valid:
        width = rect["x1"] - rect["x0"]
        height = rect["bottom"] - rect["top"]
        area = width * height
        nested = False

        for outer in valid:
            if outer is rect:
                continue

            outer_area = (outer["x1"] - outer["x0"]) * (outer["bottom"] - outer["top"])

            if outer_area <= area * 1.02:
                continue

            left = rect["x0"] - outer["x0"]
            right = outer["x1"] - rect["x1"]

            if (
                0.8 <= left <= 20.0
                and 0.8 <= right <= 20.0
                and outer["top"] <= rect["top"] + 0.5
                and outer["bottom"] >= rect["bottom"] - 0.5
            ):
                nested = True
                break

        if not nested:
            keep.append(rect)

    return keep


def _column_boundaries(rects: list[dict[str, Any]]) -> list[float]:
    """Batas kolom yang didukung minimal dua baris; buang garis geser satu baris."""

    groups = _cluster_positions(
        [value for rect in rects for value in (rect["x0"], rect["x1"])]
    )

    supported: list[list[float]] = []

    for group in groups:
        rows = set()

        for rect in rects:
            if any(
                abs(edge - member) <= 1.0
                for member in group
                for edge in (rect["x0"], rect["x1"])
            ):
                rows.add(round(rect["top"], 1))

        if len(rows) >= 2:
            supported.append(group)

    # Tabel sangat kecil bisa punya batas kolom yang hanya muncul sekali.
    if len(supported) < 3:
        supported = groups

    return [sum(group) / len(group) for group in supported]


def _extract_lines_table(cropped: pdfplumber.page.Page) -> list[list[str]] | None:
    """Fallback untuk tabel yang digambar dengan garis, bukan rect sel."""

    try:
        rows = cropped.extract_table({
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "text_x_tolerance": 1,
            "text_y_tolerance": 1,
        })
    except Exception:
        return None

    if not rows:
        return None

    cleaned = [[_clean_table_cell(cell) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]

    if len(cleaned) < 2:
        return None

    width = max(len(row) for row in cleaned)

    if width < 2:
        return None

    return [row + [""] * (width - len(row)) for row in cleaned]


def _extract_rect_grid(cropped: pdfplumber.page.Page) -> list[list[str]] | None:
    """Bangun tabel dari rect sel terluar, termasuk merged cell."""

    rects = _outer_cell_rects(cropped.rects)

    if len(rects) < 4:
        return None

    # Baris footnote/catatan sering berada di dalam bbox deteksi dan
    # tergambar sebagai satu rect per baris. Buang kelompok seperti itu.
    top_groups = _cluster_positions([rect["top"] for rect in rects], tolerance=2.0)
    top_centers = [sum(group) / len(group) for group in top_groups]
    counts = [0] * len(top_centers)

    for rect in rects:
        index = min(
            range(len(top_centers)),
            key=lambda i: abs(top_centers[i] - rect["top"]),
        )
        counts[index] += 1

    valid_rows = {index for index, count in enumerate(counts) if count >= 2}

    if len(valid_rows) >= 2:
        rects = [
            rect for rect in rects
            if min(
                range(len(top_centers)),
                key=lambda i: abs(top_centers[i] - rect["top"]),
            ) in valid_rows
        ]

    if len(rects) < 4:
        return None

    xs = _column_boundaries(rects)
    ys = [
        sum(group) / len(group)
        for group in _cluster_positions(
            [value for rect in rects for value in (rect["top"], rect["bottom"])]
        )
    ]

    if len(xs) < 3 or len(ys) < 2:
        return None

    words = cropped.extract_words(x_tolerance=1.0, y_tolerance=1.0)
    grid_x0, grid_x1 = min(xs), max(xs)
    grid_y0, grid_y1 = min(ys), max(ys)
    cells: dict[tuple[int, int], str] = {}

    # Kelompokkan kata per baris visual. Superscript/footnote membuat
    # ``top`` sedikit berbeda sehingga sort per ``top`` bisa menukar
    # urutan kata dalam satu baris (mis. "Jenis Sarana Transportasi").
    line_centers = [
        sum(group) / len(group)
        for group in _cluster_positions(
            [(word["top"] + word["bottom"]) / 2 for word in words],
            tolerance=3.0,
        )
    ]

    def line_index(word: dict[str, Any]) -> int:
        center = (word["top"] + word["bottom"]) / 2
        return min(
            range(len(line_centers)),
            key=lambda i: abs(line_centers[i] - center),
        )

    for rect in rects:
        parts: list[tuple[int, float, str]] = []

        for word in words:
            center_x = (word["x0"] + word["x1"]) / 2
            center_y = (word["top"] + word["bottom"]) / 2

            if not (grid_x0 <= center_x <= grid_x1 and grid_y0 <= center_y <= grid_y1):
                continue

            if (
                rect["x0"] <= center_x <= rect["x1"]
                and rect["top"] <= center_y <= rect["bottom"]
            ):
                parts.append((line_index(word), word["x0"], word["text"]))

        text = _clean_table_cell(
            " ".join(part[2] for part in sorted(parts))
        )

        if not text:
            continue

        row_index = min(range(len(ys)), key=lambda i: abs(ys[i] - rect["top"]))
        col_index = min(range(len(xs)), key=lambda i: abs(xs[i] - rect["x0"]))

        if row_index >= len(ys) - 1 or col_index >= len(xs) - 1:
            continue

        key = (row_index, col_index)
        cells[key] = f"{cells[key]} {text}".strip() if key in cells else text

    out = [
        ["" for _ in range(len(xs) - 1)]
        for _ in range(len(ys) - 1)
    ]

    for (row_index, col_index), text in cells.items():
        out[row_index][col_index] = text

    out = [row for row in out if any(row)]

    caption_re = re.compile(
        r"^(lanjutan\s+)?tabel\b|^sumber\b|^catatan\b|^keterangan\b",
        re.IGNORECASE,
    )
    stray_re = re.compile(r"^(desa|kelurahan|kecamatan|kabupaten|provinsi)$", re.IGNORECASE)

    filtered: list[list[str]] = []

    for row_index, row in enumerate(out):
        values = [value for value in row if value]

        if len(values) == 1 and caption_re.match(values[0]):
            continue

        # Fragmen judul/header yang jatuh ke baris sendiri.
        if row_index < 4 and len(values) == 1 and stray_re.match(values[0]):
            continue

        filtered.append(row)

    if not filtered:
        return None

    # Buang kolom yang kosong di semua baris (kolom pemisah sempit).
    keep = [
        index for index in range(len(filtered[0]))
        if any(row[index] for row in filtered)
    ]
    result = [[row[index] for index in keep] for row in filtered]

    if len(result) < 2 or len(result[0]) < 2:
        return None

    return result


def extract_textlayer_table(
    page: pdfplumber.page.Page,
    bbox: tuple[float, float, float, float],
) -> list[list[str]] | None:
    """
    Ambil tabel dari text layer PDF, tanpa karakter watermark yang diputar.

    Hanya dipakai sebagai rekonsiliasi nilai: PPStructureV3 tetap menjadi
    sumber deteksi tabel dan fallback untuk halaman yang tidak punya
    text layer (hasil scan).
    """

    x0, y0, x1, y1 = bbox
    padding = 2.0

    crop_box = (
        max(0.0, x0 - padding),
        max(0.0, y0 - padding),
        min(float(page.width), x1 + padding),
        min(float(page.height), y1 + padding),
    )

    try:
        cropped = page.filter(_is_upright_object).crop(crop_box)
    except Exception:
        return None

    return _extract_rect_grid(cropped) or _extract_lines_table(cropped)


def build_tables(
    table_records: list[dict[str, Any]],
    metadata: dict[str, Any],
    pdf_path: Path | None = None,
) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    document = pymupdf.open(pdf_path) if pdf_path else None
    plumber = pdfplumber.open(pdf_path) if pdf_path else None

    try:
        for record in table_records:
            rows = record.get("raw_rows") or record.get("clean_rows") or []

            caption = normalize_text(record.get("caption") or "")

            table_number = record.get("table_number")

            if not table_number:
                match = TABLE_NUMBER_RE.search(caption)
                if match:
                    table_number = match.group(1)

            recovered = recover_caption(document, record, table_number)

            if (
                recovered
                and len(recovered) > len(caption)
                and TABLE_NUMBER_RE.search(recovered)
            ):
                caption = recovered

                if not table_number:
                    match = TABLE_NUMBER_RE.search(caption)
                    if match:
                        table_number = match.group(1)

            is_continuation = bool(record.get("is_continuation")) or bool(
                re.search(r"\blanjutan\b", caption, re.IGNORECASE)
            )

            status = record.get("status") or "accepted"

            if status not in ("accepted", "needs_review"):
                status = "needs_review"

            reconciled = None
            value_source = "ocr"
            if plumber is not None:
                bbox = record.get("bbox_pymupdf") or record.get("bbox")

                try:
                    page = plumber.pages[int(record["page"]) - 1]
                except (KeyError, IndexError, ValueError, TypeError):
                    page = None

                if page is not None and bbox and len(bbox) == 4:
                    reconciled = extract_textlayer_table(
                        page,
                        tuple(float(value) for value in bbox),
                    )

            if reconciled:
                value_source = "text_layer"
                status = "accepted"
            elif not rows:
                status = "needs_review"

            tables.append({
                **record,
                "raw_rows": reconciled or rows,
                "clean_rows": reconciled or rows,
                "value_source": value_source,
                "caption": caption,
                "table_number": table_number,
                "is_continuation": is_continuation,
                "status": status,
                "content_type": "table",
                "source": metadata["document"],
                "document": metadata["document"],
                **metadata,
            })
    finally:
        if document is not None:
            document.close()

        if plumber is not None:
            plumber.close()

    tables.sort(key=lambda item: (item.get("page", 0), item.get("table_id", 0)))

    # Fragmen tanpa judul di halaman berikutnya adalah lanjutan tabel
    # sebelumnya. Beri nomor agar sitasi dan penggabungan tetap benar.
    last_number = None
    last_end = None

    for table in tables:
        if not table.get("table_number") and not table.get("caption"):
            page = int(table.get("page") or 0)

            if last_number and last_end and page <= last_end + 1:
                table["table_number"] = last_number
                table["caption"] = f"Lanjutan Tabel {last_number}"
                table["is_continuation"] = True

        if table.get("table_number"):
            last_number = table["table_number"]
            last_end = int(table.get("page_end") or table.get("page") or 0)

    return tables


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bangun narasi + tabel dari text layer dan OCR Paddle."
    )

    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--detection", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    for path in (args.pdf, args.detection, args.tables):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = extract_metadata(args.pdf)

    print("Metadata:")
    for key, value in metadata.items():
        print(f"  {key:16}: {value}")

    detections = read_jsonl(args.detection)

    table_rects: dict[int, list[tuple[float, float, float, float]]] = {}

    for record in detections:
        bbox = record.get("bbox_expanded") or record.get("bbox_pymupdf")

        if not bbox or len(bbox) != 4:
            continue

        table_rects.setdefault(
            int(record["page"]), []
        ).append(tuple(float(value) for value in bbox))

    narratives, glossaries, raw_pages = build_narrative(
        args.pdf,
        table_rects,
        metadata,
    )

    table_records = read_jsonl(args.tables)
    tables = build_tables(table_records, metadata, pdf_path=args.pdf)

    write_jsonl(args.out / "narrative.jsonl", narratives)
    write_jsonl(args.out / "glossary.jsonl", glossaries)
    write_jsonl(args.out / "tables.jsonl", tables)
    write_jsonl(args.out / "raw_pages.jsonl", raw_pages)

    summary = {
        **metadata,
        "page_count": len(raw_pages),
        "narrative_records": len(narratives),
        "glossary_records": len(glossaries),
        "table_records": len(tables),
        "tables_accepted": sum(
            item.get("status") == "accepted" for item in tables
        ),
        "tables_needs_review": sum(
            item.get("status") == "needs_review" for item in tables
        ),
        "tables_without_rows": sum(
            not item.get("raw_rows") for item in tables
        ),
        "engine": "paddleocr+textlayer",
    }

    with (args.out / "build_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nRingkasan:")
    for key, value in summary.items():
        print(f"  {key:20}: {value}")


if __name__ == "__main__":
    main()
