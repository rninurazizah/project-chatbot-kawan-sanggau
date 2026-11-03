
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import camelot
import pdfplumber
import pymupdf

from content_separation import (
    classify_table,
    expand_table_rect,
    parse_caption,
    rect_iou,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================
from pathlib import Path

# Direktori python-extractor
PYTHON_DIR = Path(__file__).resolve().parent

# Direktori utama proyek
BASE_DIR = PYTHON_DIR.parent

# Direktori sumber dan output
PDF_DIR = BASE_DIR / "knowledge"
OUTPUT_DIR = BASE_DIR / "extracted"

# Direktori tabel manual
MANUAL_DIR = PYTHON_DIR / "manual"

# Pastikan direktori tersedia
MANUAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

CAMELOT_LINE_SCALE = 40

HEADER_ZONE = 0.08
FOOTER_ZONE = 0.92

ROTATED_MARGIN_X = 0.10
ROTATED_MARGIN_Y = 0.07
MAX_ROTATED_FRACTION = 0.20

TABLE_MARGIN = 1.5

# Ambang deduplikasi.
DUPLICATE_IOU_THRESHOLD = 0.90
DUPLICATE_CONTAINMENT_THRESHOLD = 0.90
DUPLICATE_TEXT_SIMILARITY = 0.50

SOFT_REASONS = {
    "baris_kurang_dari_3",
    "sel_terisi_kurang_dari_6",
    "area_tabel_terlalu_pendek",
    "area_hampir_seluruh_halaman",
    "terlalu_banyak_sel_satu_karakter",
}

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

NUMERIC_PATTERN = re.compile(
    r"^[\s(+-]*\d[\d.,/%\- ]*[\s)]*$"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFC", str(value))

    for src, dst in LIGATURES.items():
        text = text.replace(src, dst)

    text = text.translate(ZERO_WIDTH)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)

    return text.strip()


def clean_cell(value: Any) -> str:
    text = normalize_text(value)

    if text in {"–", "—", "−", "‐", "‑"}:
        return "-"

    return text


def clean_table_rows(
    raw_rows: list[list[Any]],
) -> tuple[list[list[str]], dict[str, int]]:

    diagnostics = {
        "removed_empty_rows": 0,
        "removed_empty_columns": 0,
        "removed_repeated_rows": 0,
    }

    if not raw_rows:
        return [], diagnostics

    rows = [
        [clean_cell(cell) for cell in row]
        for row in raw_rows
    ]

    max_columns = max(
        (len(row) for row in rows),
        default=0,
    )

    if max_columns == 0:
        return [], diagnostics

    for row in rows:
        row.extend([""] * (max_columns - len(row)))

    nonempty = []

    for row in rows:
        if not any(cell for cell in row):
            diagnostics["removed_empty_rows"] += 1
        else:
            nonempty.append(row)

    if not nonempty:
        return [], diagnostics

    keep_columns = [
        col
        for col in range(max_columns)
        if any(row[col] for row in nonempty)
    ]

    diagnostics["removed_empty_columns"] = (
        max_columns - len(keep_columns)
    )

    rows = [
        [row[col] for col in keep_columns]
        for row in nonempty
    ]

    return rows, diagnostics


# ============================================================
# TABLE QUALITY
# ============================================================

def table_quality(
    rows: list[list[str]],
    rect: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
) -> dict[str, Any]:

    row_count = len(rows)

    col_count = max(
        (len(row) for row in rows),
        default=0,
    )

    cells = [
        cell
        for row in rows
        for cell in row
        if cell.strip()
    ]

    nonempty_count = len(cells)

    x1, y1, x2, y2 = rect

    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)

    page_area = max(
        1.0,
        page_width * page_height,
    )

    bbox_area_ratio = (
        bbox_width * bbox_height
    ) / page_area

    single_char_ratio = (
        sum(
            len(cell.strip()) <= 1
            for cell in cells
        ) / nonempty_count
        if nonempty_count else 1.0
    )

    numeric_count = sum(
        bool(NUMERIC_PATTERN.match(cell.strip()))
        for cell in cells
    )

    numeric_ratio = (
        numeric_count / nonempty_count
        if nonempty_count else 0.0
    )

    reasons = []

    if row_count < 3:
        reasons.append("baris_kurang_dari_3")

    if col_count < 2:
        reasons.append("kolom_kurang_dari_2")

    if nonempty_count < 6:
        reasons.append("sel_terisi_kurang_dari_6")

    if bbox_width < page_width * 0.12:
        reasons.append("area_tabel_terlalu_sempit")

    if bbox_height < page_height * 0.04:
        reasons.append("area_tabel_terlalu_pendek")

    if bbox_area_ratio > 0.88:
        reasons.append("area_hampir_seluruh_halaman")

    if single_char_ratio > 0.55:
        reasons.append("terlalu_banyak_sel_satu_karakter")

    needs_review = (
        numeric_ratio < 0.05
        or single_char_ratio > 0.20
        or bbox_area_ratio > 0.70
    )

    return {
        "accepted": len(reasons) == 0,
        "needs_review": needs_review,
        "rejection_reasons": reasons,
        "row_count": row_count,
        "column_count": col_count,
        "nonempty_cells": nonempty_count,
        "numeric_cell_ratio": round(numeric_ratio, 4),
        "single_character_cell_ratio": round(
            single_char_ratio, 4
        ),
        "bbox_area_ratio": round(bbox_area_ratio, 4),
    }


def is_obvious_page_noise(
    table: dict[str, Any],
) -> bool:

    quality = table.get("quality", {})
    classification = table.get("classification", {})
    features = classification.get("features", {})

    bbox_ratio = quality.get("bbox_area_ratio", 0)
    single_ratio = quality.get(
        "single_character_cell_ratio", 0
    )
    nonempty = quality.get("nonempty_cells", 0)

    noise_cells = features.get("noise_cells", 0)
    has_caption = features.get("caption", False)
    has_colnum = features.get("colnum_row", False)

    return bool(
        bbox_ratio >= 0.95
        and single_ratio >= 0.85
        and nonempty <= 35
        and noise_cells >= 20
        and not has_caption
        and not has_colnum
    )


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def clip_rect(
    rect: pymupdf.Rect,
    page: pymupdf.Page,
) -> tuple[float, float, float, float]:

    rect.normalize()

    clipped = rect & page.rect

    if not clipped.is_empty:
        rect = clipped

    return (
        float(rect.x0),
        float(rect.y0),
        float(rect.x1),
        float(rect.y1),
    )


def camelot_bbox_to_pymupdf(
    bbox: tuple[float, float, float, float],
    page: pymupdf.Page,
) -> tuple[float, float, float, float]:

    rect = pymupdf.Rect(*bbox)

    try:
        matrix = (
            page.transformation_matrix
            * page.rotation_matrix
        )

        rect = rect * matrix

    except AttributeError:
        height = page.rect.height
        x1, y1, x2, y2 = bbox

        rect = pymupdf.Rect(
            x1,
            height - y2,
            x2,
            height - y1,
        )

    return clip_rect(rect, page)


def pdfplumber_bbox_to_pymupdf(
    bbox: tuple[float, float, float, float],
    page: pymupdf.Page,
    pdfplumber_height: float,
) -> tuple[float, float, float, float]:

    x0, top, x1, bottom = bbox

    # pdfplumber menggunakan origin kiri-atas.
    # Ubah menjadi koordinat PDF kiri-bawah.
    pdf_rect = pymupdf.Rect(
        x0,
        pdfplumber_height - bottom,
        x1,
        pdfplumber_height - top,
    )

    matrix = (
        page.transformation_matrix
        * page.rotation_matrix
    )

    rect = pdf_rect * matrix

    return clip_rect(rect, page)


def word_in_any_rect(
    word: tuple,
    rects: list[tuple],
) -> bool:

    x0, y0, x1, y1 = word[:4]

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    return any(
        rx0 - TABLE_MARGIN <= cx <= rx1 + TABLE_MARGIN
        and ry0 - TABLE_MARGIN <= cy <= ry1 + TABLE_MARGIN
        for rx0, ry0, rx1, ry1 in rects
    )


# ============================================================
# SANITIZE PDF
# ============================================================

def _in_margin(
    bbox: tuple,
    width: float,
    height: float,
) -> bool:

    x0, y0, x1, y1 = bbox

    return (
        x1 < width * ROTATED_MARGIN_X
        or x0 > width * (1 - ROTATED_MARGIN_X)
        or y1 < height * ROTATED_MARGIN_Y
        or y0 > height * (1 - ROTATED_MARGIN_Y)
    )


def sanitize_pdf(
    src_path: Path,
    dst_path: Path,
) -> int:

    removed_total = 0

    with pymupdf.open(src_path) as doc:

        for page in doc:

            width = page.rect.width
            height = page.rect.height

            raw = page.get_text("rawdict")
            margin_boxes = []
            total_chars = 0

            for block in raw["blocks"]:

                if block.get("type") != 0:
                    continue

                for line in block["lines"]:

                    is_rotated = abs(line["dir"][1]) > 0.2

                    in_margin = (
                        is_rotated
                        and _in_margin(
                            line["bbox"],
                            width,
                            height,
                        )
                    )

                    for span in line["spans"]:

                        for char in span["chars"]:

                            total_chars += 1

                            if in_margin:
                                margin_boxes.append(
                                    char["bbox"]
                                )

            if not margin_boxes:
                continue

            if (
                total_chars
                and len(margin_boxes) / total_chars
                > MAX_ROTATED_FRACTION
            ):
                continue

            for box in margin_boxes:

                rect = pymupdf.Rect(box)
                rect = rect + (0.5, 0.5, -0.5, -0.5)

                if not rect.is_empty:
                    page.add_redact_annot(rect)

            try:
                page.apply_redactions(
                    images=0,
                    graphics=0,
                )
            except TypeError:
                page.apply_redactions()

            removed_total += len(margin_boxes)

        if removed_total:
            doc.save(
                dst_path,
                garbage=3,
                deflate=True,
            )

    return removed_total


# ============================================================
# PAGE LINES
# ============================================================

def read_page_lines(page) -> list[dict[str, Any]]:

    lines = []

    for block in page.get_text("dict")["blocks"]:

        if block.get("type") != 0:
            continue

        for line in block["lines"]:

            if abs(line["dir"][1]) > 0.2:
                continue

            text = normalize_text(
                "".join(
                    span["text"]
                    for span in line["spans"]
                )
            )

            if not text:
                continue

            x0, y0, x1, y1 = line["bbox"]

            lines.append({
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            })

    lines.sort(
        key=lambda item: (
            item["y0"],
            item["x0"],
        )
    )

    return lines


# ============================================================
# HEADER / FOOTER DETECTION
# ============================================================

PAGE_NUMBER_RE = re.compile(
    r"^(halaman\s*)?[-–\s]*(\d{1,4}|[ivxlcdm]{1,6})[-–\s]*$",
    re.IGNORECASE,
)

CAPTION_START_RE = re.compile(
    r"^(tabel|table|gambar|figure|grafik|lampiran)\b",
    re.IGNORECASE,
)


def detect_running_rects(
    pdf_path: Path,
) -> dict[int, list[tuple]]:

    per_page: dict[
        int,
        list[tuple[str, str, tuple]]
    ] = {}

    with pymupdf.open(pdf_path) as doc:

        page_count = len(doc)

        for page_no, page in enumerate(doc, start=1):

            height = page.rect.height
            found = []

            for line in read_page_lines(page):

                in_header = (
                    line["y1"] < height * HEADER_ZONE
                )

                in_footer = (
                    line["y0"] > height * FOOTER_ZONE
                )

                if not (in_header or in_footer):
                    continue

                text = line["text"]

                if CAPTION_START_RE.match(text):
                    continue

                key = re.sub(
                    r"\d+",
                    "#",
                    text.lower(),
                )

                rect = (
                    line["x0"],
                    line["y0"],
                    line["x1"],
                    line["y1"],
                )

                found.append((key, text, rect))

            per_page[page_no] = found

    counts: Counter[str] = Counter()

    for found in per_page.values():
        counts.update({
            key
            for key, _, _ in found
        })

    threshold = max(
        3,
        int(page_count * 0.30),
    )

    result: dict[int, list[tuple]] = {}

    for page_no, found in per_page.items():

        rects = []

        for key, text, rect in found:

            repeated = counts[key] >= threshold

            if repeated or PAGE_NUMBER_RE.match(text):
                rects.append(rect)

        if rects:
            result[page_no] = rects

    return result


# ============================================================
# EXTRACTOR 1: CAMELOT
# ============================================================

def camelot_candidates(
    pdf_path: Path,
    flavor: str,
) -> list[dict[str, Any]]:

    kwargs: dict[str, Any] = {
        "pages": "all",
        "flavor": flavor,
        "suppress_stdout": True,
        "split_text": True,
    }

    if flavor == "lattice":
        kwargs["line_scale"] = CAMELOT_LINE_SCALE
        kwargs["copy_text"] = ["h", "v"]

    found_tables = camelot.read_pdf(
        str(pdf_path),
        **kwargs,
    )

    candidates = []

    with pymupdf.open(pdf_path) as doc:

        for table in found_tables:

            page_no = int(table.page)
            page = doc[page_no - 1]

            bbox = tuple(
                float(v)
                for v in table._bbox
            )

            candidates.append({
                "page": page_no,
                "engine": "camelot",
                "raw_rows": table.df.values.tolist(),
                "bbox": list(bbox),
                "rect": camelot_bbox_to_pymupdf(
                    bbox,
                    page,
                ),
                "accuracy": float(
                    table.parsing_report.get(
                        "accuracy",
                        0,
                    )
                ),
                "whitespace": float(
                    table.parsing_report.get(
                        "whitespace",
                        0,
                    )
                ),
            })

    return candidates


# ============================================================
# EXTRACTOR 2: PDFPLUMBER
# ============================================================

def pdfplumber_candidates(
    pdf_path: Path,
) -> list[dict[str, Any]]:

    candidates = []

    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 5,
        "edge_min_length": 3,
        "text_tolerance": 3,
    }

    with pdfplumber.open(pdf_path) as pdf:

        with pymupdf.open(pdf_path) as doc:

            for page_index, pdf_page in enumerate(pdf.pages):

                page_no = page_index + 1
                mupdf_page = doc[page_index]

                try:
                    found_tables = pdf_page.find_tables(
                        table_settings=table_settings
                    )

                except Exception as exc:

                    print(
                        f"      pdfplumber gagal "
                        f"di halaman {page_no}: {exc}",
                        file=sys.stderr,
                    )

                    continue

                for table in found_tables:

                    raw_rows = table.extract()

                    if not raw_rows:
                        continue

                    bbox = tuple(
                        float(v)
                        for v in table.bbox
                    )

                    rect = pdfplumber_bbox_to_pymupdf(
                        bbox,
                        mupdf_page,
                        pdf_page.height,
                    )

                    candidates.append({
                        "page": page_no,
                        "engine": "pdfplumber",
                        "raw_rows": raw_rows,
                        "bbox": list(bbox),
                        "rect": rect,
                        "accuracy": 0.0,
                        "whitespace": 0.0,
                    })

    return candidates


# ============================================================
# EXTRACTOR 3: PYMUPDF
# ============================================================

def pymupdf_candidates(
    pdf_path: Path,
) -> list[dict[str, Any]]:

    candidates = []

    with pymupdf.open(pdf_path) as doc:

        for page_no, page in enumerate(doc, start=1):

            try:
                found = page.find_tables()

            except Exception as exc:

                print(
                    f"      find_tables gagal "
                    f"di halaman {page_no}: {exc}",
                    file=sys.stderr,
                )

                continue

            for table in found.tables:

                rect = tuple(
                    float(v)
                    for v in table.bbox
                )

                candidates.append({
                    "page": page_no,
                    "engine": "pymupdf",
                    "raw_rows": table.extract(),
                    "bbox": list(rect),
                    "rect": rect,
                    "accuracy": 0.0,
                    "whitespace": 0.0,
                })

    return candidates


# ============================================================
# CANDIDATE SCORING
# ============================================================

def candidate_text(
    candidate: dict[str, Any],
) -> str:

    rows = candidate.get("raw_rows", [])

    return " ".join(
        normalize_text(cell).lower()
        for row in rows
        for cell in row
        if normalize_text(cell)
    )


def candidate_content_similarity(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
) -> float:

    text_a = candidate_text(candidate_a)
    text_b = candidate_text(candidate_b)

    if not text_a or not text_b:
        return 0.0

    return SequenceMatcher(
        None,
        text_a,
        text_b,
    ).ratio()

def rect_containment(
    rect_a: tuple[float, float, float, float],
    rect_b: tuple[float, float, float, float],
) -> float:

    ax0, ay0, ax1, ay1 = rect_a
    bx0, by0, bx1, by1 = rect_b

    # Hitung luas masing-masing rectangle.
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)

    if area_a == 0 or area_b == 0:
        return 0.0

    # Hitung luas irisan kedua rectangle.
    intersection_width = max(
        0.0,
        min(ax1, bx1) - max(ax0, bx0),
    )

    intersection_height = max(
        0.0,
        min(ay1, by1) - max(ay0, by0),
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    # Bandingkan irisan dengan rectangle yang lebih kecil.
    return intersection_area / min(area_a, area_b)

def is_partial_table_duplicate(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
) -> bool:

    rect_a = tuple(candidate_a["rect"])
    rect_b = tuple(candidate_b["rect"])

    containment = rect_containment(
        rect_a,
        rect_b,
    )

    # Ambil ukuran masing-masing bounding box.
    width_a = max(0.0, rect_a[2] - rect_a[0])
    height_a = max(0.0, rect_a[3] - rect_a[1])

    width_b = max(0.0, rect_b[2] - rect_b[0])
    height_b = max(0.0, rect_b[3] - rect_b[1])

    if min(width_a, width_b, height_a, height_b) <= 0:
        return False

    # Perbandingan ukuran kandidat.
    width_ratio = min(width_a, width_b) / max(width_a, width_b)
    height_ratio = min(height_a, height_b) / max(height_a, height_b)

    # Kandidat kecil harus memiliki lebar yang hampir sama,
    # tetapi tinggi yang jauh lebih kecil.
    return (
        containment >= 0.90
        and width_ratio >= 0.85
        and height_ratio <= 0.30
    )

def candidate_score(
    candidate: dict[str, Any],
    page_size: tuple[float, float],
) -> float:

    clean_rows, _ = clean_table_rows(
        candidate.get("raw_rows", [])
    )

    quality = table_quality(
        clean_rows,
        tuple(candidate["rect"]),
        page_size[0],
        page_size[1],
    )

    row_count = quality["row_count"]
    numeric_ratio = quality["numeric_cell_ratio"]
    single_ratio = quality[
        "single_character_cell_ratio"
    ]

    score = (
        min(row_count, 40) * 2
        + numeric_ratio * 20
        - single_ratio * 15
    )

    if candidate["engine"] == "camelot":
        score += (
            candidate.get("accuracy", 0) * 0.05
        )

    # Sedikit penalti untuk kandidat yang memiliki struktur
    # terlalu banyak sel kosong.
    rows = clean_rows

    if rows:
        total_cells = sum(len(row) for row in rows)
        filled_cells = sum(
            bool(cell.strip())
            for row in rows
            for cell in row
        )

        if total_cells:
            empty_ratio = (
                total_cells - filled_cells
            ) / total_cells

            score -= empty_ratio * 10

    return score


# ============================================================
# DUPLICATE GROUPING
# ============================================================

def are_duplicate_candidates(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
) -> bool:

    if candidate_a["page"] != candidate_b["page"]:
        return False

    rect_a = tuple(candidate_a["rect"])
    rect_b = tuple(candidate_b["rect"])

    overlap = rect_iou(
        rect_a,
        rect_b,
    )

    containment = rect_containment(
        rect_a,
        rect_b,
    )

    similarity = candidate_content_similarity(
        candidate_a,
        candidate_b,
    )

    # Duplikasi dengan area yang hampir sama.
    same_area_duplicate = (
        overlap >= DUPLICATE_IOU_THRESHOLD
        and similarity >= DUPLICATE_TEXT_SIMILARITY
    )

    # Duplikasi dengan area yang saling mencakup.
    contained_duplicate = (
        containment >= DUPLICATE_CONTAINMENT_THRESHOLD
        and similarity >= DUPLICATE_TEXT_SIMILARITY
    )

    # Deteksi khusus ketika salah satu extractor
    # hanya mengambil sebagian tabel, misalnya header.
    partial_duplicate = is_partial_table_duplicate(
        candidate_a,
        candidate_b,
    )

    return (
        same_area_duplicate
        or contained_duplicate
        or partial_duplicate
    )

def group_candidates(
    candidates: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:

    groups = []

    for candidate in candidates:

        matched_group = None

        for group in groups:

            if any(
                are_duplicate_candidates(
                    candidate,
                    existing,
                )
                for existing in group
            ):
                matched_group = group
                break

        if matched_group is None:
            groups.append([candidate])
        else:
            matched_group.append(candidate)

    return groups


def select_candidates(
    candidates: list[dict[str, Any]],
    page_sizes: dict[int, tuple[float, float]],
) -> tuple[list[dict[str, Any]], int]:

    priority = {
        "camelot": 0,
        "pdfplumber": 1,
        "pymupdf": 2,
    }

    groups = group_candidates(candidates)

    selected = []
    duplicate_count = 0

    for group in groups:

        page_no = group[0]["page"]
        page_size = page_sizes[page_no]

        ranked = sorted(
            group,
            key=lambda candidate: (
                -candidate_score(
                    candidate,
                    page_size,
                ),
                priority.get(
                    candidate["engine"],
                    9,
                ),
            ),
        )

        # Simpan semua kandidat dalam kelompok agar kandidat
        # alternatif masih bisa diperiksa bila kandidat utama
        # gagal validasi kualitas.
        selected.append({
            "candidates": ranked,
            "page": page_no,
        })

        duplicate_count += len(group) - 1

    return selected, duplicate_count


# ============================================================
# HEADER / CAPTION HELPERS
# ============================================================
# ============================================================
# DOCUMENT METADATA
# ============================================================

def extract_document_metadata(pdf_path: Path) -> dict[str, Any]:
    """
    Mengidentifikasi metadata dokumen berdasarkan nama file.

    Contoh:
    kecamatan-kapuas-dalam-angka-2025.pdf

    Hasil:
    kecamatan = Kapuas
    kabupaten = Sanggau
    tahun = 2025
    """

    stem = pdf_path.stem.lower()

    # Identifikasi tahun dari nama file.
    year_match = re.search(
        r"\b(20\d{2})\b",
        stem,
    )

    year = int(year_match.group(1)) if year_match else None

    # Identifikasi nama kecamatan.
    kecamatan_match = re.search(
        r"kecamatan[-_\s]+(.+?)[-_\s]+dalam[-_\s]+angka",
        stem,
        re.IGNORECASE,
    )

    kecamatan = None

    if kecamatan_match:
        kecamatan = kecamatan_match.group(1)

        kecamatan = re.sub(
            r"[-_]+",
            " ",
            kecamatan,
        )

        kecamatan = " ".join(
            word.capitalize()
            for word in kecamatan.split()
        )

    return {
        "kecamatan": kecamatan,
        "kabupaten": "Sanggau",
        "tahun": year,
        "document_type": "Kecamatan Dalam Angka",
        "source": pdf_path.name,
        "document": pdf_path.stem,
    }

def get_page_metadata(
    pdf_path: Path,
) -> tuple[
    dict[int, tuple[float, float]],
    dict[int, list[dict[str, Any]]],
]:

    page_sizes = {}
    page_lines = {}

    with pymupdf.open(pdf_path) as doc:

        for page_no, page in enumerate(doc, start=1):

            page_sizes[page_no] = (
                page.rect.width,
                page.rect.height,
            )

            page_lines[page_no] = read_page_lines(page)

    return page_sizes, page_lines


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_tables(
    pdf_path: Path,
    engine: str,
    flavor: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, list[tuple]],
    dict[int, list[str]],
]:

    tables = []
    reclassified = []
    areas: dict[int, list[tuple]] = {}
    page_hints: dict[int, list[str]] = {}

    # --------------------------------------------------------
    # 1. Jalankan ekstraktor
    # --------------------------------------------------------

    candidates = []

    if engine in ("all", "camelot"):

        print("      Mengekstrak dengan Camelot...")

        candidates.extend(
            camelot_candidates(
                pdf_path,
                flavor,
            )
        )

    if engine in ("all", "pdfplumber"):

        print("      Mengekstrak dengan pdfplumber...")

        candidates.extend(
            pdfplumber_candidates(pdf_path)
        )

    if engine in ("all", "pymupdf"):

        print("      Mengekstrak dengan PyMuPDF...")

        candidates.extend(
            pymupdf_candidates(pdf_path)
        )

    # --------------------------------------------------------
    # 2. Informasi halaman
    # --------------------------------------------------------

    page_sizes, page_lines = get_page_metadata(
        pdf_path
    )

    # --------------------------------------------------------
    # 3. Kelompokkan kandidat duplikat
    # --------------------------------------------------------

    candidate_groups, duplicate_count = select_candidates(
        candidates,
        page_sizes,
    )

    print(
        f"      Kandidat ditemukan: {len(candidates)}"
    )

    print(
        f"      Kandidat duplikat dalam kelompok: "
        f"{duplicate_count}"
    )

    # --------------------------------------------------------
    # 4. Validasi setiap kelompok
    # --------------------------------------------------------

    table_id = 0

    for group in candidate_groups:

        page_no = group["page"]

        page_width, page_height = page_sizes[page_no]

        accepted_record = None
        rejected_records = []

        for cand in group["candidates"]:

            raw_rows = cand["raw_rows"]

            clean_rows, cleaning = clean_table_rows(
                raw_rows
            )

            rect = tuple(cand["rect"])

            quality = table_quality(
                clean_rows,
                rect,
                page_width,
                page_height,
            )

            verdict = classify_table(clean_rows)

            record = {
                "table_id": None,
                "page": page_no,
                "engine": cand["engine"],
                "flavor": (
                    flavor
                    if cand["engine"] == "camelot"
                    else None
                ),
                "bbox": cand["bbox"],
                "bbox_pymupdf": [
                    round(v, 2)
                    for v in rect
                ],
                "raw_rows": [
                    [
                        "" if cell is None else str(cell)
                        for cell in row
                    ]
                    for row in raw_rows
                ],
                "clean_rows": clean_rows,
                "cleaning": cleaning,
                "quality": quality,
                "classification": verdict,
                "accuracy": cand["accuracy"],
                "whitespace": cand["whitespace"],
                "content_type": "table",
            }

            # --------------------------------------------
            # Artefak halaman
            # --------------------------------------------

            if is_obvious_page_noise(record):

                record["status"] = "rejected"
                record["content_type"] = (
                    "rejected_detection"
                )

                rejected_records.append(record)
                continue

            # --------------------------------------------
            # Rescue untuk alasan penolakan lunak
            # --------------------------------------------

            hard_reasons = [
                reason
                for reason in quality["rejection_reasons"]
                if reason not in SOFT_REASONS
            ]

            rescued = (
                not quality["accepted"]
                and not hard_reasons
                and verdict["kind"] == "data_table"
                and verdict["confidence"] in (
                    "high",
                    "medium",
                )
            )

            if rescued:

                quality["accepted"] = True
                quality["needs_review"] = True

                record["rescued_from_quality_filter"] = True

            if not quality["accepted"]:

                record["status"] = "rejected"
                record["content_type"] = (
                    "rejected_detection"
                )

                rejected_records.append(record)
                continue

            # --------------------------------------------
            # Bukan tabel data
            # --------------------------------------------

            if verdict["kind"] != "data_table":

                record["status"] = "not_a_table"
                record["content_type"] = verdict["kind"]

                rejected_records.append(record)

                hints = page_hints.setdefault(
                    page_no,
                    [],
                )

                if verdict["kind"] not in hints:
                    hints.append(verdict["kind"])

                continue

            # --------------------------------------------
            # Kandidat valid ditemukan
            # --------------------------------------------

            expanded, caption, notes = expand_table_rect(
                rect,
                page_lines[page_no],
            )

            record["bbox_expanded"] = [
                round(v, 2)
                for v in expanded
            ]

            record["caption"] = caption
            record["notes"] = notes

            record.update(
                parse_caption(caption or "")
            )

            low_confidence = (
                verdict["confidence"] == "low"
            )

            record["status"] = (
                "needs_review"
                if (
                    quality["needs_review"]
                    or low_confidence
                )
                else "accepted"
            )

            accepted_record = record

            # Kandidat pertama yang lolos validasi
            # adalah kandidat terbaik berdasarkan skor.
            break

        # ----------------------------------------------------
        # 5. Simpan hasil kelompok
        # ----------------------------------------------------

        if accepted_record is not None:

            table_id += 1
            accepted_record["table_id"] = table_id

            tables.append(accepted_record)

            areas.setdefault(
                page_no,
                [],
            ).append(
                tuple(
                    accepted_record["bbox_expanded"]
                )
            )

            # Kandidat alternatif tidak disimpan sebagai
            # tabel aktif karena berada dalam kelompok duplikat.
            for rejected in rejected_records:

                rejected["status"] = "duplicate"
                rejected["content_type"] = (
                    "duplicate_detection"
                )

                reclassified.append(rejected)

        else:

            # Jika tidak ada kandidat yang lolos,
            # simpan semua kandidat untuk pemeriksaan.
            reclassified.extend(rejected_records)

    return (
        tables,
        reclassified,
        areas,
        page_hints,
    )


# ============================================================
# NARRATIVE EXTRACTION
# ============================================================

def extract_narrative(
    pdf_path: Path,
    table_areas: dict[int, list[tuple]],
    table_ids_by_page: dict[int, list[int]],
    page_hints: dict[int, list[str]],
    running_rects: dict[int, list[tuple]],
) -> list[dict[str, Any]]:

    records = []

    with pymupdf.open(pdf_path) as document:

        for page_index, page in enumerate(document):

            page_no = page_index + 1

            words = page.get_text(
                "words",
                sort=True,
            )

            table_rects = table_areas.get(
                page_no,
                [],
            )

            page_running = running_rects.get(
                page_no,
                [],
            )

            blocks: dict[
                int,
                dict[int, list[tuple]]
            ] = {}

            removed_table = 0
            removed_running = 0

            for word in words:

                if (
                    table_rects
                    and word_in_any_rect(
                        word,
                        table_rects,
                    )
                ):

                    removed_table += 1
                    continue

                if (
                    page_running
                    and word_in_any_rect(
                        word,
                        page_running,
                    )
                ):

                    removed_running += 1
                    continue

                blocks.setdefault(
                    word[5],
                    {},
                ).setdefault(
                    word[6],
                    [],
                ).append(word)

            paragraphs = []

            for block_no in sorted(blocks):

                line_texts = []

                for line_no in sorted(blocks[block_no]):

                    line_words = sorted(
                        blocks[block_no][line_no],
                        key=lambda item: item[7],
                    )

                    text = normalize_text(
                        " ".join(
                            word[4]
                            for word in line_words
                        )
                    )

                    if text:
                        line_texts.append(text)

                if line_texts:
                    paragraphs.append(
                        " ".join(line_texts)
                    )

            page_text = "\n\n".join(
                paragraphs
            ).strip()

            records.append({
                "page": page_no,
                "text": page_text,
                "paragraphs": paragraphs,
                "word_count": (
                    len(page_text.split())
                    if page_text else 0
                ),
                "table_ids": table_ids_by_page.get(
                    page_no,
                    [],
                ),
                "words_removed_as_table": removed_table,
                "words_removed_as_header_footer": (
                    removed_running
                ),
                "page_hints": page_hints.get(
                    page_no,
                    [],
                ),
                "content_type": "narrative",
            })

    return records


# ============================================================
# DEBUG OVERLAY
# ============================================================

def write_debug_overlay(
    pdf_path: Path,
    out_path: Path,
    tables: list[dict[str, Any]],
    reclassified: list[dict[str, Any]],
    running_rects: dict[int, list[tuple]],
) -> None:

    with pymupdf.open(pdf_path) as doc:

        for table in tables:

            doc[table["page"] - 1].draw_rect(
                pymupdf.Rect(
                    table["bbox_expanded"]
                ),
                color=(0, 0.6, 0),
                width=1.2,
            )

        for table in reclassified:

            doc[table["page"] - 1].draw_rect(
                pymupdf.Rect(
                    table["bbox_pymupdf"]
                ),
                color=(1, 0, 0),
                width=1,
            )

        for page_no, rects in running_rects.items():

            for rect in rects:

                doc[page_no - 1].draw_rect(
                    pymupdf.Rect(rect),
                    color=(0, 0, 1),
                    width=0.8,
                )

        doc.save(
            out_path,
            garbage=3,
            deflate=True,
        )


# ============================================================
# JSON OUTPUT
# ============================================================

def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# PROCESS ONE PDF
# ============================================================

def process_pdf(
    pdf_path: Path,
    output_root: Path,
    engine: str,
    flavor: str,
    sanitize: bool = True,
    debug: bool = False,
) -> dict[str, Any]:

    print("\n" + "=" * 60)
    print(f"Memproses PDF: {pdf_path.name}")
    print("=" * 60)

    if (
        not pdf_path.exists()
        or pdf_path.suffix.lower() != ".pdf"
    ):

        raise FileNotFoundError(
            f"PDF tidak ditemukan/invalid: {pdf_path}"
        )

    label = (
        f"{flavor}_all"
        if engine == "all"
        else flavor
        if engine == "camelot"
        else engine
    )

    document_output = (
        output_root
        / pdf_path.stem
        / label
    )

    document_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    working_pdf = pdf_path
    rotated_removed = 0

    if sanitize:

        print(
            "[0/4] Menghapus teks rotasi di margin..."
        )

        clean_pdf = (
            document_output
            / "_sanitized.pdf"
        )

        rotated_removed = sanitize_pdf(
            pdf_path,
            clean_pdf,
        )

        if rotated_removed:
            working_pdf = clean_pdf

        print(
            f"      karakter rotasi dihapus: "
            f"{rotated_removed}"
        )

    print(
        "[1/4] Mendeteksi header/footer berulang..."
    )

    running_rects = detect_running_rects(
        working_pdf
    )

    running_lines = sum(
        len(value)
        for value in running_rects.values()
    )

    print(
        f"      baris header/footer ditemukan: "
        f"{running_lines}"
    )

    print(
        f"[2/4] Mengekstrak tabel menggunakan {engine}..."
    )

    (
        tables,
        reclassified,
        table_areas,
        page_hints,
    ) = extract_tables(
        working_pdf,
        engine,
        flavor,
    )

    table_ids_by_page: dict[
        int,
        list[int]
    ] = {}

    for record in tables:

        table_ids_by_page.setdefault(
            record["page"],
            [],
        ).append(
            record["table_id"]
        )

    print(
        "[3/4] Mengekstrak narasi..."
    )

    narratives = extract_narrative(
        working_pdf,
        table_areas,
        table_ids_by_page,
        page_hints,
        running_rects,
    )

    # --------------------------------------------------------
# Tambahkan metadata dokumen ke seluruh hasil ekstraksi
# --------------------------------------------------------

    document_metadata = extract_document_metadata(pdf_path)

    if not document_metadata["kecamatan"]:
        print(
            f"PERINGATAN: Kecamatan tidak dapat diidentifikasi "
            f"dari nama file {pdf_path.name}"
        )

    for record in (
        tables
        + reclassified
        + narratives
    ):
        record.update(document_metadata)

    print(
        "[4/4] Menyimpan hasil..."
    )

    write_jsonl(
        document_output / "narrative.jsonl",
        narratives,
    )

    write_jsonl(
        document_output / "tables.jsonl",
        tables,
    )

    write_jsonl(
        document_output / "rejected_tables.jsonl",
        reclassified,
    )

    if debug:

        overlay = (
            document_output
            / "_debug_overlay.pdf"
        )

        write_debug_overlay(
            working_pdf,
            overlay,
            tables,
            reclassified,
            running_rects,
        )

        print(
            f"      overlay debug: {overlay}"
        )

    with pymupdf.open(pdf_path) as doc:
        page_count = len(doc)

    def count(status: str) -> int:

        return sum(
            record["status"] == status
            for record in reclassified
        )

    engine_counts = dict(
        Counter(
            record["engine"]
            for record in tables
        )
    )

    summary = {
        **document_metadata,
        "engine": engine,
        "flavor": (
            flavor
            if engine in ("camelot", "all")
            else None
        ),
        "page_count": page_count,
        "rotated_characters_removed": rotated_removed,
        "header_footer_lines_removed": running_lines,
        "narrative_pages": len(narratives),
        "nonempty_narrative_pages": sum(
            bool(record["text"])
            for record in narratives
        ),
        "empty_narrative_pages": sum(
            not bool(record["text"])
            for record in narratives
        ),
        "data_tables": len(tables),
        "tables_by_engine": engine_counts,
        "tables_needs_review": sum(
            record["status"] == "needs_review"
            for record in tables
        ),
        "returned_to_narrative": count(
            "not_a_table"
        ),
        "rejected_by_quality": count(
            "rejected"
        ),
        "duplicate_detections": count(
            "duplicate"
        ),
        "output_directory": str(
            document_output
        ),
    }

    write_json(
        document_output / "summary.json",
        summary,
    )

    print("\nSelesai:")
    print(f"  Halaman PDF             : {page_count}")
    print(f"  Karakter rotasi dihapus : {rotated_removed}")
    print(f"  Header/footer dibuang   : {running_lines}")
    print(
        f"  Halaman narasi berisi   : "
        f"{summary['nonempty_narrative_pages']}"
    )
    print(
        f"  Tabel data              : "
        f"{summary['data_tables']}"
    )
    print(
        f"  Tabel perlu diperiksa   : "
        f"{summary['tables_needs_review']}"
    )
    print(
        f"  Dikembalikan ke narasi  : "
        f"{summary['returned_to_narrative']}"
    )
    print(
        f"  Ditolak kualitas        : "
        f"{summary['rejected_by_quality']}"
    )
    print(
        f"  Deteksi ganda           : "
        f"{summary['duplicate_detections']}"
    )
    print(
        f"  Tabel berdasarkan mesin : "
        f"{engine_counts}"
    )
    print(
        f"  Output                  : "
        f"{document_output}"
    )

    return summary


# ============================================================
# BATCH PROCESSING
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Ekstraksi PDF BPS: "
            "Camelot + pdfplumber + PyMuPDF."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=PDF_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--engine",
        choices=[
            "all",
            "camelot",
            "pdfplumber",
            "pymupdf",
        ],
        default="all",
        help=(
            "Mesin deteksi tabel. "
            "'all' menjalankan ketiga ekstraktor."
        ),
    )

    parser.add_argument(
        "--flavor",
        choices=[
            "stream",
            "lattice",
        ],
        default="lattice",
        help=(
            "Flavor Camelot. "
            "Hanya digunakan jika engine all atau camelot."
        ),
    )

    parser.add_argument(
        "--docs",
        default="",
        help=(
            "Filter nama file PDF (substring), dipisah koma. "
            "Contoh: --docs pdf_3240,pdf_4846"
        ),
    )

    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help=(
            "Jangan hapus teks rotasi di margin."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Tulis _debug_overlay.pdf."
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():

        raise FileNotFoundError(
            f"Folder input tidak ditemukan: {args.input}"
        )

    pdf_files = sorted(
        path
        for path in args.input.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == ".pdf"
        )
    )

    filters = [
        item.strip()
        for item in args.docs.split(",")
        if item.strip()
    ]

    if filters:
        pdf_files = [
            path
            for path in pdf_files
            if any(
                item.lower() in path.name.lower()
                for item in filters
            )
        ]

    if not pdf_files:

        if filters:
            print(
                "Tidak ada PDF yang cocok dengan filter: "
                + ", ".join(filters)
            )
        else:
            print(
                f"Tidak ada file PDF di folder: {args.input}"
            )

        return

    print(
        f"Ditemukan {len(pdf_files)} file PDF."
    )

    results = []

    for pdf_path in pdf_files:

        try:

            result = process_pdf(
                pdf_path,
                args.output,
                args.engine,
                args.flavor,
                sanitize=not args.no_sanitize,
                debug=args.debug,
            )

            result["status"] = "success"
            result["error"] = None

        except Exception as exc:

            print(
                f"\nGagal memproses {pdf_path.name}: {exc}",
                file=sys.stderr,
            )

            result = {
                "source": pdf_path.name,
                "status": "failed",
                "error": str(exc),
            }

        results.append(result)

    label = (
        f"{args.flavor}_all"
        if args.engine == "all"
        else args.flavor
        if args.engine == "camelot"
        else args.engine
    )

    write_json(
        args.output / f"batch_summary_{label}.json",
        {
            "total_pdfs": len(pdf_files),
            "successful": sum(
                result["status"] == "success"
                for result in results
            ),
            "failed": sum(
                result["status"] == "failed"
                for result in results
            ),
            "documents": results,
        },
    )

    print("\n" + "=" * 60)
    print("PROSES BATCH SELESAI")
    print("=" * 60)

    print(
        f"Total PDF : {len(pdf_files)}"
    )

    print(
        f"Berhasil  : "
        f"{sum(r['status'] == 'success' for r in results)}"
    )

    print(
        f"Gagal     : "
        f"{sum(r['status'] == 'failed' for r in results)}"
    )


if __name__ == "__main__":
    main()
