
"""Fungsi bersama untuk cleaner.py, chunking.py, dan rescue_tables.py."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

HEADER_ROWS_MAX = 3

HEADING_RE = re.compile(
    r"^(BAB|SUBBAB|LAMPIRAN)\b|^\d+(?:\.\d+)*\.?\s+\S",
    re.IGNORECASE,
)

NUM_RE = re.compile(r"^[\s(+-]*\d[\d.,/%\-\u2013\u2014 ]*[\s)]*$")

PERIOD_RE = re.compile(
    r"^(?:19|20)\d{2}(?:\s*[/–\-]\s*(?:19|20)?\d{2})?$"
)


# ============================================================
# I/O JSONL
# ============================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada {path}, baris {line_number}: {exc}"
                ) from exc

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# NARASI
# ============================================================

def is_heading(text: str) -> bool:
    text = str(text or "").strip()

    if not text or len(text) > 100:
        return False

    if text[-1] in ".;,:":
        return False

    letters = [c for c in text if c.isalpha()]

    if not letters:
        return False

    upper_ratio = sum(c.isupper() for c in letters) / len(letters)

    return upper_ratio > 0.8 or bool(HEADING_RE.match(text))


# ============================================================
# TABEL
# ============================================================

COLNUM_RE = re.compile(r"(?:\(?\s*\d+\s*\)?\s*)+")


def is_colnum_row(row: list[str]) -> bool:
    """
    Mendeteksi baris nomor kolom:
    (1) (2) (3)
    1   2   3

    Sel gabungan seperti "(2) (3)" juga dikenali. Tanpa tanda kurung,
    minimal tiga nomor diperlukan agar baris data berisi tahun tidak
    ikut terbuang.
    """

    original = [str(c).strip() for c in row if str(c).strip()]

    if len(original) < 2:
        return False

    has_parentheses = any("(" in cell or ")" in cell for cell in original)
    numbers: list[int] = []

    for cell in original:
        if not COLNUM_RE.fullmatch(cell):
            return False

        numbers.extend(int(value) for value in re.findall(r"\d+", cell))

    if not has_parentheses and len(numbers) < 3:
        return False

    if not numbers or numbers[0] not in (0, 1):
        return False

    # Kolom ganda hasil ekstraksi bisa menghasilkan nomor berulang
    # atau melompat, mis. (1) (1) (2) (2) (12) (12) (13).
    return all(
        left <= right
        for left, right in zip(numbers, numbers[1:])
    )


def numeric_ratio(values: list[str]) -> float:
    values = [
        str(v).strip()
        for v in values
        if str(v).strip()
    ]

    if not values:
        return 0.0

    return sum(bool(NUM_RE.fullmatch(v)) for v in values) / len(values)



def extract_column_numbers(
    rows: list[list[str]]
) -> list[str]:
    """
    Mengambil baris nomor kolom tanpa menghilangkan
    posisi sel kosong.
    """
    for row in rows:
        if is_colnum_row(row):
            return [str(cell).strip() for cell in row]

    return []


def split_header(
    rows: list[list[str]]
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Memisahkan header dan data.

    Header dapat memiliki beberapa tingkatan.
    Baris nomor kolom menjadi batas akhir header
    dan tidak dimasukkan ke header maupun data.
    """
    header: list[list[str]] = []
    i = 0

    while i < len(rows):
        row = [str(cell).strip() for cell in rows[i]]

        # Nomor kolom menandakan akhir header.
        if is_colnum_row(row):
            i += 1
            break

        # Batas maksimal tingkatan header.
        if len(header) >= HEADER_ROWS_MAX:
            break

        # Baris numerik dominan menandakan awal data, kecuali baris
        # pertama masih memiliki label teks di kolom pertama
        # (mis. "Jenis Tanaman | 2022 | 2023 | 2024 | 2025").
        first_has_text = bool(row and re.search(r"[A-Za-z]", row[0]))

        # Header bertingkat: baris kedua berisi periode/tahun dengan
        # kolom pertama kosong (mis. Negeri | 2024/2025 | 2025/2026).
        period_cells = [cell for cell in row[1:] if cell]

        if (
            header
            and period_cells
            and all(PERIOD_RE.fullmatch(cell) for cell in period_cells)
        ):
            header.append(row)
            i += 1
            continue

        if numeric_ratio(row[1:]) >= 0.5 and (
            header or not first_has_text
        ):
            break

        header.append(row)
        i += 1

    data = rows[i:]

    # Jika tidak ditemukan data, pertahankan
    # baris pertama sebagai header.
    if not data and len(header) > 1:
        header = header[:1]
        data = rows[1:]

    return header, data