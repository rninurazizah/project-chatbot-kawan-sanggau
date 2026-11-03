"""
Logika murni (tanpa PyMuPDF/Camelot) untuk memisahkan tabel dari narasi.

- classify_table   : tabel data / daftar isi (TOC) / narasi yang salah dideteksi
- parse_caption    : nomor tabel dan penanda "Lanjutan"
- expand_table_rect: perluas area tabel agar mencakup caption dan catatan
- rect_iou         : deteksi deteksi ganda
"""

from __future__ import annotations

import re
from typing import Any

INVISIBLE = re.compile(r"[\u2060\u200b\u200c\u200d\ufeff]")
COLNUM_CELL = re.compile(r"\(\d{1,2}\)")
DOT_LEADER = re.compile(r"(\.\s*){4,}")
DASH_ONLY = re.compile(r"[-–—−.\s]+")
DASH_CHAR = re.compile(r"[-–—−]")
NUMERIC = re.compile(r"^[\s(+-]*\d[\d.,/%\- ]*[\s)]*$")
CAPTION_RE = re.compile(r"^\s*(lanjutan\s+)?tabel\s+(\d+(?:\.\d+)*)", re.IGNORECASE)
NOTE_RE = re.compile(r"^\s*(sumber|catatan|keterangan|note|source)\b", re.IGNORECASE)


def strip_invisible(text: str) -> str:
    return INVISIBLE.sub("", text or "").strip()


def is_dash_cell(cell: str) -> bool:
    return bool(DASH_CHAR.search(cell)) and bool(DASH_ONLY.fullmatch(cell))


def is_noise_cell(cell: str) -> bool:
    """
    Sisa teks vertikal/watermark: sel yang semua token-nya satu karakter
    bukan angka dan bukan tanda strip (mis. 'd', 'o g', '.').
    """
    if NUMERIC.match(cell) or is_dash_cell(cell):
        return False
    return all(len(token) <= 1 for token in cell.split())


# ============================================================
# KLASIFIKASI
# ============================================================

def table_features(rows: list[list[str]]) -> dict[str, Any]:
    clean_rows = [[strip_invisible(c) for c in row] for row in rows]
    all_cells = [c for row in clean_rows for c in row if c]
    cells = [c for c in all_cells if not is_noise_cell(c)]
    n = len(cells)

    colnum_row = any(
        sum(bool(COLNUM_CELL.fullmatch(c)) for c in row) >= 2
        for row in clean_rows
    )

    def ratio(pred) -> float:
        return sum(1 for c in cells if pred(c)) / n if n else 0.0

    caption = any(
        CAPTION_RE.match(c)
        for row in clean_rows[:3]
        for c in row
    )

    return {
        "noise_cells": len(all_cells) - n,
        "row_count": len(clean_rows),
        "column_count": max((len(r) for r in clean_rows), default=0),
        "colnum_row": colnum_row,
        "caption": caption,
        "dot_leader_ratio": round(ratio(lambda c: bool(DOT_LEADER.search(c))), 3),
        "long_cell_ratio": round(ratio(lambda c: len(c.split()) >= 8), 3),
        "data_value_ratio": round(
            ratio(lambda c: bool(NUMERIC.match(c)) or is_dash_cell(c)),
            3,
        ),
    }


def classify_table(rows: list[list[str]]) -> dict[str, Any]:
    """
    kind:
      data_table : tabel data BPS (diterima, dikeluarkan dari narasi)
      toc        : daftar isi / daftar tabel (tetap narasi)
      narrative  : teks biasa yang salah dibaca Camelot sebagai tabel
                   (sampul, kata pengantar, singkatan, dst.; tetap narasi)
    """

    f = table_features(rows)

    def verdict(kind: str, confidence: str, reason: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "confidence": confidence,
            "reason": reason,
            "features": f,
        }

    # 1. Daftar isi / daftar tabel: titik-titik penghubung
    if f["dot_leader_ratio"] >= 0.10:
        return verdict("toc", "high", "titik_penghubung")

    # 2. Baris nomor kolom khas BPS: (1) (2) (3) ...
    if f["colnum_row"]:
        return verdict("data_table", "high", "baris_nomor_kolom")

    # 3. Paragraf: banyak sel panjang
    if f["long_cell_ratio"] >= 0.20:
        return verdict("narrative", "high", "sel_berisi_kalimat_panjang")

    # 4. Ada caption "Tabel x.y.z" dan ada nilai data
    if f["caption"] and f["data_value_ratio"] >= 0.10:
        return verdict("data_table", "medium", "caption_dan_nilai_data")

    # 5. Dominan angka / tanda strip, minimal 3 kolom
    if f["data_value_ratio"] >= 0.30 and f["column_count"] >= 3:
        return verdict("data_table", "medium", "dominan_nilai_data")

    # 6. Hanya caption (judul tabel terpotong)
    if f["caption"]:
        return verdict("data_table", "low", "hanya_caption")

    return verdict("narrative", "medium", "tidak_ada_ciri_tabel")


def parse_caption(text: str) -> dict[str, Any]:
    match = CAPTION_RE.match(text or "")
    if not match:
        return {"table_number": None, "is_continuation": False}
    return {
        "table_number": match.group(2),
        "is_continuation": bool(match.group(1)),
    }


# ============================================================
# AREA TABEL
# ============================================================

def rect_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (
        (ax1 - ax0) * (ay1 - ay0)
        + (bx1 - bx0) * (by1 - by0)
        - inter
    )
    return inter / union if union > 0 else 0.0


def expand_table_rect(
    rect: tuple[float, float, float, float],
    lines: list[dict[str, Any]],
    max_caption_distance: float = 70.0,
    max_note_distance: float = 45.0,
) -> tuple[tuple[float, float, float, float], str, str]:
    """
    Perluas area tabel (koordinat PyMuPDF, y ke bawah) agar mencakup
      - caption di atas tabel ("Tabel x.y.z ..." / "Lanjutan Tabel ...")
      - catatan di bawah tabel ("Sumber:", "Catatan:", ...)
    yang sering berada di luar bbox Camelot sehingga bocor ke narasi.

    lines: [{"text","x0","y0","x1","y1"}] baris horizontal halaman.
    Mengembalikan (rect_baru, teks_caption, teks_catatan).
    """

    x0, y0, x1, y1 = rect

    def cy(line):
        return (line["y0"] + line["y1"]) / 2

    inside = [l for l in lines if y0 <= cy(l) <= y1]
    caption_inside = any(CAPTION_RE.match(l["text"]) for l in inside)

    caption_lines: list[dict[str, Any]] = []
    if not caption_inside:
        above = [
            l for l in lines
            if cy(l) < y0 and y0 - cy(l) <= max_caption_distance
        ]
        above.sort(key=cy, reverse=True)  # terdekat dengan tabel dulu

        for index, line in enumerate(above):
            if CAPTION_RE.match(line["text"]):
                caption_lines = sorted(above[: index + 1], key=cy)
                break

    note_lines: list[dict[str, Any]] = []
    below = [
        l for l in lines
        if cy(l) > y1 and cy(l) - y1 <= max_note_distance
    ]
    below.sort(key=cy)

    start = next(
        (i for i, l in enumerate(below) if NOTE_RE.match(l["text"])),
        None,
    )

    if start is not None:
        note_lines = [below[start]]
        for line in below[start + 1:]:
            previous = note_lines[-1]
            height = max(1.0, previous["y1"] - previous["y0"])
            if line["y0"] - previous["y1"] <= height * 1.0:
                note_lines.append(line)
            else:
                break

    extra = caption_lines + note_lines

    if extra:
        x0 = min([x0] + [l["x0"] for l in extra])
        x1 = max([x1] + [l["x1"] for l in extra])
        y0 = min([y0] + [l["y0"] for l in caption_lines])
        y1 = max([y1] + [l["y1"] for l in note_lines])

    caption_text = " ".join(l["text"] for l in caption_lines)
    notes_text = " ".join(l["text"] for l in note_lines)

    if caption_inside and not caption_text:
        caption_text = " ".join(
            l["text"] for l in inside[:3] if l["text"]
        )

    return (x0, y0, x1, y1), caption_text, notes_text
