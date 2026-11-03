from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    NUM_RE,
    extract_column_numbers,
    is_colnum_row,
    is_heading,
    numeric_ratio,
    read_jsonl,
    split_header,
    write_jsonl,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "extracted"

# Urutan proses:
#   1. filter halaman      (rentang --skip-pages, daftar isi)
#   2. cleaning tabel      (teks, baris/kolom kosong, header, catatan)
#   3. dedup lintas engine (satu hasil terbaik per area tabel)
#   4. gabung tabel lanjutan antarhalaman
#   5. filter tabel tanpa informasi
#   6. cleaning narasi     (teks, kata terpotong, paragraf terputus, sampah)


# ============================================================
# POLA
# ============================================================

CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
LEADER_RE = re.compile(r"(?:\.\s?){4,}|…{2,}")            # titik-titik penuntun
HYPHEN_RE = re.compile(r"([A-Za-z]{2,})-\s+([a-z]{2,})")  # "pen- duduk"
SPACE_PUNCT_RE = re.compile(r"\s+([,.;:!?%)])")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.I)
BLANK_PAGE_RE = re.compile(r"sengaja\s+dikosongkan", re.I)
NUM_TOKEN_RE = re.compile(r"^[(+\-]?\d[\d.,/%\-]*\)?$|^[-–…]+$")
ELLIPSIS_CELL_RE = re.compile(r"^(?:\.\s*){2,}$|^…+$")
NOTE_ROW_RE = re.compile(r"^(sumber|source|catatan|note|keterangan)\b", re.I)
CAPTION_RE = re.compile(r"(?:Tabel|Table)\s+(\d+(?:\.\d+)*)", re.I)

TERMINAL = ('.', '!', '?', ':', ';', ')', '"', '”', "'", '’')
DASHES = {"–", "—", "−", "‐", "‑"}
# Placeholder BPS: '-' = nol/tidak ada, '...' = data belum tersedia, 'x' = tidak relevan
PLACEHOLDERS = {"-", "...", "x", "X", "*", "**"}

LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
}
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


# ============================================================
# TEKS UMUM
# ============================================================

def normalize_basic(text: str) -> str:
    """NFC + ligatur + spasi aneh. Tidak menyentuh angka maupun tanda baca."""
    text = unicodedata.normalize("NFC", text)
    for src, dst in LIGATURES.items():
        text = text.replace(src, dst)
    text = text.translate(ZERO_WIDTH).replace("\u00a0", " ")
    return text


def fix_hyphenation(text: str, report: Counter) -> str:
    """
    "pen- duduk" -> "penduduk". Hyphen sah tidak diikuti spasi ("anak-anak"),
    jadi pola "kata- kata" hanya muncul dari pemenggalan di akhir baris.
    Reduplikasi ("anak- anak") dijaga: hasilnya "anak-anak".
    """

    def join(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        if left.lower() == right.lower():
            return f"{left}-{right}"
        report["hyphen_joined"] += 1
        return left + right

    return HYPHEN_RE.sub(join, text)


def clean_text(text: str, report: Counter) -> str:
    text = normalize_basic(text)
    text = CTRL_RE.sub("", text)
    text = LEADER_RE.sub(" ", text)
    text = fix_hyphenation(text, report)
    text = SPACE_PUNCT_RE.sub(r"\1", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def parse_pages(spec: str) -> set[int]:
    """'1-22,30,35-36' -> {1..22, 30, 35, 36}"""
    pages: set[int] = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        if "-" in part:
            start, end = part.split("-", 1)
            pages.update(range(int(start), int(end) + 1))
        else:
            pages.add(int(part))
    return pages


# ============================================================
# 1. FILTER HALAMAN
# ============================================================

def filter_pages(
    narrative: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    skip_pages: set[int],
    report: Counter,
    dropped_tables: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    toc_pages = {
        rec["page"] for rec in narrative
        if "toc" in (rec.get("page_hints") or [])
    }
    excluded = skip_pages | toc_pages
    report["halaman_dikecualikan_rentang"] = len(skip_pages & {r["page"] for r in narrative})
    report["halaman_dikecualikan_daftar_isi"] = len(toc_pages - skip_pages)

    kept_narrative = [r for r in narrative if r["page"] not in excluded]
    kept_tables = []
    for table in tables:
        if table["page"] in excluded:
            report["tabel_dibuang_halaman_dikecualikan"] += 1
            dropped_tables.append(drop_record(table, "halaman_dikecualikan"))
        else:
            kept_tables.append(table)
    return kept_narrative, kept_tables


def drop_record(table: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "page": table.get("page"),
        "table_id": table.get("table_id"),
        "engine": table.get("engine"),
        "caption": table.get("caption"),
        "reason": reason,
    }


# ============================================================
# 2. CLEANING TABEL
# Nilai statistik tidak diubah: sel yang berbentuk angka hanya dirapikan
# spasinya. Perbaikan tanda baca dan kata terpotong hanya untuk sel teks.
# ============================================================

def clean_table_cell(value: Any, report: Counter) -> str:
    text = normalize_basic("" if value is None else str(value))
    text = re.sub(r"\s*\n\s*", " ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    # "–3" adalah tanda nol dengan nomor catatan kaki, bukan angka negatif.
    text = re.sub(r"^[\u2013\u2014]\s*[1-9]$", "-", text)
    if text in DASHES:
        return "-"
    if ELLIPSIS_CELL_RE.match(text):
        return "..."
    if NUM_RE.match(text):
        return text                      # angka: jangan diubah
    return clean_text(text, report)


def as_note_list(notes: Any) -> list[str]:
    if not notes:
        return []
    if isinstance(notes, str):
        return [notes]
    return [str(n) for n in notes]


def drop_empty(
    rows: list[list[str]],
    report: Counter
) -> list[list[str]]:
    """
    Menghapus baris dan kolom yang sepenuhnya kosong.
    """
    cleaned_rows, _ = drop_empty_with_indices(rows, report)
    return cleaned_rows


def drop_empty_with_indices(
    rows: list[list[str]],
    report: Counter
) -> tuple[list[list[str]], list[int]]:
    """
    Menghapus baris dan kolom kosong.

    Mengembalikan:
    1. Baris yang sudah dibersihkan.
    2. Indeks kolom asli yang masih dipertahankan.
    """
    if not rows:
        return [], []

    ncols = max(len(row) for row in rows)

    # Samakan jumlah kolom.
    rows = [
        row + [""] * (ncols - len(row))
        for row in rows
    ]

    # Hapus baris kosong.
    before = len(rows)
    rows = [row for row in rows if any(row)]

    report["baris_kosong_dibuang"] += before - len(rows)

    if not rows:
        return [], []

    # Cari indeks kolom yang memiliki setidaknya
    # satu nilai pada seluruh baris.
    keep_indices = [
        j
        for j in range(ncols)
        if any(row[j] for row in rows)
    ]

    report["kolom_kosong_dibuang"] += (
        ncols - len(keep_indices)
    )

    # Susun ulang tabel berdasarkan indeks tersebut.
    cleaned_rows = [
        [row[j] for j in keep_indices]
        for row in rows
    ]

    return cleaned_rows, keep_indices
def clean_one_table(
    table: dict[str, Any],
    report: Counter
) -> dict[str, Any] | None:

    # ========================================================
    # 1. CLEANING SEL
    # ========================================================

    source_rows = [
        [clean_table_cell(c, report) for c in row]
        for row in table.get("clean_rows") or table.get("rows") or []
    ]

    if not source_rows:
        return None

    # Ambil nomor kolom sebelum kolom kosong dihapus.
    source_column_numbers = extract_column_numbers(source_rows)

    # ========================================================
    # 2. HAPUS BARIS DAN KOLOM KOSONG
    # ========================================================

    rows, kept_indices = drop_empty_with_indices(
        source_rows,
        report
    )

    if not rows:
        return None

    # Selaraskan nomor kolom dengan kolom yang dipertahankan.
    column_numbers = []

    if source_column_numbers:
        column_numbers = [
            source_column_numbers[j]
            if j < len(source_column_numbers)
            else ""
            for j in kept_indices
        ]

    # ========================================================
    # 3. PISAHKAN HEADER DAN DATA
    # ========================================================

    notes = as_note_list(table.get("notes"))

    header, data = split_header(rows)

    header_keys = {
        tuple(cell.lower() for cell in row)
        for row in header
    }

    body: list[list[str]] = []

    for row in data:
        key = tuple(cell.lower() for cell in row)

        # Buang header yang berulang.
        if key in header_keys:
            report["tabel_header_berulang_dibuang"] += 1
            continue

        # Buang baris nomor kolom jika masih ditemukan.
        if is_colnum_row(row):
            report["tabel_nomor_kolom_dibuang"] += 1
            continue

        # Pindahkan catatan/sumber dari baris tabel.
        if (
            row[0]
            and not any(row[1:])
            and NOTE_ROW_RE.match(row[0])
        ):
            notes.append(row[0])
            report["tabel_catatan_dipindah"] += 1
            continue

        body.append(row)

    # ========================================================
    # 4. REKONSTRUKSI LABEL BARIS
    # ========================================================

    merged: list[list[str]] = []
    i = 0

    while i < len(body):
        row = body[i]

        if (
            row[0]
            and not any(row[1:])
            and i + 1 < len(body)
            and body[i + 1][0][:1].islower()
        ):
            nxt = list(body[i + 1])

            nxt[0] = f"{row[0]} {nxt[0]}"

            merged.append(nxt)

            report["tabel_label_baris_disambung"] += 1

            i += 2
            continue

        merged.append(row)
        i += 1

    # ========================================================
    # 5. CLEANING STRUKTUR FINAL
    # ========================================================

    all_rows, final_indices = drop_empty_with_indices(
        header + merged,
        report
    )

    if not all_rows:
        return None

    # Sesuaikan kembali metadata nomor kolom.
    if column_numbers:
        column_numbers = [
            column_numbers[j]
            for j in final_indices
            if j < len(column_numbers)
        ]

    # ========================================================
    # 6. SIMPAN HASIL
    # ========================================================

    out = dict(table)

    out["clean_rows"] = all_rows
    out["header_row_count"] = min(
        len(header),
        len(all_rows)
    )
    out["notes"] = notes

    if column_numbers:
        out["column_numbers"] = column_numbers

    # ========================================================
    # 7. DETEKSI KUALITAS TABEL
    # ========================================================

    cells = [
        cell
        for row in all_rows[out["header_row_count"]:]
        for cell in row
    ]

    if cells:
        empty_ratio = sum(
            not cell for cell in cells
        ) / len(cells)

        if empty_ratio > 0.5:
            reasons = out.setdefault("review_reasons", [])

            if "lebih_dari_50persen_sel_kosong" not in reasons:
                reasons.append("lebih_dari_50persen_sel_kosong")

            if out.get("status") == "accepted":
                out["status"] = "needs_review"

    return out
# ============================================================
# 3. DEDUP LINTAS ENGINE
# ============================================================

def caption_number(table: dict[str, Any]) -> str | None:
    match = CAPTION_RE.search((table.get("caption") or "").strip())
    return match.group(1) if match else None


def rect_of(table: dict[str, Any]) -> list[float] | None:
    rect = table.get("bbox_pymupdf")
    return list(rect) if rect and len(rect) == 4 else None


def overlap_metrics(a: list[float], b: list[float]) -> tuple[float, float]:
    """(IoU, irisan / luas yang lebih kecil)"""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (inter / union if union else 0.0, inter / smaller if smaller else 0.0)


def same_table(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ra, rb = rect_of(a), rect_of(b)
    if ra and rb:
        iou, contain = overlap_metrics(ra, rb)
        if iou > 0.5 or contain > 0.8:
            return True
    na, nb = caption_number(a), caption_number(b)
    return bool(na and na == nb)


def table_score(table: dict[str, Any]) -> float:
    """Makin banyak sel terisi, makin banyak angka, makin baik."""
    rows = table["clean_rows"]
    data = rows[table.get("header_row_count", 0):]
    cells = [c for r in data for c in r if c]
    if not cells:
        return 0.0
    total = sum(len(r) for r in data) or 1
    score = len(cells) * (0.3 + numeric_ratio(cells)) * (0.5 + len(cells) / total)
    if table.get("status") == "needs_review":
        score *= 0.8
    if table.get("quality", {}).get("single_character_cell_ratio", 0) > 0.20:
        score *= 0.5
    if (table.get("caption") or "").strip():
        score *= 1.2
    return score


def dedupe_tables(
    tables: list[dict[str, Any]],
    prefer: list[str],
    report: Counter,
    dropped: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_page: dict[int, list[dict[str, Any]]] = {}
    for t in tables:
        by_page.setdefault(t["page"], []).append(t)

    kept: list[dict[str, Any]] = []
    for page_tables in by_page.values():
        parent = list(range(len(page_tables)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(page_tables)):
            for j in range(i + 1, len(page_tables)):
                if same_table(page_tables[i], page_tables[j]):
                    parent[find(j)] = find(i)

        clusters: dict[int, list[dict[str, Any]]] = {}
        for i, t in enumerate(page_tables):
            clusters.setdefault(find(i), []).append(t)

        for cluster in clusters.values():
            if len(cluster) == 1:
                kept.append(cluster[0])
                continue

            def rank(t: dict[str, Any]) -> tuple:
                engine = t.get("engine")
                priority = -prefer.index(engine) if engine in prefer else -len(prefer)
                return (table_score(t), priority, len(t["clean_rows"]))

            winner = max(cluster, key=rank)
            winner = dict(winner)
            winner["dedup"] = {
                "candidates": sorted({str(t.get("engine")) for t in cluster}),
                "dropped": len(cluster) - 1,
            }
            # caption/catatan dari kandidat lain bila pemenang tidak punya
            for other in cluster:
                if not (winner.get("caption") or "").strip() and (other.get("caption") or "").strip():
                    winner["caption"] = other["caption"]
                if not winner.get("notes") and other.get("notes"):
                    winner["notes"] = other["notes"]
            kept.append(winner)
            for other in cluster:
                if other.get("table_id") != winner.get("table_id") or other.get("engine") != winner.get("engine"):
                    report["tabel_duplikat_lintas_engine_dibuang"] += 1
                    dropped.append(drop_record(other, "duplikat_lintas_engine"))

    return kept


# ============================================================
# 4. TABEL LANJUTAN ANTARHALAMAN
# ============================================================


def merge_continuations(
    tables: list[dict[str, Any]],
    report: Counter
) -> list[dict[str, Any]]:
    """
    Menggabungkan tabel yang berlanjut ke halaman berikutnya.

    Prioritas:
    1. Caption eksplisit: Lanjutan Tabel X.
    2. Jika caption tidak tersedia, gunakan kecocokan header,
       jumlah kolom, dan kedekatan halaman.

    Tabel hanya digabungkan jika struktur kolomnya sesuai.
    """

    def order(t: dict[str, Any]) -> tuple:
        rect = rect_of(t)
        return (
            t.get("page", 0),
            rect[1] if rect else 0.0,
            t.get("table_id") or 0
        )

    def normalize_cell(value: Any) -> str:
        return re.sub(
            r"\s+",
            " ",
            str(value or "").strip().lower()
        )

    def get_rows(table: dict[str, Any]) -> list[list[str]]:
        return table.get("clean_rows") or []

    def get_header(table: dict[str, Any]) -> list[list[str]]:
        rows = get_rows(table)
        count = table.get("header_row_count", 0)

        return rows[:count] if count > 0 else []

    def column_count(table: dict[str, Any]) -> int:
        rows = get_rows(table)

        if not rows:
            return 0

        return max(len(row) for row in rows)

    def header_similarity(
        previous: dict[str, Any],
        current: dict[str, Any]
    ) -> float:
        """
        Membandingkan header berdasarkan isi setiap kolom.
        Mengabaikan sel kosong agar header bertingkat tetap
        dapat dibandingkan.
        """

        prev_header = get_header(previous)
        curr_header = get_header(current)

        if not prev_header or not curr_header:
            return 0.0

        prev_cols = column_count(previous)
        curr_cols = column_count(current)

        if prev_cols != curr_cols:
            return 0.0

        def column_signature(header):
            signatures = []

            for col in range(prev_cols):
                values = []

                for row in header:
                    value = normalize_cell(
                        row[col] if col < len(row) else ""
                    )

                    if value:
                        values.append(value)

                signatures.append(" ".join(values))

            return signatures

        a = column_signature(prev_header)
        b = column_signature(curr_header)

        comparable = [
            (x, y)
            for x, y in zip(a, b)
            if x or y
        ]

        if not comparable:
            return 0.0

        matches = sum(
            x == y
            for x, y in comparable
        )

        return matches / len(comparable)

    def get_continuation_number(table):
        number = table.get("table_number")

        if number:
            return str(number).strip()

        caption = str(table.get("caption") or "")

        match = re.search(
            r"lanjutan\s+tabel\s+([\d.]+)",
            caption,
            re.IGNORECASE
        )

        return match.group(1) if match else None

    def is_explicit_continuation(table: dict[str, Any]) -> bool:
        caption = str(table.get("caption") or "")

        return bool(
            re.search(
                r"\blanjutan\s+tabel\b",
                caption,
                re.IGNORECASE
            )
        ) or bool(table.get("is_continuation"))

    def is_uncaptioned_fragment(table: dict[str, Any]) -> bool:
        """
        Fragmen tabel tanpa judul di awal halaman. Pada publikasi BPS,
        tabel yang berlanjut ke halaman baru kadang tidak mengulang
        judul, sehingga header-nya kosong dan langsung berisi data.
        """

        caption = str(table.get("caption") or "").strip()

        return (
            not caption
            and not get_continuation_number(table)
            and int(table.get("header_row_count") or 0) == 0
        )

    def remove_repeated_header(
        previous: dict[str, Any],
        current: dict[str, Any]
    ) -> list[list[str]]:
        """
        Menghilangkan baris header yang berulang pada tabel lanjutan.
        Baris data tetap dipertahankan.
        """

        rows = get_rows(current)
        current_header = get_header(current)
        previous_header = get_header(previous)

        if not current_header or not previous_header:
            return rows

        previous_keys = {
            tuple(normalize_cell(c) for c in row)
            for row in previous_header
        }

        current_header_count = current.get(
            "header_row_count", 0
        )

        data = rows[current_header_count:]

        # Buang header yang muncul kembali sebagai baris data.
        cleaned_data = []

        for row in data:
            key = tuple(normalize_cell(c) for c in row)

            if key in previous_keys:
                continue

            cleaned_data.append(row)

        return cleaned_data

    result: list[dict[str, Any]] = []

    for table in sorted(tables, key=order):
        current_page = table.get("page", 0)
        current_rows = get_rows(table)

        if not current_rows:
            result.append(dict(table))
            continue

        continuation = is_explicit_continuation(table)
        fragment = is_uncaptioned_fragment(table)
        current_number = get_continuation_number(table)

        candidate = None
        candidate_score = -1.0

        # Cari tabel sebelumnya yang berakhir tepat pada halaman
        # sebelum tabel saat ini.
        for prev in reversed(result):

            prev_end = prev.get(
                "page_end",
                prev.get("page", 0)
            )

            if prev_end != current_page - 1:
                continue

            if column_count(prev) != column_count(table):
                continue

            similarity = header_similarity(prev, table)

            # Fragmen tanpa judul dan tanpa header: cocokkan hanya
            # berdasarkan posisi halaman dan jumlah kolom.
            if fragment:
                score = 1.5

            # Caption lanjutan menjadi bukti kuat, tetapi header
            # tetap harus memiliki kemiripan struktur.
            elif continuation:
                # Lanjutan tanpa header (header hanya di halaman pertama):
                # cukup cocokkan jumlah kolom dan nomor tabel.
                if get_header(table) and similarity < 0.5:
                    continue

                prev_number = get_continuation_number(prev)

                # Jika nomor tabel sebelumnya tersedia,
                # nomor harus cocok.
                if (
                    prev_number
                    and current_number
                    and prev_number != current_number
                ):
                    continue

                score = 2.0 + similarity

            else:
                # Tanpa caption eksplisit, syarat lebih ketat.
                if similarity < 0.85:
                    continue

                # Jangan menggabungkan tabel yang sudah memiliki
                # catatan sumber, karena bisa jadi tabel sebelumnya
                # sudah selesai.
                if prev.get("notes"):
                    continue

                score = similarity

            if score > candidate_score:
                candidate = prev
                candidate_score = score

        if candidate is None:
            result.append(dict(table))
            continue

        # Ambil data tanpa header berulang.
        data = remove_repeated_header(candidate, table)

        candidate["clean_rows"].extend(data)

        candidate["page_end"] = table.get(
            "page_end",
            current_page
        )

        candidate.setdefault("merged_table_ids", []).append(
            table.get("table_id")
        )

        candidate["notes"] = (
            as_note_list(candidate.get("notes"))
            + as_note_list(table.get("notes"))
        )

        report["tabel_lanjutan_digabung"] += 1

    return result


# ============================================================
# 5. FILTER TABEL TANPA INFORMASI
# Tabel needs_review yang masih punya data tetap dipertahankan.
# '-' dan '...' adalah nilai sah di tabel BPS: tabel dengan label baris
# tetapi nilai '-' TIDAK dianggap kosong.
# ============================================================

def information_check(table: dict[str, Any]) -> str | None:
    """Alasan tabel dibuang, atau None bila tabel punya informasi."""
    rows = table["clean_rows"]
    data = rows[table.get("header_row_count", 0):]
    if not data:
        return "tanpa_baris_data"

    cells = [c for r in data for c in r if c]
    if not cells:
        return "semua_sel_kosong"

    has_label = any(any(ch.isalpha() for ch in r[0]) for r in data if r and r[0])
    has_value = any(c not in PLACEHOLDERS for r in data for c in r[1:] if c)
    if not has_label and not has_value:
        return "hanya_placeholder"

    if sum(len(c) <= 1 for c in cells) / len(cells) > 0.8 and not has_label:
        return "sel_satu_karakter_tanpa_label"
    return None


def filter_uninformative(
    tables: list[dict[str, Any]], report: Counter, dropped: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    kept = []
    for table in tables:
        reason = information_check(table)
        if reason:
            report[f"tabel_dibuang_{reason}"] += 1
            dropped.append(drop_record(table, reason))
        else:
            kept.append(table)
    return kept


# ============================================================
# 6. CLEANING NARASI
# ============================================================

def junk_reason(text: str) -> str | None:
    if len(text) <= 2:
        return "terlalu_pendek"
    if BLANK_PAGE_RE.search(text):
        return "halaman_kosong"
    if URL_RE.match(text):
        return "url_saja"
    tokens = text.split()
    if not any(c.isalpha() for c in text) and len(tokens) < 6:
        return "tanpa_huruf"
    if len(tokens) >= 6:
        numeric = sum(bool(NUM_TOKEN_RE.match(t)) for t in tokens)
        if numeric / len(tokens) >= 0.8:
            return "deretan_angka_bocoran_tabel_atau_grafik"
    return None


def should_join(prev: str, nxt: str) -> bool:
    if is_heading(prev) or not nxt[:1].islower():
        return False
    if prev.endswith("-"):
        return True
    return len(prev) >= 40 and not prev.endswith(TERMINAL)


def join_paragraphs(prev: str, nxt: str) -> str:
    return prev[:-1] + nxt if prev.endswith("-") else f"{prev} {nxt}"


def clean_narrative(
    pages: list[dict[str, Any]],
    report: Counter,
    dropped: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for rec in pages:
        paragraphs: list[str] = []
        for raw in rec.get("paragraphs") or []:
            text = clean_text(raw, report)
            reason = junk_reason(text)
            if reason:
                report[f"dibuang_{reason}"] += 1
                dropped.append({
                    "document": rec.get("document"),
                    "page": rec["page"],
                    "reason": reason,
                    "text": raw[:300],
                })
                continue
            if paragraphs and paragraphs[-1] == text:
                report["dibuang_duplikat_berurutan"] += 1
                continue
            paragraphs.append(text)
        new = dict(rec)
        new["paragraphs"] = paragraphs
        out.append(new)

    # sambung paragraf yang terpotong (dalam halaman dan antarhalaman)
    last_rec: dict[str, Any] | None = None
    for rec in out:
        kept: list[str] = []
        for text in rec["paragraphs"]:
            if kept:
                target, scope = kept, "paragraf_disambung_dalam_halaman"
            elif last_rec and last_rec["paragraphs"]:
                target, scope = last_rec["paragraphs"], "paragraf_disambung_antarhalaman"
            else:
                target, scope = None, ""
            if target and should_join(target[-1], text):
                target[-1] = join_paragraphs(target[-1], text)
                report[scope] += 1
                continue
            kept.append(text)
        rec["paragraphs"] = kept
        if kept:
            last_rec = rec

    for rec in out:
        rec["text"] = "\n\n".join(rec["paragraphs"]).strip()
        rec["word_count"] = len(rec["text"].split()) if rec["text"] else 0

    # halaman tanpa teks sama sekali tidak diteruskan
    nonempty = [r for r in out if r["text"]]
    report["halaman_narasi_kosong_dibuang"] = len(out) - len(nonempty)
    return nonempty


# ============================================================
# MAIN
# ============================================================

def dedupe_identical_rows(
    tables: list[dict[str, Any]],
    report: Counter
) -> list[dict[str, Any]]:
    """
    Buang baris data yang identik di dalam satu tabel.

    Penggabungan tabel lanjutan kadang mengulang baris terakhir halaman
    sebelumnya sehingga nilai bisa terhitung dua kali.
    """

    for table in tables:
        rows = table.get("clean_rows") or []
        header_count = int(table.get("header_row_count") or 0)
        header = rows[:header_count]
        data = rows[header_count:]

        seen: set[tuple[str, ...]] = set()
        unique: list[list[str]] = []

        for row in data:
            key = tuple(
                re.sub(r"\s+", " ", str(cell or "").strip().lower())
                for cell in row
            )

            if key in seen:
                report["baris_duplikat_dibuang"] += 1
                continue

            seen.add(key)
            unique.append(row)

        table["clean_rows"] = header + unique

    return tables


def clean_document(
    doc_dir: Path,
    skip_pages: set[int],
    merge_tables: bool,
    dedup: bool,
    prefer: list[str],
) -> dict[str, Any] | None:
    narrative = read_jsonl(doc_dir / "narrative.jsonl")
    tables = read_jsonl(doc_dir / "tables.jsonl")
    if not narrative and not tables:
        return None

    report: Counter = Counter()
    dropped_narrative: list[dict[str, Any]] = []
    dropped_tables: list[dict[str, Any]] = []
    stages = {"tabel_awal": len(tables)}

    # 1. filter halaman
    narrative, tables = filter_pages(narrative, tables, skip_pages, report, dropped_tables)
    stages["tabel_setelah_filter_halaman"] = len(tables)

    # 2. cleaning tabel
    cleaned = []
    for table in tables:
        result = clean_one_table(table, report)
        if result is None:
            report["tabel_dibuang_kosong_setelah_cleaning"] += 1
            dropped_tables.append(drop_record(table, "kosong_setelah_cleaning"))
        else:
            cleaned.append(result)
    stages["tabel_setelah_cleaning"] = len(cleaned)

    # 3. dedup lintas engine
    if dedup:
        cleaned = dedupe_tables(cleaned, prefer, report, dropped_tables)
    stages["tabel_setelah_dedup"] = len(cleaned)

    # 4. tabel lanjutan
    if merge_tables:
        cleaned = merge_continuations(cleaned, report)
    else:
        cleaned = sorted(cleaned, key=lambda t: (t["page"], t.get("table_id") or 0))
    stages["tabel_setelah_gabung_lanjutan"] = len(cleaned)

    cleaned = dedupe_identical_rows(cleaned, report)
    stages["tabel_setelah_dedup_baris"] = len(cleaned)

    # 5. filter tabel tanpa informasi
    cleaned = filter_uninformative(cleaned, report, dropped_tables)
    stages["tabel_akhir"] = len(cleaned)

    # 6. narasi
    words_before = sum(p.get("word_count", 0) for p in narrative)
    clean_pages = clean_narrative(narrative, report, dropped_narrative)

    write_jsonl(doc_dir / "narrative_clean.jsonl", clean_pages)
    write_jsonl(doc_dir / "tables_clean.jsonl", cleaned)
    write_jsonl(doc_dir / "narrative_dropped.jsonl", dropped_narrative)
    write_jsonl(doc_dir / "tables_dropped.jsonl", dropped_tables)

    summary = {
        "halaman_dikecualikan": sorted(skip_pages),
        "kata_narasi_sebelum": words_before,
        "kata_narasi_sesudah": sum(p["word_count"] for p in clean_pages),
        "halaman_narasi_akhir": len(clean_pages),
        "tahap_tabel": stages,
        "tindakan": dict(report),
    }
    with (doc_dir / "cleaning_report.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def print_summary(name: str, summary: dict[str, Any]) -> None:
    print(f"\n{name}")
    print(f"  kata narasi : {summary['kata_narasi_sebelum']} -> {summary['kata_narasi_sesudah']}"
          f"  ({summary['halaman_narasi_akhir']} halaman)")
    for stage, n in summary["tahap_tabel"].items():
        print(f"  {stage:<34}{n}")
    print()
    for action, n in sorted(summary["tindakan"].items()):
        print(f"  {action:<52}{n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtering dan cleaning hasil ekstraksi pipeline.")
    parser.add_argument("--dir", type=Path,
                        help="Satu folder hasil (berisi narrative.jsonl dan tables.jsonl), "
                             "mis. extracted/<dokumen>/lattice_all")
    parser.add_argument("--input", type=Path, default=OUTPUT_DIR,
                        help="Folder 'extracted'; dipakai bila --dir tidak diisi.")
    parser.add_argument("--label", default="lattice",
                        help="Subfolder per dokumen (lattice, stream, pymupdf, lattice_all, ...).")
    parser.add_argument("--skip-pages", default="",
                        help="Halaman yang dikecualikan, mis. 1-22 atau 1-22,30.")
    parser.add_argument("--prefer", default="",
                        help="Urutan engine untuk pemutus seri saat dedup, "
                             "mis. pdfplumber,pymupdf,camelot.")
    parser.add_argument("--no-dedup", action="store_true", help="Lewati dedup lintas engine.")
    parser.add_argument("--no-merge-tables", action="store_true",
                        help="Jangan gabungkan tabel lanjutan antarhalaman.")
    args = parser.parse_args()

    skip = parse_pages(args.skip_pages) if args.skip_pages else set()
    prefer = [p.strip() for p in args.prefer.split(",") if p.strip()]

    if args.dir:
        targets = [(args.dir.parent.name, args.dir)]
    else:
        targets = [
            (d.name, d / args.label)
            for d in sorted(p for p in args.input.iterdir() if p.is_dir())
            if (d / args.label).exists()
        ]

    for name, target in targets:
        summary = clean_document(
            target,
            skip_pages=skip,
            merge_tables=not args.no_merge_tables,
            dedup=not args.no_dedup,
            prefer=prefer,
        )
        if summary:
            print_summary(name, summary)


if __name__ == "__main__":
    main()