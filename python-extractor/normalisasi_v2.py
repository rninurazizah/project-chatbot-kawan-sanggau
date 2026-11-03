"""
Normalisasi multi-dokumen untuk pipeline PaddleOCR hybrid.

Membaca hasil per dokumen:
    extracted_paddle/<doc>/narrative_clean.jsonl
    extracted_paddle/<doc>/glossary.jsonl
    extracted_paddle/<doc>/tables_clean.jsonl
    extracted_paddle/<doc>/build_summary.json

Menulis kontrak RAG:
    extracted_paddle/normalized_data.jsonl

Tidak ada halaman yang di-hardcode. Semua metadata dokumen
(kecamatan, tahun, dokumen) dibawa dari build_summary.json / record.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from common import NUM_RE, is_colnum_row


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "extracted_paddle"
DEFAULT_OUTPUT = DEFAULT_INPUT / "normalized_data.jsonl"

DOC_FIELDS = (
    "document",
    "document_id",
    "source",
    "kecamatan",
    "tahun",
    "kabupaten",
    "document_type",
)

URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.I)
PAGE_NUMBER_RE = re.compile(r"^(?:halaman\s*)?[-–\s]*\d{1,4}[-–\s]*$", re.I)
NUM_TOKEN_RE = re.compile(r"^[(+\-]?\d[\d.,/%\-]*\)?$")
HYPHEN_RE = re.compile(r"([A-Za-z]{2,})-\s+([a-z]{2,})")

# Sisa watermark yang masih lolos OCR di awal sel header,
# contoh: "s5,- Desa/Kelurahan" -> "Desa/Kelurahan".
HEADER_RESIDUE_RE = re.compile(r"^[a-z0-9]{1,4}[.,\-]+\s+(?=[A-Z])")

# Tahun dengan angka superscript ikut terbaca di baris terpisah:
# "2 2023" -> "2023".
HEADER_FOOTNOTE_PREFIX_RE = re.compile(r"^([1-9])\s+((?:19|20)\d{2})$")

# Catatan kaki yang menempel di akhir judul kolom:
# "Jarak (km)1" -> "Jarak (km)".
HEADER_FOOTNOTE_SUFFIX_RE = re.compile(r"^(.*[A-Za-z\)\]])[1-9]$")

LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


# ============================================================
# REPARASI HEADER
# ============================================================

# Kata yang sering terpecah akibat line-break di dalam sel header.
HEADER_EXTRA_TERMS = {
    "basarah",
    "kelenteng",
    "khonghucu",
    "pembunuhan",
    "pemerasan",
    "penganiayaan",
    "penodongan",
    "perampokan",
    "perkosaan",
    "perzinahan",
    "lainnya",
    "transportasi",
    "poskesdes",
    "polindes",
    "poskesdes/polindes",
    "poliklinik",
    "pemberatan",
}

HEADER_LEXICON: set[str] = set(HEADER_EXTRA_TERMS)

HEADER_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z/.\-]*")
HEADER_WORD_RE = re.compile(r"[A-Za-z/.\-]+")


def build_header_lexicon(contents: list[str]) -> set[str]:
    """Kumpulkan kosakata dari narasi/glosarium untuk mereparasi header."""

    lexicon = set(HEADER_EXTRA_TERMS)

    for text in contents:
        for token in HEADER_TOKEN_RE.findall(text or ""):
            lexicon.add(token.lower())

    return lexicon


def repair_header_fragments(text: str) -> str:
    """
    Gabungkan kata yang terpotong line-break di dalam sel header.

    Contoh: "Persenta se Pendudu k" -> "Persentase Penduduk".
    Hanya kata hasil gabungan yang ada di kosakata yang digabung, sehingga
    pasangan kata sah seperti "Kode Wilayah" tidak ikut berubah.
    """

    tokens = text.split()

    # "Desa/Kelura-han" -> "Desa/Kelurahan" bila bentuk tanpa hyphen
    # memang kata yang dikenal.
    tokens = [
        token.replace("-", "")
        if "-" in token and token.replace("-", "").lower() in HEADER_LEXICON
        else token
        for token in tokens
    ]

    if not tokens:
        return text

    # Footnote yang menempel di tengah kata: "Transportas2" -> "Transportas".
    tokens = [
        re.sub(r"^([A-Za-z/.\-]{4,})[1-9]$", r"\1", token)
        for token in tokens
    ]

    if len(tokens) < 2:
        return " ".join(tokens)

    result: list[str] = []
    index = 0

    while index < len(tokens):
        merged = False

        for span in (3, 2):
            if index + span > len(tokens):
                continue

            group = tokens[index:index + span]

            if not all(HEADER_WORD_RE.fullmatch(token) for token in group):
                continue

            if all(token.lower() in HEADER_LEXICON for token in group):
                continue

            candidate = "".join(group)

            if candidate.lower() in HEADER_LEXICON:
                result.append(candidate)
                index += span
                merged = True
                break

        if not merged:
            result.append(tokens[index])
            index += 1

    repaired = " ".join(result)
    # Satuan ber-superscript dari text layer: "(m 3 )" -> "(m3)".
    repaired = re.sub(r"\(\s*([A-Za-z]+)\s*(\d)\s*\)", r"(\1\2)", repaired)
    return repaired



# ============================================================
# I/O
# ============================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

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
                ) from error

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# TEKS
# ============================================================

def normalize_basic(value: Any) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFC", str(value))

    for source, target in LIGATURES.items():
        text = text.replace(source, target)

    text = text.translate(ZERO_WIDTH)
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufffd", "")
    text = text.replace("\u00b2", "2").replace("\u00b3", "3")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return text


def clean_cell(value: Any) -> str:
    text = normalize_basic(value)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def clean_paragraph(value: Any) -> str:
    text = normalize_basic(value)
    text = HYPHEN_RE.sub(
        lambda match: (
            f"{match.group(1)}-{match.group(2)}"
            if match.group(1).lower() == match.group(2).lower()
            else match.group(1) + match.group(2)
        ),
        text,
    )
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_header(value: Any) -> str:
    text = clean_cell(value)
    text = HEADER_RESIDUE_RE.sub("", text)
    text = re.sub(r"\s*/\s*", "/", text)

    match = HEADER_FOOTNOTE_PREFIX_RE.match(text)
    if match:
        text = match.group(2)
    else:
        match = HEADER_FOOTNOTE_SUFFIX_RE.match(text)
        if match:
            text = match.group(1)

    return text.strip()


def junk_reason(text: str) -> str | None:
    if len(text) < 40:
        return "terlalu_pendek"
    if URL_RE.match(text):
        return "url_saja"
    if PAGE_NUMBER_RE.match(text):
        return "nomor_halaman"
    if "sengaja dikosongkan" in text.lower():
        return "halaman_kosong"

    tokens = text.split()
    if len(tokens) >= 6:
        numeric = sum(bool(NUM_TOKEN_RE.match(token)) for token in tokens)
        if numeric / len(tokens) >= 0.8:
            return "deretan_angka"

    return None


def document_meta(
    doc_dir: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    summary_path = doc_dir / "build_summary.json"

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    meta = {field: summary.get(field) for field in DOC_FIELDS}

    for record in records:
        for field in DOC_FIELDS:
            if meta.get(field) in (None, "") and record.get(field) not in (None, ""):
                meta[field] = record.get(field)

    if not meta.get("document"):
        meta["document"] = doc_dir.name

    if not meta.get("document_id"):
        meta["document_id"] = doc_dir.name

    return meta


# ============================================================
# NARASI
# ============================================================

def normalize_narrative(
    records: list[dict[str, Any]],
    meta: dict[str, Any],
    report: Counter,
    dropped: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_contents: set[str] = set()

    for record in records:
        raw = record.get("text") or record.get("content") or ""
        paragraphs = [clean_paragraph(p) for p in (record.get("paragraphs") or [])]

        if not paragraphs:
            paragraphs = [
                clean_paragraph(part)
                for part in re.split(r"\n\s*\n", normalize_basic(raw))
            ]

        paragraphs = [p for p in paragraphs if p and not junk_reason(p)]

        content = "\n\n".join(paragraphs).strip()

        reason = junk_reason(content)
        if reason:
            report[f"narasi_dibuang_{reason}"] += 1
            dropped.append({
                "type": "narrative",
                "page": record.get("page"),
                "reason": reason,
                "text": content[:200],
            })
            continue

        if content in seen_contents:
            report["narasi_duplikat_dibuang"] += 1
            continue

        seen_contents.add(content)

        title = clean_cell(record.get("title") or record.get("section"))
        if not title or title.lower() == "narasi" or len(title) > 160:
            title = meta["document"]

        page = record.get("page")
        page_end = record.get("page_end") or page

        normalized.append({
            "type": "narrative",
            "title": title,
            "section": clean_cell(record.get("section")) or title,
            "page": page,
            "page_end": page_end,
            "content": content,
            "word_count": len(content.split()),
            "metadata": {
                "normalization_method": "paddle_hybrid",
                "source_format": "text_layer",
            },
            **meta,
        })

    report["narasi_valid"] += len(normalized)
    return normalized


# ============================================================
# GLOSARIUM
# ============================================================

def normalize_glossary(
    records: list[dict[str, Any]],
    meta: dict[str, Any],
    report: Counter,
    dropped: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for record in records:
        content = clean_paragraph(record.get("content") or "")

        if len(content) < 50:
            report["glosarium_dibuang_terlalu_pendek"] += 1
            dropped.append({
                "type": "glossary",
                "page": record.get("page"),
                "reason": "terlalu_pendek",
            })
            continue

        title = clean_cell(record.get("title") or "Glosarium")

        normalized.append({
            "type": "glossary",
            "title": title,
            "page": record.get("page"),
            "page_end": record.get("page"),
            "content": content,
            "metadata": {
                "normalization_method": "paddle_hybrid",
                "source_format": "text_layer",
            },
            **meta,
        })

    report["glosarium_valid"] += len(normalized)
    return normalized


# ============================================================
# TABEL
# ============================================================

def build_headers(header_rows: list[list[str]], expected_columns: int) -> list[str]:
    """Gabungkan header bertingkat dengan forward-fill per kolom."""

    headers: list[str] = []

    # Sel header gabungan hanya menyimpan label di kolom pertama.
    # Ratakan label ke kanan agar "Negeri | | Swasta |" menjadi
    # "Negeri | Negeri | Swasta | Swasta".
    filled_rows: list[list[str]] = []

    for row_index, row in enumerate(header_rows):
        filled: list[str] = []
        previous = ""

        # Blank di ujung kanan hanya dianggap sel gabungan bila baris
        # header di bawahnya masih punya isi di kolom tersebut.
        last_value = -1

        for column in range(expected_columns):
            if column < len(row) and clean_header(row[column]):
                last_value = column

        later_covered = {
            column
            for later in header_rows[row_index + 1:]
            for column in range(expected_columns)
            if column < len(later) and clean_header(later[column])
        }

        for column in range(expected_columns):
            value = clean_header(row[column]) if column < len(row) else ""

            if value:
                previous = value
            elif previous and (column <= last_value or column in later_covered):
                value = previous

            filled.append(value)

        filled_rows.append(filled)

    for column in range(expected_columns):
        parts: list[str] = []
        previous = ""

        for row in filled_rows:
            value = row[column]

            if value:
                previous = value
            else:
                value = previous

            if value and (not parts or value != parts[-1]):
                parts.append(value)

        headers.append(
            repair_header_fragments(" ".join(parts).strip())
            or f"Kolom {column + 1}"
        )

    # Artefak header gabungan: label kolom pertama ikut terulang pada
    # sub-kolom di sebelahnya ("Jenis Tanaman TBM"). Buang prefiksnya
    # bila mayoritas sub-kolom memakai prefiks yang sama.
    first = headers[0].strip()

    if first:
        prefix = f"{first} "
        remainder = [
            header[len(prefix):]
            for header in headers[1:]
            if header.startswith(prefix) and header[len(prefix):].strip()
        ]

        if len(remainder) >= 2:
            headers = [headers[0]] + [
                header[len(prefix):]
                if header.startswith(prefix) and header[len(prefix):].strip()
                else header
                for header in headers[1:]
            ]

    counts: Counter = Counter()
    unique: list[str] = []

    for header in headers:
        counts[header] += 1
        if counts[header] > 1:
            header = f"{header} ({counts[header]})"
        unique.append(header)

    return unique


def detect_table_issues(rows: list[list[str]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        for column_index, value in enumerate(row, start=1):
            text = clean_cell(value)

            if not text:
                continue

            if re.search(r"\d+[a-zA-Z]{1,3}(?:,|\.)\d+", text):
                issues.append({
                    "code": "POSSIBLE_CORRUPTED_NUMERIC",
                    "row": row_index,
                    "column": column_index,
                    "value": text,
                })

            if re.search(r"\d{10}\s+[a-zA-Z]", text):
                issues.append({
                    "code": "POSSIBLE_CORRUPTED_REGION_CODE",
                    "row": row_index,
                    "column": column_index,
                    "value": text,
                })

    return issues


def normalize_tables(
    records: list[dict[str, Any]],
    meta: dict[str, Any],
    report: Counter,
    dropped: list[dict[str, Any]],
    issues_report: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for record in records:
        source_rows = record.get("clean_rows") or record.get("raw_rows") or []

        rows = [
            [clean_cell(cell) for cell in row]
            for row in source_rows
            if isinstance(row, (list, tuple))
        ]
        rows = [row for row in rows if any(row)]

        if not rows:
            report["tabel_dibuang_tanpa_baris"] += 1
            dropped.append({
                "type": "table",
                "table_id": record.get("table_id"),
                "page": record.get("page"),
                "reason": "tanpa_baris",
            })
            continue

        expected_columns = max(len(row) for row in rows)
        rows = [row + [""] * (expected_columns - len(row)) for row in rows]

        header_count = record.get("header_row_count")

        try:
            header_count = int(header_count)
        except (TypeError, ValueError):
            header_count = 0

        if header_count < 1:
            header_count = 1

        header_count = min(header_count, len(rows))

        header_rows = rows[:header_count]
        data_rows = rows[header_count:]

        # Buang baris nomor kolom yang masih tersisa.
        data_rows = [row for row in data_rows if not is_colnum_row(row)]

        # Baris footer sumber ("Kecamatan ... - - -") bukan data.
        data_rows = [
            row
            for row in data_rows
            if not (
                row
                and re.match(r"^(kecamatan|kabupaten)\b", row[0], re.I)
                and not any(re.search(r"\d", cell) for cell in row)
            )
        ]

        # Sebagian tabel hasil text layer punya kolom pemisah kosong
        # yang posisinya bergeser antarbaris. Jika jumlah sel berisi
        # pada header sama dengan mayoritas baris data, padatkan dengan
        # membuang sel kosong struktural tersebut.
        if header_rows and data_rows:
            header_tokens = sum(
                1 for cell in header_rows[0] if cell.strip()
            )
            data_token_counts = [
                sum(1 for cell in row if cell.strip())
                for row in data_rows
            ]

            has_gap = any(
                not cell.strip() for cell in header_rows[0]
            )
            matching = sum(
                1 for count in data_token_counts
                if count == header_tokens
            )

            if (
                has_gap
                and header_tokens >= 2
                and matching / len(data_rows) >= 0.8
            ):
                header_rows = [
                    [cell for cell in row if cell.strip()]
                    for row in header_rows
                ]
                data_rows = [
                    [cell for cell in row if cell.strip()]
                    for row in data_rows
                ]

                width = max(
                    [len(row) for row in header_rows + data_rows]
                    or [0]
                )

                header_rows = [
                    row + [""] * (width - len(row))
                    for row in header_rows
                ]
                data_rows = [
                    row + [""] * (width - len(row))
                    for row in data_rows
                ]

        # Kolom yang kosong di seluruh header dan data adalah garis
        # pemisah hasil ekstraksi, bukan kolom statistik.
        if header_rows or data_rows:
            width = max(
                [len(row) for row in header_rows + data_rows]
                or [0]
            )

            header_rows = [
                row + [""] * (width - len(row))
                for row in header_rows
            ]
            data_rows = [
                row + [""] * (width - len(row))
                for row in data_rows
            ]

            keep = [
                index for index in range(width)
                if any(
                    row[index].strip()
                    for row in header_rows + data_rows
                )
            ]

            if keep and len(keep) < width:
                header_rows = [
                    [row[index] for index in keep]
                    for row in header_rows
                ]
                data_rows = [
                    [row[index] for index in keep]
                    for row in data_rows
                ]

        rows = header_rows + data_rows
        expected_columns = max(
            [len(row) for row in rows] or [0]
        )

        # Artefak header sel gabungan dari OCR ("Desa" sendirian).
        data_rows = [
            row
            for row in data_rows
            if not (
                sum(bool(cell) for cell in row) == 1
                and row[0].strip().lower() == "desa"
            )
        ]

        if not data_rows:
            report["tabel_dibuang_tanpa_data"] += 1
            dropped.append({
                "type": "table",
                "table_id": record.get("table_id"),
                "page": record.get("page"),
                "reason": "tanpa_data",
            })
            continue

        columns = build_headers(header_rows, expected_columns)

        # Tabel satu kolom tanpa nilai bukan tabel statistik yang berguna.
        if len(columns) < 2:
            report["tabel_dibuang_satu_kolom"] += 1
            dropped.append({
                "type": "table",
                "table_id": record.get("table_id"),
                "page": record.get("page"),
                "reason": "satu_kolom",
            })
            continue

        dict_rows = [
            {column: row[index] for index, column in enumerate(columns)}
            for row in data_rows
        ]

        table_number = record.get("table_number")
        title = clean_cell(record.get("caption") or record.get("table_title"))

        if not title:
            title = (
                f"Tabel {table_number}"
                if table_number
                else f"Tabel halaman {record.get('page')}"
            )

        issues = detect_table_issues(rows)

        if issues:
            issues_report.append({
                "type": "table",
                "table_id": record.get("table_id"),
                "page": record.get("page"),
                "issues": issues,
            })

        status = record.get("status") or "accepted"
        if status not in ("accepted", "needs_review"):
            status = "needs_review"

        page = record.get("page")

        normalized.append({
            "type": "table",
            "title": title,
            "table_id": record.get("table_id"),
            "table_number": table_number,
            "page": page,
            "page_end": record.get("page_end") or page,
            "columns": columns,
            "rows": dict_rows,
            "is_continuation": bool(record.get("is_continuation")),
            "metadata": {
                "engine": record.get("engine"),
                "flavor": record.get("flavor"),
                "normalization_method": "clean_rows",
                "quality_status": status,
                "quality_issues": issues,
                "original_quality": record.get("quality"),
                "row_count": len(dict_rows),
                "column_count": len(columns),
                "notes": record.get("notes"),
            },
            **meta,
        })

    report["tabel_valid"] += len(normalized)
    return normalized


# ============================================================
# VALIDASI
# ============================================================

def validate(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    valid_types = {"narrative", "glossary", "table"}

    for index, record in enumerate(records, start=1):
        record_type = record.get("type")

        if record_type not in valid_types:
            errors.append(f"Record {index}: tipe tidak valid ({record_type}).")
            continue

        if record_type in {"narrative", "glossary"}:
            if not record.get("content"):
                errors.append(f"Record {index}: content kosong.")

        if record_type == "table":
            columns = record.get("columns") or []
            rows = record.get("rows") or []

            if not columns:
                errors.append(f"Tabel {record.get('table_id')}: columns kosong.")

            if not rows:
                errors.append(f"Tabel {record.get('table_id')}: rows kosong.")

            for row_index, row in enumerate(rows, start=1):
                if list(row.keys()) != columns:
                    errors.append(
                        f"Tabel {record.get('table_id')}, baris {row_index}: "
                        "struktur kolom tidak konsisten."
                    )

    return errors


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalisasi semua dokumen hasil pipeline Paddle hybrid."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Folder extracted_paddle.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="File normalized_data.jsonl.")
    parser.add_argument("--report", type=Path, default=None,
                        help="File laporan JSON.")
    args = parser.parse_args()

    report_path = args.report or (args.output.parent / "normalization_report.json")
    issues_path = args.output.parent / "normalization_issues.jsonl"

    doc_dirs = sorted(
        path
        for path in args.input.iterdir()
        if path.is_dir()
        and (
            (path / "narrative_clean.jsonl").exists()
            or (path / "tables_clean.jsonl").exists()
        )
    )

    if not doc_dirs:
        raise FileNotFoundError(
            f"Tidak ada dokumen hasil cleaning di {args.input}"
        )

    all_records: list[dict[str, Any]] = []
    all_dropped: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = []
    per_document: dict[str, Any] = {}

    payloads = []

    for doc_dir in doc_dirs:
        payloads.append((
            doc_dir,
            read_jsonl(doc_dir / "narrative_clean.jsonl"),
            read_jsonl(doc_dir / "glossary.jsonl"),
            read_jsonl(doc_dir / "tables_clean.jsonl"),
        ))

    HEADER_LEXICON.clear()
    HEADER_LEXICON.update(build_header_lexicon([
        str(record.get("content") or "")
        for _, narrative, glossary, _ in payloads
        for record in narrative + glossary
    ]))

    for doc_dir, narrative, glossary, tables in payloads:

        meta = document_meta(doc_dir, narrative + glossary + tables)

        report: Counter = Counter()
        dropped: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []

        normalized = (
            normalize_narrative(narrative, meta, report, dropped)
            + normalize_glossary(glossary, meta, report, dropped)
            + normalize_tables(tables, meta, report, dropped, issues)
        )

        for record in normalized:
            record["_doc_dir"] = doc_dir.name

        all_records.extend(normalized)
        all_dropped.extend(dropped)
        all_issues.extend(issues)

        per_document[doc_dir.name] = {
            "document": meta.get("document"),
            "kecamatan": meta.get("kecamatan"),
            "tahun": meta.get("tahun"),
            "narrative_source": len(narrative),
            "table_source": len(tables),
            "glossary_source": len(glossary),
            **{key: value for key, value in report.items()},
        }

        print(
            f"{doc_dir.name}: "
            f"narasi={report['narasi_valid']} "
            f"glossary={report['glosarium_valid']} "
            f"tabel={report['tabel_valid']}"
        )

    for record in all_records:
        record.pop("_doc_dir", None)

    all_records.sort(
        key=lambda item: (
            str(item.get("document_id") or ""),
            item.get("page") or 0,
            {"table": 0, "narrative": 1, "glossary": 2}.get(item.get("type"), 3),
            item.get("table_id") or 0,
        )
    )

    errors = validate(all_records)

    write_jsonl(args.output, all_records)
    write_jsonl(issues_path, all_issues)

    counts = Counter(record["type"] for record in all_records)

    summary = {
        "status": "SUCCESS" if not errors else "FAILED",
        "documents": len(doc_dirs),
        "records": len(all_records),
        "narrative": counts["narrative"],
        "glossary": counts["glossary"],
        "table": counts["table"],
        "total_issues": sum(len(item["issues"]) for item in all_issues),
        "dropped": len(all_dropped),
        "validation": {
            "all_valid": not errors,
            "errors": errors,
        },
        "per_document": per_document,
        "output": str(args.output),
        "issues_file": str(issues_path),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("NORMALISASI SELESAI")
    print("=" * 60)
    print(f"Dokumen        : {summary['documents']}")
    print(f"Narasi         : {summary['narrative']}")
    print(f"Glosarium      : {summary['glossary']}")
    print(f"Tabel          : {summary['table']}")
    print(f"Total record   : {summary['records']}")
    print(f"Validasi       : {'LULUS' if not errors else 'GAGAL'}")
    print(f"Output         : {args.output}")
    print(f"Laporan        : {report_path}")

    if errors:
        for error in errors[:20]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
