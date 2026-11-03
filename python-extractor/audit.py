
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR.parent / "extracted" / "chunks.jsonl"

OUTPUT_DIR = BASE_DIR / "audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = OUTPUT_DIR / "audit_report.json"
FLAGGED_JSONL = OUTPUT_DIR / "flagged_chunks.jsonl"

# Simbol yang sah dalam publikasi BPS
VALID_SYMBOLS = ["...", "–", "NA", "e", "x", "xx", "r"]

# Karakter OCR yang sering muncul akibat ekstraksi PDF
SUSPICIOUS_PATTERNS = [
    (
        r"[|\\]",
        "Karakter garis asing"
    ),
    (
        r"\d+\s+[a-zA-Z](?=\s|;|$)",
        "Angka diikuti karakter asing"
    ),
    (
        r"[a-zA-Z]\s+\d{2,}",
        "Karakter asing sebelum angka"
    ),
    (
        r"\d+[a-zA-Z]{1,3}(?=\s|;|$)",
        "Angka tercampur huruf"
    ),
    (
        r"\b[a-zA-Z]\s*[./]\s*[a-zA-Z]\b",
        "Kemungkinan karakter OCR di antara huruf"
    ),
]

# ============================================================
# UTILITAS
# ============================================================

def load_jsonl(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada baris {line_number}: {exc}"
                )

    return records


def save_json(path: Path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def save_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def add_issue(issues, severity, code, message):
    issues.append({
        "severity": severity,
        "code": code,
        "message": message
    })


def get_text(record):
    return str(record.get("text", "") or "")


# ============================================================
# PEMERIKSAAN UMUM
# ============================================================

def check_general(record, issues):
    text = get_text(record)

    if not text.strip():
        add_issue(
            issues, "HIGH", "EMPTY_TEXT",
            "Chunk tidak memiliki isi teks."
        )

    if len(text) < 30:
        add_issue(
            issues, "LOW", "VERY_SHORT_TEXT",
            "Isi chunk terlalu pendek."
        )

    if len(text) > 3000:
        add_issue(
            issues, "MEDIUM", "LONG_TEXT",
            f"Chunk memiliki {len(text)} karakter."
        )

    if "\ufffd" in text:
        add_issue(
            issues, "HIGH", "REPLACEMENT_CHARACTER",
            "Terdapat karakter Unicode pengganti akibat decoding."
        )

    if text.count("https://") > 1:
        add_issue(
            issues, "LOW", "REPEATED_SOURCE_URL",
            "URL sumber muncul berulang."
        )

    if re.search(r"\s{4,}", text):
        add_issue(
            issues, "LOW", "EXCESSIVE_SPACES",
            "Terdapat spasi berlebihan."
        )


# ============================================================
# PEMERIKSAAN NARRATIVE
# ============================================================

def check_narrative(record, issues):
    text = get_text(record)

    # Karakter asing yang mencurigakan
    for pattern, description in SUSPICIOUS_PATTERNS:
        matches = re.findall(pattern, text)

        if matches:
            add_issue(
                issues,
                "MEDIUM",
                "SUSPICIOUS_OCR",
                f"{description}: {matches[:5]}"
            )

    # Kata yang kemungkinan terpecah
    broken_words = re.findall(
    r"\b[a-zA-Z]{1,2}\s+[a-zA-Z]{1,2}\b",
    text
)

    if broken_words:
        add_issue(
            issues, "LOW", "POSSIBLE_BROKEN_WORD",
            f"Fragmen kata: {broken_words[:5]}"
        )

    # Angka dengan pola pemisahan tidak wajar
    suspicious_numbers = re.findall(
        r"\d+\s+\d{1,3}(?=\s|[.,]|$)",
        text
    )

    if suspicious_numbers:
        add_issue(
            issues, "MEDIUM", "SUSPICIOUS_NUMBER",
            f"Angka perlu diperiksa: {suspicious_numbers[:5]}"
        )


# ============================================================
# PEMERIKSAAN TABLE
# ============================================================

def check_table(record, issues):
    text = get_text(record)

    table_id = record.get("table_id")
    columns = record.get("columns", [])
    row_count = record.get("row_count")

    if not isinstance(columns, list) or not columns:
        add_issue(
            issues, "HIGH", "MISSING_COLUMNS",
            "Metadata kolom tidak tersedia."
        )

    if row_count is not None and row_count <= 0:
        add_issue(
            issues, "HIGH", "INVALID_ROW_COUNT",
            "Jumlah baris tidak valid."
        )

    if not table_id:
        add_issue(
            issues, "HIGH", "MISSING_TABLE_ID",
            "Identitas tabel tidak tersedia."
        )

    # Header yang kemungkinan masuk sebagai data
    if re.search(
        r"(?i)(kode wilayah:\s*desa|desa/kelurahan:\s*$)",
        text
    ):
        add_issue(
            issues, "MEDIUM", "HEADER_AS_DATA",
            "Kemungkinan baris header masuk ke bagian data."
        )

    # Kode wilayah BPS: 10 digit
    codes = re.findall(r"\b\d{10}\b", text)

    if codes:
        for code in codes:
            if not code.startswith("610506"):
                add_issue(
                    issues, "MEDIUM", "UNEXPECTED_REGION_CODE",
                    f"Kode wilayah perlu diperiksa: {code}"
                )

    # Kode wilayah dengan karakter tambahan
    suspicious_codes = re.findall(
        r"\b\d{10}\s*[a-zA-Z]+\b",
        text
    )

    if suspicious_codes:
        add_issue(
            issues, "HIGH", "CORRUPTED_REGION_CODE",
            f"Kode wilayah tercampur karakter: {suspicious_codes[:5]}"
        )

    # Nilai numerik yang tercampur huruf
    suspicious_values = re.findall(
        r"(?<!\w)\d+(?:[.,]\d+)?\s*[a-zA-Z](?=\s|;|$)",
        text
    )

    if suspicious_values:
        add_issue(
            issues, "HIGH", "CORRUPTED_NUMERIC_VALUE",
            f"Nilai numerik mencurigakan: {suspicious_values[:10]}"
        )

    # Fragmen OCR di tabel
    for pattern, description in SUSPICIOUS_PATTERNS:
        matches = re.findall(pattern, text)

        if matches:
            add_issue(
                issues, "MEDIUM", "TABLE_OCR_ARTIFACT",
                f"{description}: {matches[:5]}"
            )


# ============================================================
# PEMERIKSAAN GLOSSARY
# ============================================================

def check_glossary(record, issues):
    text = get_text(record)

    missing_symbols = [
        symbol for symbol in VALID_SYMBOLS
        if symbol not in text
    ]

    # Tidak semua halaman glossary harus memuat seluruh simbol.
    # Pemeriksaan ini hanya memberikan informasi, bukan kesalahan.
    if missing_symbols:
        add_issue(
            issues, "INFO", "GLOSSARY_SYMBOLS_NOT_PRESENT",
            "Simbol yang tidak ditemukan pada chunk ini: "
            + ", ".join(missing_symbols)
        )


# ============================================================
# AUDIT UTAMA
# ============================================================

def audit_record(record):
    issues = []

    record_type = record.get("type")

    check_general(record, issues)

    if record_type == "narrative":
        check_narrative(record, issues)

    elif record_type == "table":
        check_table(record, issues)

    elif record_type == "glossary":
        check_glossary(record, issues)

    else:
        add_issue(
            issues, "HIGH", "UNKNOWN_TYPE",
            f"Tipe chunk tidak dikenal: {record_type}"
        )

    return {
        "id": record.get("id"),
        "chunk_key": record.get("chunk_key"),
        "type": record_type,
        "document": record.get("document"),
        "page_start": record.get("page_start"),
        "page_end": record.get("page_end"),
        "table_id": record.get("table_id"),
        "char_count": len(get_text(record)),
        "status": (
            "NEEDS_REVIEW"
            if any(i["severity"] in ["HIGH", "MEDIUM"] for i in issues)
            else "PASS"
        ),
        "issues": issues,
        "text": get_text(record)
    }


def main():
    print("=" * 60)
    print("AUDIT KUALITAS CHUNK RAG")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_FILE}"
        )

    records = load_jsonl(INPUT_FILE)

    results = [
        audit_record(record)
        for record in records
    ]

    flagged = [
        item for item in results
        if item["status"] == "NEEDS_REVIEW"
    ]

    type_counts = Counter(
        item["type"] for item in results
    )

    status_counts = Counter(
        item["status"] for item in results
    )

    issue_counts = Counter()

    for item in results:
        for issue in item["issues"]:
            if issue["severity"] != "INFO":
                issue_counts[issue["code"]] += 1

    summary = {
        "input_file": str(INPUT_FILE),
        "total_chunks": len(records),
        "type_counts": dict(type_counts),
        "status_counts": dict(status_counts),
        "total_flagged": len(flagged),
        "issue_counts": dict(issue_counts),
        "flagged_chunks": [
            {
                "id": item["id"],
                "chunk_key": item["chunk_key"],
                "type": item["type"],
                "page_start": item["page_start"],
                "table_id": item["table_id"],
                "status": item["status"],
                "issues": item["issues"]
            }
            for item in flagged
        ]
    }

    save_json(REPORT_JSON, summary)
    save_jsonl(FLAGGED_JSONL, flagged)

    print(f"Total chunk       : {len(records)}")
    print(f"Narrative         : {type_counts.get('narrative', 0)}")
    print(f"Table             : {type_counts.get('table', 0)}")
    print(f"Glossary          : {type_counts.get('glossary', 0)}")
    print("-" * 60)
    print(f"Lolos pemeriksaan : {status_counts.get('PASS', 0)}")
    print(f"Perlu diperiksa   : {len(flagged)}")
    print("-" * 60)

    print("Jenis masalah:")
    for code, count in issue_counts.most_common():
        print(f"- {code}: {count}")

    print("-" * 60)
    print(f"Laporan: {REPORT_JSON}")
    print(f"Chunk bermasalah: {FLAGGED_JSONL}")


if __name__ == "__main__":
    main()
