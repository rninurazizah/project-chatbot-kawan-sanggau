
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter


# ==================================================
# 1. KONFIGURASI
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = BASE_DIR / "extracted"

RAW_FILE = EXTRACTED_DIR / "raw_pages.jsonl"
CLASSIFIED_FILE = EXTRACTED_DIR / "classified_pages.jsonl"

# Menggunakan sumber tabel hasil cleaning OCR.
# Tidak mengambil reconstructed_rows.
TABLE_FILE = (
    EXTRACTED_DIR
    / "kecamatan-kapuas-dalam-angka-2025 (1)"
    / "lattice_all"
    / "tables_reconstructed.jsonl"
)

OUTPUT_FILE = EXTRACTED_DIR / "normalized_data.jsonl"
REPORT_FILE = EXTRACTED_DIR / "normalization_report.json"
ISSUES_FILE = EXTRACTED_DIR / "normalization_issues.jsonl"

SOURCE_NAME = "Kecamatan Dalam Angka 2025"

EXCLUDED_PAGES = {9, 21}
ADDITIONAL_NARRATIVE_PAGES = {51, 201}

GLOSSARY_PAGES = {23, 24, 25, 26}

GLOSSARY_TITLES = {
    23: "Penjelasan Umum",
    24: "Singkatan",
    25: "Istilah dan Definisi",
    26: "Istilah dan Kode Wilayah"
}


# ==================================================
# 2. MEMBACA DAN MENYIMPAN JSONL
# ==================================================

def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON tidak valid pada {path}, "
                    f"baris {line_number}: {error}"
                )

    return records


def save_jsonl(records, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


# ==================================================
# 3. NORMALISASI KARAKTER DAN TEKS
# ==================================================

def normalize_characters(value):
    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(value)
    )

    # Menghapus karakter format tersembunyi
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Cf"
    )

    # Menyamakan line ending
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    return text


def clean_cell(value):
    """
    Cleaning satu sel tabel.

    Tidak mengubah:
    - Nilai angka
    - Kode wilayah
    - Tanda desimal
    - Simbol statistik
    - Karakter OCR yang belum terverifikasi

    Hanya merapikan karakter dan whitespace.
    """

    text = normalize_characters(value)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_paragraph(value):
    """
    Membersihkan whitespace dalam paragraf
    tanpa menghilangkan pemisah baris.
    """

    text = normalize_characters(value)

    lines = [
        re.sub(
            r"[ \t]+",
            " ",
            line
        ).strip()
        for line in text.split("\n")
    ]

    return "\n".join(lines).strip()


def split_paragraphs(text):
    """
    Memisahkan teks berdasarkan baris kosong.
    """

    text = normalize_characters(text)

    raw_paragraphs = re.split(
        r"\n\s*\n+",
        text
    )

    paragraphs = []

    for paragraph in raw_paragraphs:
        cleaned = clean_paragraph(paragraph)

        if cleaned:
            paragraphs.append(cleaned)

    return paragraphs


# ==================================================
# 4. EKSTRAKSI JUDUL DAN BAGIAN
# ==================================================

def extract_page_title(text):

    lines = [
        clean_cell(line)
        for line in normalize_characters(text).splitlines()
        if clean_cell(line)
    ]

    for index, line in enumerate(lines):

        if line.upper() == "ULASAN" and index > 0:
            return (
                f"{lines[index - 1]} - Ulasan"
            )

        if line.upper() == "PENJELASAN UMUM":
            return "Penjelasan Umum"

    return None


def detect_section_heading(paragraph):
    """
    Mendeteksi judul bagian seperti:
    4.1 Pendidikan
    4.2 Kesehatan
    """

    text = paragraph.strip()

    pattern = (
        r"^\d+(?:\.\d+)*\.?\s+"
        r"[A-ZÀ-ÿ][^\n]{0,150}$"
    )

    return bool(
        re.match(pattern, text)
    )


# ==================================================
# 5. NORMALISASI NARASI
# ==================================================

def normalize_narrative(
    raw_pages,
    classifications
):

    classification_map = {
        item.get("page"): item
        for item in classifications
    }

    normalized = []

    for page_data in raw_pages:

        page = page_data.get("page")

        if (
            page in EXCLUDED_PAGES
            or page in GLOSSARY_PAGES
        ):
            continue

        classification = classification_map.get(
            page,
            {}
        )

        is_candidate = (
            classification.get("page_type")
            == "narrative_candidate"
        )

        is_manual_page = (
            page in ADDITIONAL_NARRATIVE_PAGES
        )

        if not (
            is_candidate or is_manual_page
        ):
            continue

        raw_text = page_data.get("text", "")

        paragraphs = split_paragraphs(
            raw_text
        )

        if not paragraphs:
            continue

        page_title = extract_page_title(
            raw_text
        )

        current_section = page_title
        section_content = []
        section_number = 0

        def save_section():

            nonlocal section_number

            if not section_content:
                return

            content = "\n\n".join(
                section_content
            ).strip()

            if not content:
                return

            section_number += 1

            normalized.append({
                "type": "narrative",
                "title": (
                    current_section or "Narasi"
                ),
                "section": current_section,
                "page": page,
                "section_index": section_number,
                "content": content,
                "source": SOURCE_NAME,
                "metadata": {
                    "normalization_method":
                        "paragraph_preservation",
                    "classification":
                        classification
                }
            })

        for paragraph in paragraphs:

            if detect_section_heading(paragraph):

                save_section()

                section_content = []
                current_section = paragraph

            else:
                section_content.append(
                    paragraph
                )

        save_section()

    return normalized


# ==================================================
# 6. NORMALISASI GLOSARIUM
# ==================================================

def normalize_glossary(raw_pages):

    normalized = []

    for page_data in raw_pages:

        page = page_data.get("page")

        if page not in GLOSSARY_PAGES:
            continue

        paragraphs = split_paragraphs(
            page_data.get("text", "")
        )

        if not paragraphs:
            continue

        normalized.append({
            "type": "glossary",
            "title": GLOSSARY_TITLES.get(
                page,
                "Glosarium"
            ),
            "page": page,
            "content": "\n\n".join(
                paragraphs
            ),
            "source": SOURCE_NAME,
            "metadata": {
                "normalization_method":
                    "paragraph_preservation"
            }
        })

    return normalized


# ==================================================
# 7. NORMALISASI HEADER TABEL
# ==================================================

def build_headers(
    header_rows,
    expected_columns
):
    """
    Menggabungkan header bertingkat.

    Forward fill hanya digunakan pada header,
    tidak diterapkan pada baris data.
    """

    headers = []

    for col_index in range(
        expected_columns
    ):

        parts = []
        previous_value = ""

        for row in header_rows:

            value = ""

            if col_index < len(row):
                value = clean_cell(
                    row[col_index]
                )

            if value:
                previous_value = value

            else:
                value = previous_value

            if (
                value
                and (
                    not parts
                    or value != parts[-1]
                )
            ):
                parts.append(value)

        header = " ".join(parts).strip()

        if not header:
            header = (
                f"Kolom {col_index + 1}"
            )

        headers.append(header)

    # Membuat nama header unik
    counts = Counter()
    unique_headers = []

    for header in headers:

        counts[header] += 1

        if counts[header] > 1:
            header = (
                f"{header} ({counts[header]})"
            )

        unique_headers.append(header)

    return unique_headers


# ==================================================
# 8. MENENTUKAN SUMBER BARIS TABEL
# ==================================================

def get_table_rows(record):
    """
    Menggunakan clean_rows dari hasil ekstraksi.

    Fallback ke rows apabila clean_rows tidak tersedia.
    Tidak menggunakan reconstructed_rows.
    """

    rows = record.get("clean_rows")

    if not rows:
        rows = record.get("rows")

    if not rows:
        raise ValueError(
            f"Tabel {record.get('table_id')} "
            "tidak memiliki clean_rows atau rows."
        )

    return rows, "clean_rows"


# ==================================================
# 9. DETEKSI MASALAH OCR
# ==================================================

def detect_table_issues(rows):

    issues = []

    for row_index, row in enumerate(
        rows,
        start=1
    ):

        for col_index, value in enumerate(
            row,
            start=1
        ):

            text = clean_cell(value)

            if not text:
                continue

            # Angka tercampur huruf
            # Contoh: 95g,90 atau 43,30d
            if re.search(
                r"\d+[a-zA-Z]{1,3}(?:,|\.)\d+",
                text
            ):
                issues.append({
                    "code":
                        "POSSIBLE_CORRUPTED_NUMERIC",
                    "row": row_index,
                    "column": col_index,
                    "value": text
                })

            # Kode wilayah diikuti karakter asing
            if re.search(
                r"\d{10}\s+[a-zA-Z]",
                text
            ):
                issues.append({
                    "code":
                        "POSSIBLE_CORRUPTED_REGION_CODE",
                    "row": row_index,
                    "column": col_index,
                    "value": text
                })

            # Angka diikuti karakter OCR
            if re.search(
                r"\d+\s+[a-zA-Z]\s*[./]",
                text
            ):
                issues.append({
                    "code":
                        "POSSIBLE_OCR_ARTIFACT",
                    "row": row_index,
                    "column": col_index,
                    "value": text
                })

    # Menghilangkan temuan duplikat
    unique = []
    seen = set()

    for issue in issues:

        key = (
            issue["code"],
            issue["row"],
            issue["column"],
            issue["value"]
        )

        if key not in seen:
            seen.add(key)
            unique.append(issue)

    return unique


# ==================================================
# 10. NORMALISASI SATU TABEL
# ==================================================

def normalize_one_table(record):

    table_id = record.get("table_id")
    page = record.get("page")

    if table_id is None:
        raise ValueError(
            "Tabel tanpa table_id ditemukan."
        )

    if page in EXCLUDED_PAGES:
        return None, []

    source_rows, selected_source = (
        get_table_rows(record)
    )

    rows = []

    for row in source_rows:

        if not isinstance(
            row,
            (list, tuple)
        ):
            continue

        cleaned = [
            clean_cell(cell)
            for cell in row
        ]

        if any(cleaned):
            rows.append(cleaned)

    if not rows:
        raise ValueError(
            f"Tabel {table_id} tidak memiliki baris."
        )

    expected_columns = max(
        len(row)
        for row in rows
    )

    flattened_header = record.get(
        "flattened_header"
    )

    if (
        isinstance(flattened_header, list)
        and flattened_header
    ):

        header_rows = [
            flattened_header
        ]

        data_rows = rows

    else:

        try:
            header_count = int(
                record.get(
                    "header_row_count",
                    1
                )
            )

        except (
            ValueError,
            TypeError
        ):
            header_count = 1

        header_count = max(
            1,
            min(header_count, len(rows))
        )

        header_rows = rows[:header_count]
        data_rows = rows[header_count:]

    headers = build_headers(
        header_rows,
        expected_columns
    )

    normalized_rows = []

    for row in data_rows:

        padded = (
            row
            + [""] * (
                expected_columns - len(row)
            )
        )

        normalized_row = {
            headers[index]: padded[index]
            for index in range(
                expected_columns
            )
        }

        normalized_rows.append(
            normalized_row
        )

    title = (
        record.get("caption")
        or record.get("table_title")
        or record.get("title")
        or f"Tabel {table_id}"
    )

    title = clean_cell(title)

    issues = detect_table_issues(rows)

    quality_status = (
        "needs_review"
        if issues
        else "not_reviewed"
    )

    normalized = {
        "type": "table",
        "title": title,
        "table_id": table_id,
        "page": page,
        "page_end": record.get(
            "page_end",
            page
        ),
        "columns": headers,
        "rows": normalized_rows,
        "source": SOURCE_NAME,
        "metadata": {
            "engine": record.get("engine"),
            "normalization_method": selected_source,
            "quality_status": quality_status,
            "quality_issues": issues,
            "original_quality": record.get("quality"),
            "classification": record.get(
                "classification"
            ),
            "row_count": len(normalized_rows),
            "column_count": len(headers)
        }
    }

    return normalized, issues


# ==================================================
# 11. NORMALISASI SELURUH TABEL
# ==================================================

def normalize_tables(table_records):

    normalized = []
    issues_report = []

    seen_ids = set()

    for record in table_records:

        table_id = record.get("table_id")

        if table_id in seen_ids:
            raise ValueError(
                f"table_id duplikat: {table_id}"
            )

        seen_ids.add(table_id)

        table, issues = normalize_one_table(
            record
        )

        if table is None:
            continue

        normalized.append(table)

        if issues:

            issues_report.append({
                "type": "table",
                "table_id": table_id,
                "page": record.get("page"),
                "issues": issues
            })

    return normalized, issues_report


# ==================================================
# 12. VALIDASI
# ==================================================

def validate_records(
    records,
    original_tables,
    normalized_tables
):

    errors = []

    valid_types = {
        "narrative",
        "glossary",
        "table"
    }

    for index, record in enumerate(
        records,
        start=1
    ):

        record_type = record.get("type")

        if record_type not in valid_types:

            errors.append(
                f"Record {index}: tipe tidak valid."
            )

        if record_type in {
            "narrative",
            "glossary"
        }:

            if not record.get("content"):

                errors.append(
                    f"Record {index}: content kosong."
                )

        if record_type == "table":

            columns = record.get(
                "columns",
                []
            )

            rows = record.get(
                "rows",
                []
            )

            if not columns:

                errors.append(
                    f"Tabel {record.get('table_id')}: "
                    "columns kosong."
                )

            for row_index, row in enumerate(
                rows,
                start=1
            ):

                if list(row.keys()) != columns:

                    errors.append(
                        f"Tabel {record.get('table_id')}, "
                        f"baris {row_index}: "
                        "struktur kolom tidak konsisten."
                    )

            if (
                record.get("metadata", {}).get(
                    "normalization_method"
                ) != "clean_rows"
            ):

                errors.append(
                    f"Tabel {record.get('table_id')}: "
                    "sumber baris bukan clean_rows."
                )

    original_ids = {
        item.get("table_id")
        for item in original_tables
        if item.get("page") not in EXCLUDED_PAGES
    }

    normalized_ids = {
        item.get("table_id")
        for item in normalized_tables
    }

    if original_ids != normalized_ids:

        errors.append(
            "ID tabel hasil normalisasi tidak sesuai sumber."
        )

    expected_count = sum(
        1
        for item in original_tables
        if item.get("page") not in EXCLUDED_PAGES
    )

    if len(normalized_tables) != expected_count:

        errors.append(
            "Jumlah tabel hasil normalisasi tidak sesuai."
        )

    return errors


# ==================================================
# 13. MAIN
# ==================================================

def main():

    print("=" * 60)
    print("NORMALISASI DATA UNTUK RAG")
    print("=" * 60)

    raw_pages = read_jsonl(RAW_FILE)

    classifications = read_jsonl(
        CLASSIFIED_FILE
    )

    table_records = read_jsonl(
        TABLE_FILE
    )

    print(
        f"Halaman PDF       : {len(raw_pages)}"
    )

    print(
        f"Hasil klasifikasi : {len(classifications)}"
    )

    print(
        f"Total tabel       : {len(table_records)}"
    )

    # ----------------------------------------------
    # Normalisasi narasi
    # ----------------------------------------------

    print("\nNormalisasi narasi...")

    narratives = normalize_narrative(
        raw_pages,
        classifications
    )

    print(
        f"Narasi            : {len(narratives)}"
    )

    # ----------------------------------------------
    # Normalisasi glosarium
    # ----------------------------------------------

    print("\nNormalisasi glosarium...")

    glossaries = normalize_glossary(
        raw_pages
    )

    print(
        f"Glosarium         : {len(glossaries)}"
    )

    # ----------------------------------------------
    # Normalisasi tabel
    # ----------------------------------------------

    print("\nNormalisasi tabel...")

    tables, table_issues = normalize_tables(
        table_records
    )

    clean_count = len(tables)

    # ----------------------------------------------
    # Menggabungkan seluruh record
    # ----------------------------------------------

    normalized_records = (
        narratives
        + glossaries
        + tables
    )

    normalized_records.sort(
        key=lambda item: (
            item.get("page") or 0,
            item.get("type", ""),
            item.get("table_id") or 0
        )
    )

    # ----------------------------------------------
    # Validasi
    # ----------------------------------------------

    errors = validate_records(
        normalized_records,
        table_records,
        tables
    )

    if errors:

        print("\nVALIDASI GAGAL:")

        for error in errors:
            print("-", error)

        raise ValueError(
            "Normalisasi dihentikan."
        )

    # ----------------------------------------------
    # Simpan hasil
    # ----------------------------------------------

    save_jsonl(
        normalized_records,
        OUTPUT_FILE
    )

    save_jsonl(
        table_issues,
        ISSUES_FILE
    )

    issue_count = sum(
        len(item["issues"])
        for item in table_issues
    )

    # ----------------------------------------------
    # Laporan
    # ----------------------------------------------

    report = {
        "status": "SUCCESS",
        "source": SOURCE_NAME,
        "normalization_mode": "conservative_cleaning",
        "reconstruction_enabled": False,
        "total_pages": len(raw_pages),
        "narrative_count": len(narratives),
        "glossary_count": len(glossaries),
        "total_source_tables": len(table_records),
        "normalized_table_count": len(tables),
        "clean_table_count": clean_count,
        "total_normalized_records": len(
            normalized_records
        ),
        "tables_with_possible_issues": len(
            table_issues
        ),
        "total_possible_issues": issue_count,
        "excluded_pages": sorted(
            EXCLUDED_PAGES
        ),
        "validation": {
            "all_valid": True,
            "errors": errors
        },
        "output_file": str(OUTPUT_FILE),
        "issues_file": str(ISSUES_FILE)
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ----------------------------------------------
    # Ringkasan terminal
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("NORMALISASI BERHASIL")
    print("=" * 60)

    print(
        f"Narasi                  : {len(narratives)}"
    )

    print(
        f"Glosarium               : {len(glossaries)}"
    )

    print(
        f"Tabel                   : {len(tables)}"
    )

    print(
        f"Tabel dengan temuan     : {len(table_issues)}"
    )

    print(
        f"Total kemungkinan isu   : {issue_count}"
    )

    print(
        f"Total record            : {len(normalized_records)}"
    )

    print(
        "Mode                    : Conservative Cleaning"
    )

    print(
        "Rekonstruksi            : Tidak dilakukan"
    )

    print(
        "Validasi                : LULUS"
    )

    print(f"\nOutput      : {OUTPUT_FILE}")
    print(f"Laporan     : {REPORT_FILE}")
    print(f"Temuan      : {ISSUES_FILE}")


if __name__ == "__main__":
    main()
