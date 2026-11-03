
import json
import re
from pathlib import Path


INPUT_PATH = Path("extracted/raw_pages.jsonl")
OUTPUT_PATH = Path("extracted/classified_pages.jsonl")



import re


def classify_page(page):

    text = page.get("text", "").strip()
    words = page.get("words", [])

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    lower_text = text.lower()

    if len(words) < 5 or len(text) < 30:
        return "low_text"

    # 1. Identifikasi daftar isi
    toc_pattern = re.compile(
        r"\b\d+\.\d+\.\d+\b"
    )

    table_numbers = toc_pattern.findall(text)

    dotted_leaders = len(
        re.findall(r"\.{3,}", text)
    )

    if (
        "daftar isi" in lower_text
        or (
            len(table_numbers) >= 5
            and dotted_leaders >= 3
        )
    ):
        return "table_of_contents"

    # 2. Identifikasi kandidat tabel
    column_headers = re.findall(
        r"^\s*\(\d+\)\s*$",
        text,
        re.MULTILINE
    )

    has_table_title = bool(
        re.search(
            r"(tabel|lanjutan tabel)\s+\d+\.\d+",
            lower_text
        )
    )

    # Hitung baris yang berpotensi menjadi data tabel
    data_rows = 0

    for line in lines:

        if re.search(r"\d", line):
            data_rows += 1

    if (
        len(column_headers) >= 2
        or (
            has_table_title
            and data_rows >= 3
        )
    ):
        return "table_candidate"

    # 3. Identifikasi kandidat narasi
    paragraph_lines = [
        line for line in lines
        if len(line.split()) >= 8
    ]

    if len(paragraph_lines) >= 5:
        return "narrative_candidate"

    # 4. Belum dapat ditentukan
    return "needs_review"


    text = page.get("text", "").strip()
    words = page.get("words", [])

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    lower_text = text.lower()

    # Halaman tanpa teks yang cukup
    if len(words) < 5 or len(text) < 30:
        return "low_text"

    # Indikasi daftar isi
    toc_keywords = [
        "daftar isi",
        "contents",
        "tabel halaman"
    ]

    toc_score = sum(
        keyword in lower_text
        for keyword in toc_keywords
    )

    # Pola nomor tabel, misalnya 3.1.2
    table_number_pattern = re.compile(
        r"\b\d+\.\d+\.\d+\b"
    )

    table_numbers = table_number_pattern.findall(text)

    # Pola nomor kolom tabel
    column_header_pattern = re.compile(
        r"^\s*\(\d+\)\s*$",
        re.MULTILINE
    )

    column_headers = column_header_pattern.findall(text)

    # Indikasi tabel
    table_score = 0

    if table_numbers:
        table_score += 1

    if len(column_headers) >= 2:
        table_score += 2

    if len(words) > 30 and len(lines) > 10:
        table_score += 1

    # Indikasi narasi
    sentence_count = len(
        re.findall(r"[.!?]", text)
    )

    narrative_score = 0

    if sentence_count >= 5:
        narrative_score += 2

    if len(words) > 50 and len(lines) > 10:
        narrative_score += 1

    # Klasifikasi
    if toc_score > 0:
        page_type = "table_of_contents"

    elif table_score >= 2 and narrative_score >= 2:
        page_type = "mixed"

    elif table_score >= 2:
        page_type = "table_candidate"

    elif narrative_score >= 2:
        page_type = "narrative_candidate"

    else:
        page_type = "needs_review"

    return page_type


def classify_documents():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = {}

    with open(INPUT_PATH, "r", encoding="utf-8") as source, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as output:

        for line in source:

            page = json.loads(line)

            page_type = classify_page(page)

            result = {
                "page": page["page"],
                "page_type": page_type,
                "word_count": page["word_count"],
                "text_length": len(page["text"]),
                "text": page["text"]
            }

            output.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

            summary[page_type] = (
                summary.get(page_type, 0) + 1
            )

    print("\nHASIL KLASIFIKASI")
    print("=" * 40)

    for category, count in summary.items():
        print(f"{category}: {count}")

    print("\nOutput:", OUTPUT_PATH)


if __name__ == "__main__":
    classify_documents()
