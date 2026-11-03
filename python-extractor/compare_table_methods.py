
import json
from pathlib import Path

import pymupdf


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "knowledge"
DETECTION_FILE = BASE_DIR / "extracted" / "table_detection.jsonl"
OUTPUT_FILE = BASE_DIR / "extracted" / "table_method_comparison.jsonl"


def load_pdf():
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("Tidak ditemukan file PDF di folder knowledge.")

    if len(pdf_files) > 1:
        print("Beberapa PDF ditemukan:")
        for i, file in enumerate(pdf_files, start=1):
            print(f"{i}. {file.name}")

        choice = int(input("Pilih nomor PDF: "))
        return pdf_files[choice - 1]

    return pdf_files[0]


def load_candidate_pages():
    pages = []

    with open(DETECTION_FILE, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)

            if item.get("page_type") == "table_candidate":
                pages.append(item["page"])

    return pages


def inspect_with_find_tables(page):
    try:
        result = page.find_tables()
        tables = []

        for index, table in enumerate(result.tables, start=1):
            extracted = table.extract()

            tables.append({
                "table_index": index,
                "bbox": list(table.bbox) if table.bbox else None,
                "row_count": table.row_count,
                "column_count": table.col_count,
                "data": extracted
            })

        return {
            "success": True,
            "table_count": len(tables),
            "tables": tables
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
            "table_count": 0,
            "tables": []
        }


def inspect_with_coordinates(page):
    words = page.get_text("words")

    if not words:
        return {
            "word_count": 0,
            "row_count_estimate": 0,
            "horizontal_positions": 0
        }

    # Kelompokkan kata berdasarkan posisi vertikal.
    words = sorted(words, key=lambda word: (word[1], word[0]))

    rows = []

    for word in words:
        y_center = (word[1] + word[3]) / 2

        matched_row = None

        for row in rows:
            if abs(row["y"] - y_center) <= 3:
                matched_row = row
                break

        if matched_row is None:
            rows.append({
                "y": y_center,
                "words": [word]
            })
        else:
            matched_row["words"].append(word)

    # Hitung perkiraan posisi horizontal teks.
    x_positions = sorted(word[0] for word in words)

    clusters = []

    for x in x_positions:
        if not clusters or abs(x - clusters[-1][-1]) > 18:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    return {
        "word_count": len(words),
        "row_count_estimate": len(rows),
        "horizontal_positions": len(clusters)
    }


def main():
    pdf_path = load_pdf()
    candidate_pages = load_candidate_pages()

    print(f"PDF: {pdf_path.name}")
    print(f"Jumlah halaman kandidat: {len(candidate_pages)}")

    document = pymupdf.open(pdf_path)

    results = []

    for page_number in candidate_pages:
        if page_number < 1 or page_number > len(document):
            continue

        page = document[page_number - 1]

        automatic = inspect_with_find_tables(page)
        coordinates = inspect_with_coordinates(page)

        result = {
            "page": page_number,
            "find_tables": automatic,
            "coordinate_features": coordinates
        }

        results.append(result)

        print(
            f"Halaman {page_number}: "
            f"find_tables={automatic['table_count']} tabel, "
            f"baris={coordinates['row_count_estimate']}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    detected = sum(
        1 for result in results
        if result["find_tables"]["table_count"] > 0
    )

    print("\n" + "=" * 55)
    print("PENGUJIAN SELESAI")
    print("=" * 55)
    print(f"Halaman diuji             : {len(results)}")
    print(f"Halaman terdeteksi tabel  : {detected}")
    print(f"Tidak terdeteksi          : {len(results) - detected}")
    print(f"Output                    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
