
import json
from pathlib import Path

import pdfplumber

PDF_PATH = Path(
    "../knowledge/kecamatan-kapuas-dalam-angka-2025 (1).pdf"
)

OUTPUT_PATH = Path("pdfplumber_test.jsonl")

# Nomor halaman PDF menggunakan hitungan mulai dari 1.
PAGES = [100, 101, 102, 129, 178]

SETTINGS = {
    "lines": {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 5,
        "edge_min_length": 3,
    },
    "text": {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 5,
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
        "text_x_tolerance": 3,
        "text_y_tolerance": 3,
    },
}

results = []

with pdfplumber.open(PDF_PATH) as pdf:
    for page_number in PAGES:
        page = pdf.pages[page_number - 1]

        print(f"\n{'=' * 60}")
        print(f"HALAMAN {page_number}")

        for method, settings in SETTINGS.items():
            tables = page.extract_tables(table_settings=settings)

            print(f"\nMetode: {method}")
            print(f"Jumlah tabel: {len(tables)}")

            for table_index, table in enumerate(tables, start=1):
                rows = [
                    [cell.strip() if cell else "" for cell in row]
                    for row in table
                ]

                print(f"\nTabel {table_index}")
                print(f"Jumlah baris: {len(rows)}")

                for row in rows[:5]:
                    print(row)

                results.append({
                    "page": page_number,
                    "method": method,
                    "table_index": table_index,
                    "row_count": len(rows),
                    "rows": rows,
                })

with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"\nHasil disimpan di: {OUTPUT_PATH}")
