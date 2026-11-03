
import json
from pathlib import Path
from collections import defaultdict


INPUT_PATH = Path("extracted/raw_pages.jsonl")

SAMPLE_PAGES = [53, 76]


def group_words_by_row(words, tolerance=3.5):

    # Urutkan berdasarkan posisi vertikal, kemudian horizontal
    words = sorted(
        words,
        key=lambda word: (
            word["y0"],
            word["x0"]
        )
    )

    rows = []

    for word in words:

        placed = False

        for row in rows:

            # Bandingkan posisi tengah vertikal
            row_y = sum(
                item["y0"] for item in row
            ) / len(row)

            if abs(word["y0"] - row_y) <= tolerance:
                row.append(word)
                placed = True
                break

        if not placed:
            rows.append([word])

    # Urutkan kata dalam setiap baris berdasarkan X
    for row in rows:
        row.sort(key=lambda word: word["x0"])

    return rows


def inspect_pages():

    with open(INPUT_PATH, "r", encoding="utf-8") as file:

        pages = {
            data["page"]: data
            for data in map(json.loads, file)
            if data["page"] in SAMPLE_PAGES
        }

    for page_number in SAMPLE_PAGES:

        page = pages.get(page_number)

        if not page:
            print(f"Halaman {page_number} tidak ditemukan")
            continue

        print("\n" + "=" * 90)
        print(f"INSPEKSI HALAMAN PDF: {page_number}")
        print("=" * 90)

        rows = group_words_by_row(page["words"])

        for row in rows:

            # Hilangkan baris yang kosong
            if not row:
                continue

            y_position = sum(
                word["y0"] for word in row
            ) / len(row)

            row_text = " ".join(
                word["text"] for word in row
            )

            positions = [
                round(word["x0"], 1)
                for word in row
            ]

            print(
                f"Y={y_position:7.2f} | "
                f"X={positions} | "
                f"{row_text}"
            )


if __name__ == "__main__":
    inspect_pages()
