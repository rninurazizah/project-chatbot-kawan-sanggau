
import json

INPUT_FILE = "extracted/raw_pages.jsonl"
TARGET_PAGE = 100

COLUMNS = {
    "tingkat_pendidikan": (40, 92),
    "negeri_2023_2024": (92, 137),
    "negeri_2024_2025": (137, 183),
    "swasta_2023_2024": (183, 228),
    "swasta_2024_2025": (228, 274),
    "jumlah_2023_2024": (274, 320),
    "jumlah_2024_2025": (320, 370),
}

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    pages = [json.loads(line) for line in file if line.strip()]

page = next(item for item in pages if item["page"] == TARGET_PAGE)

words = sorted(
    page["words"],
    key=lambda word: (word["y0"], word["x0"])
)

# 1. Kelompokkan kata berdasarkan posisi vertikal
rows = []

for word in words:
    matched = None

    for row in rows:
        if abs(word["y0"] - row["y"]) <= 3:
            matched = row
            break

    if matched is None:
        rows.append({"y": word["y0"], "words": [word]})
    else:
        matched["words"].append(word)

# 2. Pisahkan setiap baris berdasarkan kolom
table_rows = []

for row in rows:
    if not 120 <= row["y"] <= 415:
        continue

    result = {}

    for column_name, (xmin, xmax) in COLUMNS.items():
        selected = [
            word for word in row["words"]
            if xmin <= (word["x0"] + word["x1"]) / 2 < xmax
        ]

        selected.sort(key=lambda word: word["x0"])

        result[column_name] = " ".join(
            word["text"] for word in selected
        ).strip()

    if any(result.values()):
        table_rows.append({
            "y": row["y"],
            **result
        })

# 3. Gabungkan teks kategori sampai ditemukan baris angka
merged_rows = []
current_category = []

numeric_columns = list(COLUMNS.keys())[1:]

for row in table_rows:
    category = row["tingkat_pendidikan"]

    numeric_values = [
        row[column] for column in numeric_columns
    ]

    has_numeric_data = any(
        value and value != "https://sanggaukab.bps.go.id"
        for value in numeric_values
    )

    # Baris dengan angka menjadi penanda akhir satu entri
    if has_numeric_data:
        if category:
            current_category.append(category)

        merged_rows.append({
            "tingkat_pendidikan": " ".join(current_category),
            **{
                column: row[column]
                for column in numeric_columns
            }
        })

        current_category = []

    elif category:
        # Baris teks saja dianggap sebagai bagian nama kategori
        current_category.append(category)

# 4. Tampilkan hasil simulasi
print("HASIL PENGGABUNGAN BARIS TABEL")
print("=" * 90)

for index, row in enumerate(merged_rows, start=1):
    print(f"\nData {index}")
    for key, value in row.items():
        print(f"  {key}: {value}")
