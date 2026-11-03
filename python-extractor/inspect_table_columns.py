
import json

INPUT_FILE = "extracted/raw_pages.jsonl"
TARGET_PAGE = 100

# Rentang kolom berdasarkan hasil inspeksi koordinat
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

# Kelompokkan kata berdasarkan posisi vertikal
rows = []

for word in words:
    matched = None

    for row in rows:
        if abs(word["y0"] - row["y"]) <= 3:
            matched = row
            break

    if matched is None:
        rows.append({
            "y": word["y0"],
            "words": [word]
        })
    else:
        matched["words"].append(word)

print(f"INSPEKSI KOLOM HALAMAN {TARGET_PAGE}")
print("=" * 90)

for row in rows:
    result = {}

    for column_name, (xmin, xmax) in COLUMNS.items():
        selected = [
            word for word in row["words"]
            if xmin <= (word["x0"] + word["x1"]) / 2 < xmax
        ]

        selected.sort(key=lambda word: word["x0"])

        result[column_name] = " ".join(
            word["text"] for word in selected
        )

    # Tampilkan baris yang memiliki isi pada area tabel
    if any(result.values()) and 120 <= row["y"] <= 415:
        print(f"\nY={row['y']:.2f}")

        for column_name, value in result.items():
            if value:
                print(f"  {column_name}: {value}")
