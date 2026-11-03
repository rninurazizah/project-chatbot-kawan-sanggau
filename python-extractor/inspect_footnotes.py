
import json

INPUT_FILE = "extracted/raw_pages.jsonl"
TARGET_PAGE = 100

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    pages = [json.loads(line) for line in file if line.strip()]

page = next(p for p in pages if p["page"] == TARGET_PAGE)

# Periksa kata pada kategori dan nilai yang memiliki catatan kaki
targets = [
    (40, 92, 120, 155),    # Taman Kanak-Kanak
    (40, 92, 155, 180),    # Raudatul Athfal
    (92, 137, 155, 175),   # Tanda – dan angka 3
]

for xmin, xmax, ymin, ymax in targets:
    print(f"\nAREA X={xmin}-{xmax}, Y={ymin}-{ymax}")
    print("-" * 70)

    selected = [
        word for word in page["words"]
        if xmin <= word["x0"] <= xmax
        and ymin <= word["y0"] <= ymax
    ]

    selected.sort(key=lambda w: (w["y0"], w["x0"]))

    for word in selected:
        print(
            f"Teks={word['text']!r} | "
            f"X={word['x0']:.2f}-{word['x1']:.2f} | "
            f"Y={word['y0']:.2f}-{word['y1']:.2f}"
        )
