
import json

INPUT_FILE = "extracted/raw_pages.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    pages = [json.loads(line) for line in file]

page = next(p for p in pages if p["page"] == 53)

words = sorted(
    page["words"],
    key=lambda w: (w["y0"], w["x0"])
)

# Tampilkan kata di area tabel halaman 53
for word in words:

    if 125 <= word["y0"] <= 490:

        print(
            f"Y={word['y0']:7.2f} | "
            f"X={word['x0']:7.2f} | "
            f"X1={word['x1']:7.2f} | "
            f"TEXT={word['text']}"
        )
