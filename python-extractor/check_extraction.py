
import json
from pathlib import Path

INPUT_PATH = Path("extracted/documents.jsonl")

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    documents = [json.loads(line) for line in file]

# Periksa panjang teks setiap halaman
print("HALAMAN DENGAN TEKS TERBANYAK")
print("=" * 60)

sorted_docs = sorted(
    documents,
    key=lambda x: len(x["text"]),
    reverse=True
)

for doc in sorted_docs[:10]:
    print(
        f"PDF halaman {doc['page']}: "
        f"{len(doc['text'])} karakter"
    )

# Periksa beberapa halaman yang dipilih
for page_number in [14, 15, 75, 76, 77]:
    doc = next(
        (d for d in documents if d["page"] == page_number),
        None
    )

    if doc:
        print("\n" + "=" * 60)
        print("PDF HALAMAN:", page_number)
        print("=" * 60)
        print(doc["text"][:3000])
