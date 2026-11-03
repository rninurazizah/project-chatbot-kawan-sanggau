
import json
import re

INPUT_FILE = "extracted/classified_pages.jsonl"

# Kata-kata yang sering muncul pada judul tabel
TABLE_TITLE_PATTERN = re.compile(
    r"\b(tabel|table)\s+\d+(?:\.\d+)*",
    re.IGNORECASE
)

# Membaca hasil klasifikasi halaman
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    pages = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

results = []

for page in pages:

    page_number = page.get("page")
    page_type = page.get("page_type", "unknown")
    text = page.get("text", "")

    # Memisahkan teks menjadi baris
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Mencari baris yang mengandung judul tabel
    titles = [
        line for line in lines
        if TABLE_TITLE_PATTERN.search(line)
    ]

    # Memasukkan halaman yang diklasifikasikan sebagai tabel
    # atau halaman yang memiliki pola judul tabel
    if page_type == "table_candidate" or titles:

        results.append({
            "page": page_number,
            "page_type": page_type,
            "word_count": page.get("word_count", 0),
            "titles": titles,
            "preview": lines[:8]
        })


# Menampilkan hasil
print("\nDAFTAR HALAMAN TERINDIKASI TABEL")
print("=" * 100)

print(f"Total halaman ditemukan: {len(results)}")

for item in results:

    print("\n" + "-" * 100)
    print(f"Halaman PDF : {item['page']}")
    print(f"Tipe        : {item['page_type']}")
    print(f"Jumlah kata : {item['word_count']}")

    if item["titles"]:
        print("Judul tabel:")
        for title in item["titles"]:
            print(f"  - {title}")
    else:
        print("Judul tabel  : Tidak terdeteksi")

    print("Cuplikan teks:")
    for line in item["preview"]:
        print(f"  {line}")


# Menyimpan hasil agar mudah diperiksa
OUTPUT_FILE = "extracted/table_page_candidates.json"

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\n" + "=" * 100)
print(f"Hasil disimpan ke: {OUTPUT_FILE}")
