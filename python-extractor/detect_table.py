"""
Deteksi halaman dan kandidat tabel secara generik dari hasil ekstraksi PyMuPDF.

Input:
    extracted/raw_pages.jsonl
    Format setiap baris: {"page": int, "text": str, "words": [...]}

Output:
    extracted/table_detection.jsonl

Catatan:
- Skrip ini TIDAK menganggap semua halaman kandidat pasti berisi tabel.
- Skrip ini belum mengekstrak isi tabel. Tugasnya mendeteksi halaman dan
  menyimpan bukti/fitur agar tahap ekstraksi dapat memilih strategi.
- Tidak ada nomor halaman atau koordinat khusus publikasi yang di-hardcode.
"""

import json
import re
from collections import Counter
from pathlib import Path

INPUT_FILE = Path("extracted/raw_pages.jsonl")
OUTPUT_FILE = Path("extracted/table_detection.jsonl")

# Kata-kata yang sering muncul pada header tabel statistik.
HEADER_TERMS = {
    "no", "kode", "wilayah", "desa", "kelurahan", "kecamatan",
    "jumlah", "tahun", "jenis", "kelompok", "umur", "laki-laki",
    "perempuan", "total", "persentase", "satuan", "indikator",
    "uraian", "kategori", "masjid", "mushola", "gereja",
}

# Pola umum teks tabel: nomor urut, tahun, kode wilayah, angka/persentase.
CODE_RE = re.compile(r"(?<!\d)\d{10}(?!\d)")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}(?:/\d{2,4})?\b")
NUMBER_RE = re.compile(r"(?<![\w])[-–—]?\d+(?:[.,]\d+)?%?(?![\w])")


def load_pages(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan: {path}\n"
            "Pastikan extract.py sudah menghasilkan extracted/raw_pages.jsonl."
        )

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[PERINGATAN] JSON rusak pada baris {line_number}: {exc}")


def word_rows(words, y_tolerance=3.0):
    """Kelompokkan kata berdasarkan kedekatan koordinat vertikal."""
    rows = []
    for word in sorted(words or [], key=lambda w: (float(w.get("y0", 0)), float(w.get("x0", 0)))):
        y = float(word.get("y0", 0))
        target = None
        for row in rows:
            if abs(y - row["y"]) <= y_tolerance:
                target = row
                break
        if target is None:
            rows.append({"y": y, "words": [word]})
        else:
            target["words"].append(word)
    return rows


def get_features(page):
    text = page.get("text") or ""
    words = page.get("words") or []
    rows = word_rows(words)

    nonempty_rows = [
        row for row in rows
        if any(str(w.get("text", "")).strip() for w in row["words"])
    ]

    row_token_counts = [
        len(row["words"]) for row in nonempty_rows
    ]

    # Perkiraan kolom: kelompokkan pusat-x kata ke posisi horizontal.
    x_centers = sorted(
        (float(w.get("x0", 0)) + float(w.get("x1", 0))) / 2
        for w in words
        if "x0" in w and "x1" in w
    )
    x_clusters = []
    for x in x_centers:
        if not x_clusters or abs(x - x_clusters[-1]) > 18:
            x_clusters.append(x)

    header_hits = sum(
        1 for token in re.findall(r"[a-zA-ZÀ-ÿ-]+", text.lower())
        if token in HEADER_TERMS
    )

    numeric_tokens = NUMBER_RE.findall(text)
    code_count = len(CODE_RE.findall(text))
    year_count = len(YEAR_RE.findall(text))

    # Baris dengan banyak token dan variasi posisi-x dapat mengindikasikan tabel.
    repeated_row_shape = 0
    if len(row_token_counts) >= 4:
        counts = Counter(row_token_counts)
        repeated_row_shape = sum(
            count for token_count, count in counts.items()
            if token_count >= 3 and count >= 3
        )

    return {
        "word_count": len(words),
        "text_length": len(text),
        "row_count_estimate": len(nonempty_rows),
        "median_words_per_row": (
            sorted(row_token_counts)[len(row_token_counts) // 2]
            if row_token_counts else 0
        ),
        "horizontal_position_clusters": len(x_clusters),
        "header_term_hits": header_hits,
        "numeric_token_count": len(numeric_tokens),
        "region_code_count": code_count,
        "year_count": year_count,
        "repeated_row_shape_count": repeated_row_shape,
    }


def classify(features):
    """
    Heuristik konservatif untuk menentukan kandidat halaman tabel.
    Skor bukan jaminan bahwa halaman benar-benar berisi tabel.
    """
    score = 0
    reasons = []

    if features["word_count"] >= 50:
        score += 1
        reasons.append("cukup banyak kata")

    if features["numeric_token_count"] >= 12:
        score += 2
        reasons.append("banyak token numerik")

    if features["horizontal_position_clusters"] >= 5:
        score += 2
        reasons.append("sebaran posisi horizontal")

    if features["header_term_hits"] >= 3:
        score += 1
        reasons.append("memiliki istilah header tabel")

    if features["repeated_row_shape_count"] >= 4:
        score += 2
        reasons.append("pola baris berulang")

    if features["region_code_count"] >= 3:
        score += 2
        reasons.append("memiliki beberapa kode wilayah")

    if score >= 5:
        page_type = "table_candidate"
    elif score >= 3:
        page_type = "needs_review"
    else:
        page_type = "not_table_candidate"

    return page_type, score, reasons


def main():
    results = []

    for page in load_pages(INPUT_FILE):
        page_number = page.get("page")

        features = get_features(page)
        page_type, score, reasons = classify(features)

        results.append({
            "page": page_number,
            "page_type": page_type,
            "score": score,
            "reasons": reasons,
            "features": features,
        })

    results.sort(
        key=lambda item: (
            item["page"] if item["page"] is not None else 0
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    counts = Counter(
        item["page_type"] for item in results
    )

    print("DETEKSI HALAMAN SELESAI")
    print("=" * 60)
    print(f"Jumlah halaman diproses : {len(results)}")
    print(f"Kandidat tabel          : {counts['table_candidate']}")
    print(f"Perlu ditinjau          : {counts['needs_review']}")
    print(f"Bukan kandidat          : {counts['not_table_candidate']}")
    print(f"Output                  : {OUTPUT_FILE}")

    print("\n10 kandidat dengan skor tertinggi:")

    candidates = sorted(
        results,
        key=lambda item: item["score"],
        reverse=True
    )

    for item in candidates[:10]:
        print(
            f"Halaman {item['page']}: "
            f"skor={item['score']}, "
            f"jenis={item['page_type']}, "
            f"alasan={', '.join(item['reasons']) or '-'}"
        )


if __name__ == "__main__":
    main()
        