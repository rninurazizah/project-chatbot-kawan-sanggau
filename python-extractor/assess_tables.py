
from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import re

from common import read_jsonl, write_jsonl


# ============================================================
# KONFIGURASI
# ============================================================

def normalize(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def is_numeric(value):
    value = normalize(value)
    return bool(re.fullmatch(r"[\s(+-]*\d[\d.,/%\- ]*[\s)]*", value))


# ============================================================
# PEMERIKSAAN KUALITAS TABEL
# ============================================================

def assess_table_quality(table):

    rows = table.get("clean_rows") or table.get("rows") or []

    reasons = []

    if not rows:
        return {
            "status": "MANUAL",
            "reasons": ["Tidak ditemukan baris tabel"],
            "row_count": 0,
            "column_count": 0,
            "inconsistent_ratio": 0,
            "empty_row_ratio": 0,
            "fragmented_ratio": 0,
            "suspicious_ratio": 0,
        }

    rows = [
        [normalize(cell) for cell in row]
        for row in rows
        if isinstance(row, list)
        and any(normalize(cell) for cell in row)
    ]

    if not rows:
        return {
            "status": "MANUAL",
            "reasons": ["Seluruh baris tabel kosong"],
            "row_count": 0,
            "column_count": 0,
            "inconsistent_ratio": 0,
            "empty_row_ratio": 0,
            "fragmented_ratio": 0,
            "suspicious_ratio": 0,
        }

    # --------------------------------------------------------
    # 1. Konsistensi jumlah kolom
    # --------------------------------------------------------

    widths = [len(row) for row in rows]

    width_counts = Counter(widths)
    common_width, _ = width_counts.most_common(1)[0]

    inconsistent_count = sum(
        width != common_width for width in widths
    )

    inconsistent_ratio = inconsistent_count / len(rows)

    if inconsistent_ratio > 0.30:
        reasons.append(
            f"Jumlah kolom tidak konsisten: {inconsistent_count} baris"
        )

    # --------------------------------------------------------
    # 2. Baris dengan terlalu banyak sel kosong
    # --------------------------------------------------------

    sparse_count = 0

    for row in rows:
        nonempty = sum(bool(cell) for cell in row)

        if len(row) >= 3 and nonempty <= 1:
            sparse_count += 1

    empty_row_ratio = sparse_count / len(rows)

    if empty_row_ratio > 0.15:
        reasons.append(
            f"Baris dengan sedikit isi: {sparse_count} baris"
        )

    # --------------------------------------------------------
    # 3. Indikasi teks terfragmentasi
    # --------------------------------------------------------

    fragmented_count = 0

    for row in rows:
        cells = [cell for cell in row if cell]

        if len(row) >= 3 and len(cells) == 1:
            value = cells[0]

            if (
                len(value) <= 25
                and not is_numeric(value)
            ):
                fragmented_count += 1

    fragmented_ratio = fragmented_count / len(rows)

    if fragmented_ratio > 0.10:
        reasons.append(
            f"Indikasi teks terfragmentasi: {fragmented_count} baris"
        )

    # --------------------------------------------------------
    # 4. Indikasi OCR pada sel
    # --------------------------------------------------------

    suspicious_count = 0
    total_cells = 0

    for row in rows:
        for cell in row:
            if not cell:
                continue

            total_cells += 1

            # Campuran huruf dan angka perlu diperiksa.
            # Tidak otomatis dianggap salah karena kode wilayah
            # atau simbol tertentu bisa memiliki karakter khusus.
            if (
                re.search(r"\d", cell)
                and re.search(r"[a-zA-Z]", cell)
            ):
                suspicious_count += 1

    suspicious_ratio = (
        suspicious_count / total_cells
        if total_cells else 0
    )

    if suspicious_ratio > 0.10:
        reasons.append(
            f"Sel mengandung campuran huruf dan angka: {suspicious_count} sel"
        )

    # --------------------------------------------------------
    # 5. Deteksi baris dengan satu sel berisi teks panjang
    # --------------------------------------------------------

    long_single_cell_count = 0

    for row in rows:
        cells = [cell for cell in row if cell]

        if len(row) >= 3 and len(cells) == 1:
            if len(cells[0]) > 25:
                long_single_cell_count += 1

    if long_single_cell_count > 0:
        reasons.append(
            f"Baris dengan teks panjang hanya pada satu sel: "
            f"{long_single_cell_count} baris"
        )

    # --------------------------------------------------------
    # 6. Penentuan status
    # --------------------------------------------------------

    # Bobot indikasi, bukan kepastian bahwa tabel rusak.
    score = 0

    if inconsistent_ratio > 0.30:
        score += 3

    if empty_row_ratio > 0.15:
        score += 2

    if fragmented_ratio > 0.10:
        score += 2

    if suspicious_ratio > 0.10:
        score += 1

    if long_single_cell_count > 0:
        score += 1

    if inconsistent_ratio > 0.60 and len(rows) >= 3:
        status = "MANUAL"
    elif score >= 5:
        status = "REKONSTRUKSI"
    elif score >= 1:
        status = "PERIKSA"
    else:
        status = "AMAN"

    return {
        "status": status,
        "reasons": reasons,
        "quality_score": score,
        "row_count": len(rows),
        "column_count": common_width,
        "inconsistent_ratio": round(inconsistent_ratio, 3),
        "empty_row_ratio": round(empty_row_ratio, 3),
        "fragmented_ratio": round(fragmented_ratio, 3),
        "suspicious_ratio": round(suspicious_ratio, 3),
    }


# ============================================================
# PEMERIKSAAN SELURUH TABEL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Pemeriksaan kualitas struktur tabel"
    )

    parser.add_argument(
        "--dir",
        required=True,
        help="Direktori yang berisi tables_clean.jsonl"
    )

    args = parser.parse_args()

    directory = Path(args.dir)

    input_path = directory / "tables_clean.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {input_path.resolve()}"
        )

    tables = read_jsonl(input_path)

    if not tables:
        raise ValueError("Tidak ada data tabel dalam file JSONL.")

    results = []
    summary = Counter()

    # --------------------------------------------------------
    # Assess setiap tabel
    # --------------------------------------------------------

    for table in tables:

        assessment = assess_table_quality(table)

        result = {
            "table_id": table.get("table_id"),
            "page": table.get("page"),
            "page_end": table.get(
                "page_end",
                table.get("page")
            ),
            "caption": table.get("caption", ""),
            "table_number": table.get("table_number"),
            "is_continuation": table.get("is_continuation", False),
            **assessment,
        }

        results.append(result)
        summary[assessment["status"]] += 1

    # --------------------------------------------------------
    # Simpan JSONL
    # --------------------------------------------------------

    json_path = directory / "table_quality_report.jsonl"

    write_jsonl(json_path, results)

    # --------------------------------------------------------
    # Simpan CSV
    # --------------------------------------------------------

    csv_path = directory / "table_quality_report.csv"

    fieldnames = list(results[0].keys())

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            row = result.copy()
            row["reasons"] = " | ".join(row["reasons"])
            writer.writerow(row)

    # --------------------------------------------------------
    # Tampilkan ringkasan
    # --------------------------------------------------------

    print("\n" + "=" * 50)
    print("HASIL ASSESSMENT KUALITAS TABEL")
    print("=" * 50)

    print(f"Total tabel       : {len(tables)}")
    print(f"Aman              : {summary['AMAN']}")
    print(f"Periksa           : {summary['PERIKSA']}")
    print(f"Rekonstruksi      : {summary['REKONSTRUKSI']}")
    print(f"Manual            : {summary['MANUAL']}")

    print("\nTABEL YANG MEMERLUKAN PEMERIKSAAN")
    print("-" * 50)

    for result in results:

        if result["status"] == "AMAN":
            continue

        print(
            f"\nID: {result['table_id']} | "
            f"Halaman: {result['page']}-{result['page_end']}"
        )

        print(f"Status: {result['status']}")
        print(f"Skor: {result['quality_score']}")

        for reason in result["reasons"]:
            print(f"  - {reason}")

    print("\n" + "=" * 50)
    print("LAPORAN BERHASIL DISIMPAN")
    print("=" * 50)

    print(f"JSONL: {json_path.resolve()}")
    print(f"CSV  : {csv_path.resolve()}")


if __name__ == "__main__":
    main()
