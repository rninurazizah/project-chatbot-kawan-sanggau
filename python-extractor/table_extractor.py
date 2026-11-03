
import json
import re
import unicodedata
from pathlib import Path


# =========================================================
# 1. KONFIGURASI FILE
# =========================================================

INPUT_FILE = Path("extracted/raw_pages.jsonl")
OUTPUT_FILE = Path("extracted/table_results.jsonl")


# =========================================================
# 2. KONFIGURASI POLA TABEL
# =========================================================

TABLE_PATTERNS = {
    "population_by_age": {
        "table_id": "3.1.2",
        "table_title": (
            "Jumlah Penduduk Berdasarkan Kelompok Umur "
            "dan Jenis Kelamin"
        ),
        "title_keywords": [
            "jumlah penduduk",
            "kelompok umur",
            "jenis kelamin"
        ],
        "columns": {
            "kelompok_umur": (40, 150),
            "laki_laki": (150, 225),
            "perempuan": (225, 295),
            "jumlah": (295, 380)
        },
        "extractor": "population"
    },

    "village_facilities": {
        "table_id": "2.1.1",
        "table_title": (
            "Jumlah Rukun Warga (RW), Rukun Tetangga (RT) "
            "dan Dusun Menurut Desa/Kelurahan"
        ),
        "title_keywords": [
            "jumlah rukun warga",
            "rukun tetangga",
            "dusun menurut desa"
        ],
        "columns": {
            "kode_wilayah": (60, 135),
            "desa": (135, 245),
            "rw": (245, 285),
            "rt": (285, 335),
            "dusun": (335, 390)
        },
        "extractor": "village"
    }
}


# =========================================================
# 3. NORMALISASI TEKS
# =========================================================

def clean_text(value):
    """
    Menghapus karakter Unicode tersembunyi,
    seperti zero-width dan word joiner.

    Karakter tanda pisah tetap dipertahankan.
    """

    value = str(value)

    value = "".join(
        char
        for char in value
        if unicodedata.category(char) != "Cf"
    )

    return value.strip()


# =========================================================
# 4. MEMBACA DATA HASIL EKSTRAKSI PDF
# =========================================================

def load_pages(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {file_path}"
        )

    pages = []

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                pages.append(json.loads(line))

            except json.JSONDecodeError as error:

                print(
                    f"JSON tidak valid pada baris "
                    f"{line_number}: {error}"
                )

    return pages


# =========================================================
# 5. DETEKSI TABEL BERDASARKAN JUDUL
# =========================================================

def detect_table(page, pattern):

    text = clean_text(
        page.get("text", "")
    ).lower()

    keywords = pattern["title_keywords"]

    return all(
        keyword in text
        for keyword in keywords
    )


# =========================================================
# 6. MENGELOMPOKKAN KATA BERDASARKAN KOORDINAT Y
# =========================================================

def group_words_by_y(words, tolerance=3.5):

    words = sorted(
        words,
        key=lambda word: (
            word["y0"],
            word["x0"]
        )
    )

    rows = []

    for word in words:

        matched_row = None

        for row in rows:

            if abs(word["y0"] - row["y"]) <= tolerance:

                matched_row = row
                break

        if matched_row is None:

            rows.append({
                "y": word["y0"],
                "words": [word]
            })

        else:

            matched_row["words"].append(word)

    # Mengurutkan kata dalam setiap baris berdasarkan X
    for row in rows:

        row["words"].sort(
            key=lambda word: word["x0"]
        )

    return rows


# =========================================================
# 7. MEMISAHKAN KATA BERDASARKAN KOLOM X
# =========================================================

def split_columns(row_words, column_ranges):

    columns = {
        name: []
        for name in column_ranges
    }

    for word in row_words:

        x = word["x0"]

        text = clean_text(
            word["text"]
        )

        for name, (xmin, xmax) in column_ranges.items():

            if xmin <= x < xmax:

                columns[name].append(text)
                break

    return {
        name: " ".join(values).strip()
        for name, values in columns.items()
    }


# =========================================================
# 8. NORMALISASI NILAI ANGKA
# =========================================================

def normalize_number(value):

    value = clean_text(value)

    # Mempertahankan tanda kosong sebagaimana dokumen asli
    if value in {"-", "–", "—", "..."}:
        return value

    # Contoh: 46.521 menjadi 46521
    if re.fullmatch(
        r"\d{1,3}(?:\.\d{3})+",
        value
    ):
        return int(
            value.replace(".", "")
        )

    # Angka biasa
    if value.isdigit():
        return int(value)

    # Nilai yang tidak dapat dikonversi tetap berupa teks
    return value


# =========================================================
# 9. VALIDASI BARIS TABEL PENDUDUK
# =========================================================

def is_valid_population_row(values):

    age = clean_text(
        values.get("kelompok_umur", "")
    )

    # Contoh: 0-4, 5-9, 75+, Jumlah
    if not re.fullmatch(
        r"\d{1,2}-\d{1,2}|\d{1,2}\+|Jumlah",
        age
    ):
        return False

    required_columns = [
        "kelompok_umur",
        "laki_laki",
        "perempuan",
        "jumlah"
    ]

    for key in required_columns:

        if not values.get(key):
            return False

    return True


# =========================================================
# 10. VALIDASI BARIS TABEL DESA
# =========================================================

def is_valid_village_row(values):

    code = clean_text(
        values.get("kode_wilayah", "")
    )

    village = clean_text(
        values.get("desa", "")
    )

    # Kode wilayah harus 10 digit
    if not re.fullmatch(r"\d{10}", code):
        return False

    # Nama desa harus tersedia
    if not village:
        return False

    # Memastikan nilai RW, RT, dan dusun tersedia
    for key in ["rw", "rt", "dusun"]:

        value = clean_text(
            values.get(key, "")
        )

        if not value:
            return False

        # Tanda pisah dan elipsis merupakan nilai yang sah
        if value in {"-", "–", "—", "..."}:
            continue

        # Nilai lainnya harus berupa angka
        if not re.fullmatch(r"\d+", value):
            return False

    return True


# =========================================================
# 11. EKSTRAKSI TABEL PENDUDUK
# =========================================================

def extract_population_table(page, pattern):

    words = page.get("words", [])

    rows = group_words_by_y(words)

    results = []

    for row in rows:

        values = split_columns(
            row["words"],
            pattern["columns"]
        )

        if not is_valid_population_row(values):
            continue

        results.append({

            "type": "table_row",

            "table_id": pattern["table_id"],

            "table_title": pattern["table_title"],

            "page": page["page"],

            "data": {

                "kelompok_umur": clean_text(
                    values["kelompok_umur"]
                ),

                "laki_laki": normalize_number(
                    values["laki_laki"]
                ),

                "perempuan": normalize_number(
                    values["perempuan"]
                ),

                "jumlah": normalize_number(
                    values["jumlah"]
                )
            }
        })

    return results


# =========================================================
# 12. EKSTRAKSI TABEL DESA
# =========================================================

def extract_village_table(page, pattern):

    words = page.get("words", [])

    rows = group_words_by_y(words)

    results = []

    for row in rows:

        values = split_columns(
            row["words"],
            pattern["columns"]
        )

        if not is_valid_village_row(values):
            continue

        results.append({

            "type": "table_row",

            "table_id": pattern["table_id"],

            "table_title": pattern["table_title"],

            "page": page["page"],

            "data": {

                "kode_wilayah": clean_text(
                    values["kode_wilayah"]
                ),

                "desa": clean_text(
                    values["desa"]
                ),

                "rw": normalize_number(
                    values["rw"]
                ),

                "rt": normalize_number(
                    values["rt"]
                ),

                "dusun": normalize_number(
                    values["dusun"]
                )
            }
        })

    return results


# =========================================================
# 13. VALIDASI HASIL TABEL PENDUDUK
# =========================================================

def validate_population_rows(rows):

    age_rows = [
        row
        for row in rows
        if row["data"]["kelompok_umur"] != "Jumlah"
    ]

    total_rows = [
        row
        for row in rows
        if row["data"]["kelompok_umur"] == "Jumlah"
    ]

    if len(age_rows) != 16:

        return False, (
            f"Ditemukan {len(age_rows)} kelompok umur, "
            "seharusnya 16."
        )

    if len(total_rows) != 1:

        return False, (
            "Jumlah baris total tidak sesuai."
        )

    # Memeriksa penjumlahan laki-laki dan perempuan
    for row in age_rows:

        data = row["data"]

        if (
            data["laki_laki"] + data["perempuan"]
            != data["jumlah"]
        ):

            return False, (
                f"Total tidak cocok pada kelompok "
                f"{data['kelompok_umur']}."
            )

    # Memeriksa total penduduk berdasarkan referensi
    total = total_rows[0]["data"]

    if (
        total["laki_laki"] != 46521
        or total["perempuan"] != 44311
        or total["jumlah"] != 90832
    ):

        return False, (
            "Total penduduk tidak sesuai referensi."
        )

    return True, (
        "Validasi tabel penduduk berhasil."
    )


# =========================================================
# 14. VALIDASI HASIL TABEL DESA
# =========================================================

def validate_village_rows(rows):

    if not rows:

        return False, (
            "Tidak ada baris desa yang ditemukan."
        )

    codes = [
        row["data"]["kode_wilayah"]
        for row in rows
    ]

    # Memeriksa kode wilayah duplikat
    if len(codes) != len(set(codes)):

        return False, (
            "Ditemukan kode wilayah duplikat."
        )

    # Jumlah baris yang diharapkan berdasarkan
    # inspeksi halaman 53 sebelumnya.
    expected_rows = 20

    if len(rows) != expected_rows:

        return False, (
            f"Ditemukan {len(rows)} baris, "
            f"seharusnya {expected_rows}."
        )

    return True, (
        "Validasi tabel desa berhasil."
    )


# =========================================================
# 15. MENYIMPAN HASIL EKSTRAKSI
# =========================================================

def save_results(rows):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        for row in rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )


# =========================================================
# 16. PROSES UTAMA
# =========================================================

def main():

    print("=" * 60)
    print("PDF TABLE EXTRACTOR")
    print("=" * 60)

    pages = load_pages(INPUT_FILE)

    print(
        f"Total halaman dibaca: {len(pages)}"
    )

    all_results = []

    # Memproses setiap pola tabel
    for table_name, pattern in TABLE_PATTERNS.items():

        print("\n" + "-" * 60)

        print(
            f"Memproses pola: {table_name}"
        )

        print("-" * 60)

        table_results = []

        for page in pages:

            # Deteksi halaman kandidat
            if not detect_table(page, pattern):
                continue

            # Memilih fungsi ekstraksi
            if pattern["extractor"] == "population":

                extracted = extract_population_table(
                    page,
                    pattern
                )

            elif pattern["extractor"] == "village":

                extracted = extract_village_table(
                    page,
                    pattern
                )

            else:
                continue

            if not extracted:

                print(
                    f"Halaman {page['page']}: "
                    "kandidat tidak menghasilkan data valid."
                )

                continue

            table_results.extend(extracted)

            print(
                f"Halaman {page['page']}: "
                f"{len(extracted)} baris ditemukan."
            )

        # Melakukan validasi berdasarkan jenis tabel
        if pattern["extractor"] == "population":

            valid, message = validate_population_rows(
                table_results
            )

        else:

            valid, message = validate_village_rows(
                table_results
            )

        print(message)

        # Hanya menyimpan hasil yang lolos validasi
        if valid:

            all_results.extend(
                table_results
            )

        else:

            print(
                "Peringatan: hasil pola ini tidak "
                "dimasukkan ke output."
            )

    # Menyimpan seluruh hasil
    save_results(all_results)

    print("\n" + "=" * 60)
    print("RINGKASAN EKSTRAKSI")
    print("=" * 60)

    print(
        f"Total baris berhasil: {len(all_results)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )


# =========================================================
# 17. MENJALANKAN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()
