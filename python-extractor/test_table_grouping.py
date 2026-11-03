
import json
import re

# ============================================================
# 1. KONFIGURASI
# ============================================================

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

NUMERIC_COLUMNS = list(COLUMNS.keys())[1:]


# ============================================================
# 2. MEMBACA DATA HALAMAN PDF
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    pages = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

page = next(
    (p for p in pages if p["page"] == TARGET_PAGE),
    None
)

if page is None:
    raise ValueError(f"Halaman {TARGET_PAGE} tidak ditemukan.")

words = sorted(
    page["words"],
    key=lambda word: (word["y0"], word["x0"])
)


# ============================================================
# 3. MENGELOMPOKKAN KATA BERDASARKAN KOORDINAT Y
# ============================================================

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


# ============================================================
# 4. MEMISAHKAN KATA BERDASARKAN KOLOM
# ============================================================

table_rows = []

for row in rows:

    # Batasi area isi tabel
    if not 120 <= row["y"] <= 415:
        continue

    result = {}

    for column_name, (xmin, xmax) in COLUMNS.items():

        selected = [
            word for word in row["words"]
            if xmin <= (word["x0"] + word["x1"]) / 2 < xmax
        ]

        selected.sort(
            key=lambda word: word["x0"]
        )

        result[column_name] = " ".join(
            word["text"] for word in selected
        ).strip()

    table_rows.append({
        "y": row["y"],
        **result
    })


# ============================================================
# 5. MENGELOMPOKKAN TEKS KATEGORI
# ============================================================

category_lines = [
    {
        "y": row["y"],
        "text": row["tingkat_pendidikan"]
    }
    for row in table_rows
    if row["tingkat_pendidikan"]
]

category_groups = []

for line in category_lines:

    if (
        not category_groups
        or line["y"] - category_groups[-1]["last_y"] > 11
    ):

        category_groups.append({
            "first_y": line["y"],
            "last_y": line["y"],
            "lines": [line["text"]]
        })

    else:

        group = category_groups[-1]

        group["last_y"] = line["y"]
        group["lines"].append(line["text"])


# ============================================================
# 6. MENDETEKSI BARIS ANGKA
# ============================================================

numeric_rows = []

for row in table_rows:

    values = [
        row[column]
        for column in NUMERIC_COLUMNS
    ]

    filled = [
        value
        for value in values
        if re.search(r"\d|–|-|\.\.\.", value)
    ]

    if len(filled) >= 4:
        numeric_rows.append(row)


# ============================================================
# 7. NORMALISASI TEKS DAN CATATAN KAKI
# ============================================================

def clean_footnote_marker(text):

    if not text:
        return text

    # Hapus karakter Unicode tak terlihat
    text = re.sub(
        r"[\u200b-\u200f\u2060\ufeff]",
        "",
        text
    )

    # Rapikan spasi
    text = re.sub(r"\s+", " ", text).strip()

    # Hapus penanda catatan kaki kategori
    text = re.sub(
        r"\((TK|RA|SD|MI|SMP|MTs|SMA|SMK|MA)\)[12]$",
        r"(\1)",
        text
    )

    # Hapus penanda catatan kaki pada nilai
    text = re.sub(r"^–3$", "–", text)

    # Perbaiki pemenggalan kata
    text = text.replace(
        "Kanak- Kanak",
        "Kanak-Kanak"
    )

    return text


# ============================================================
# 8. MEMASANGKAN KATEGORI DENGAN BARIS ANGKA
# ============================================================

merged_rows = []

used_numeric_rows = set()

for group in category_groups:

    center_y = (
        group["first_y"] + group["last_y"]
    ) / 2

    # Cari baris angka terdekat yang belum digunakan
    candidates = [
        (index, row)
        for index, row in enumerate(numeric_rows)
        if index not in used_numeric_rows
    ]

    if not candidates:
        raise ValueError(
            "Tidak ada baris angka yang tersisa."
        )

    nearest_index, nearest_row = min(
        candidates,
        key=lambda item: abs(item[1]["y"] - center_y)
    )

    used_numeric_rows.add(nearest_index)

    result = {
        "tingkat_pendidikan": " ".join(group["lines"])
    }

    for column in NUMERIC_COLUMNS:
        result[column] = nearest_row[column]

    # Normalisasi seluruh nilai
    for key in result:
        result[key] = clean_footnote_marker(
            result[key]
        )

    merged_rows.append(result)


# ============================================================
# 9. MENAMPILKAN HASIL EKSTRAKSI
# ============================================================

print("\nHASIL EKSTRAKSI TABEL")
print("=" * 100)

for index, row in enumerate(merged_rows, start=1):

    print(f"\nData {index}")

    for key, value in row.items():
        print(f"  {key}: {value}")


# ============================================================
# 10. VALIDASI JUMLAH DATA
# ============================================================

print("\nVALIDASI HASIL")
print("=" * 100)

expected_rows = 9

print(
    "Jumlah kategori ditemukan :",
    len(merged_rows)
)

print(
    "Jumlah kategori diharapkan:",
    expected_rows
)

if len(merged_rows) == expected_rows:
    print("Status jumlah kategori    : LULUS")
else:
    print("Status jumlah kategori    : GAGAL")


# ============================================================
# 11. VALIDASI NILAI BERDASARKAN TABEL ASLI
# ============================================================

expected_data = [
    [
        "Taman Kanak-Kanak (TK)",
        "1", "1", "18", "21", "19", "22"
    ],
    [
        "Raudatul Athfal (RA)",
        "–", "–", "3", "...", "3", "..."
    ],
    [
        "Sekolah Dasar (SD)",
        "69", "69", "3", "4", "72", "73"
    ],
    [
        "Madrasah Ibtidaiyah (MI)",
        "1", "...", "–", "...", "1", "..."
    ],
    [
        "Sekolah Menengah Pertama (SMP)",
        "15", "15", "8", "8", "23", "23"
    ],
    [
        "Madrasah Tsanawiyah (MTs)",
        "1", "...", "1", "...", "2", "..."
    ],
    [
        "Sekolah Menengah Atas (SMA)",
        "3", "3", "3", "2", "6", "5"
    ],
    [
        "Sekolah Menengah Kejuruan (SMK)",
        "1", "1", "3", "3", "4", "4"
    ],
    [
        "Madrasah Aliyah (MA)",
        "1", "...", "–", "...", "1", "..."
    ],
]

validation_errors = []

if len(merged_rows) == expected_rows:

    for index, (actual, expected) in enumerate(
        zip(merged_rows, expected_data),
        start=1
    ):

        actual_values = list(actual.values())

        if actual_values != expected:

            validation_errors.append({
                "baris": index,
                "aktual": actual_values,
                "diharapkan": expected
            })

    if validation_errors:

        print("Status nilai              : GAGAL")

        for error in validation_errors:

            print(f"\nBaris {error['baris']}")
            print("Aktual     :", error["aktual"])
            print("Diharapkan :", error["diharapkan"])

    else:
        print("Status nilai              : LULUS")

else:
    print("Status nilai              : TIDAK DIUJI")


# ============================================================
# 12. MENYIMPAN HASIL KE JSON
# ============================================================

OUTPUT_FILE = (
    f"extracted/table_page_{TARGET_PAGE}.json"
)

if len(merged_rows) == expected_rows and not validation_errors:

    output_data = {
        "page": TARGET_PAGE,
        "table_title": (
            "Jumlah Sekolah Menurut Tingkat Pendidikan "
            "di Kecamatan Kapuas, 2023/2024 dan 2024/2025"
        ),
        "columns": list(COLUMNS.keys()),
        "row_count": len(merged_rows),
        "data": merged_rows
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nJSON berhasil disimpan:", OUTPUT_FILE)


# ============================================================
# 13. RINGKASAN
# ============================================================

print("\nRINGKASAN")
print("=" * 100)

if len(merged_rows) == expected_rows and not validation_errors:

    print("EKSTRAKSI DAN VALIDASI BERHASIL.")

else:

    print("EKSTRAKSI BELUM LOLOS VALIDASI.")
