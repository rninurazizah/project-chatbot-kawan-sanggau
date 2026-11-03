
import re
import camelot
from pathlib import Path

PDF_PATH = Path(
    "../knowledge/kecamatan-kapuas-dalam-angka-2025 (1).pdf"
)

PAGE = "186"


def clean_cell(value):
    if not value:
        return ""

    value = str(value)
    value = value.replace("\u2060", "")
    value = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def remove_noise_suffix(value):
    """
    Menghapus token satu huruf di akhir teks.
    Contoh: 'Darat dan air b' -> 'Darat dan air'
    """
    value = clean_cell(value)

    # Hapus token satu huruf di bagian akhir
    value = re.sub(r"\s+[a-zA-Z]$", "", value)

    return value.strip()


def extract_table():
    tables = camelot.read_pdf(
        str(PDF_PATH),
        pages=PAGE,
        flavor="stream"
    )

    if not tables:
        raise ValueError("Tabel tidak ditemukan.")

    return tables[0]


def main():
    table = extract_table()

    cleaned_rows = []

    for row in table.df.values.tolist():
        row = [clean_cell(cell) for cell in row]

        # Ambil hanya baris yang memiliki kode wilayah
        if not re.fullmatch(r"\d{10}", row[0]):
            continue

        # Struktur halaman 186:
        # Kolom 0 = kode wilayah
        # Kolom 1 = desa
        # Kolom 2 = jenis prasarana
        # Kolom 4 = keberadaan angkutan umum
        village = remove_noise_suffix(row[1])

# Membersihkan karakter pengganggu pada nama desa
        village = village.rstrip("/").strip()

        cleaned_row = [
            row[0],
            village,
            remove_noise_suffix(row[2]),
            remove_noise_suffix(row[4])
        ]
       

        cleaned_rows.append(cleaned_row)

    print("\nHASIL CLEANING HALAMAN 186")
    print("-" * 100)

    for row in cleaned_rows:
        print(row)

    print("\nJumlah baris data:", len(cleaned_rows))


if __name__ == "__main__":
    main()
