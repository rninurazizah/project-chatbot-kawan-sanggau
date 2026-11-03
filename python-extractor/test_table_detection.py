
import pymupdf

PDF_PATH = "knowledge/kecamatan-kapuas-dalam-angka-2025 (1).pdf"

# Nomor halaman PDF yang ingin diuji (dimulai dari 1)
TEST_PAGES = [53, 76, 100, 109, 143, 165, 186, 188, 194, 203, 207]

pdf = pymupdf.open(PDF_PATH)

print("=" * 65)
print("PENGUJIAN DETEKSI TABEL PYMUPDF")
print("=" * 65)

for page_number in TEST_PAGES:
    page = pdf[page_number - 1]

    print(f"\nHalaman PDF: {page_number}")

    try:
        finder = page.find_tables()
        tables = finder.tables

        print("Jumlah tabel terdeteksi:", len(tables))

        for index, table in enumerate(tables, start=1):
            extracted = table.extract()

            print(f"\nTabel ke-{index}")
            print("Jumlah baris:", len(extracted))
            print("Jumlah kolom:", table.col_count)

            for row in extracted[:4]:
                print(row)

    except Exception as error:
        print("Gagal mendeteksi tabel:", error)

pdf.close()
