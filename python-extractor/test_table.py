
import pymupdf
from pathlib import Path

PDF_PATH = Path(
    "knowledge/kecamatan-kapuas-dalam-angka-2025 (1).pdf"
)

pdf = pymupdf.open(PDF_PATH)
page = pdf[75]

print("PDF halaman: 76")

for strategy in ["lines", "text"]:

    print("\n" + "=" * 60)
    print("STRATEGI:", strategy)
    print("=" * 60)

    tables = page.find_tables(strategy=strategy)

    print("Jumlah tabel:", len(tables.tables))

    for index, table in enumerate(tables.tables, start=1):

        print(f"\nTABEL {index}")
        print("Jumlah baris:", table.row_count)
        print("Jumlah kolom:", table.col_count)

        rows = table.extract()

        for row in rows:
            print(row)

pdf.close()

