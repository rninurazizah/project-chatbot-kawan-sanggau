
import camelot
from pathlib import Path

PDF_PATH = Path(
    "../knowledge/kecamatan-kapuas-dalam-angka-2025 (1).pdf"
)

TEST_PAGES = [32, 76, 100, 143, 186]

tables = camelot.read_pdf(
    str(PDF_PATH),
    pages=",".join(map(str, TEST_PAGES)),
    flavor="stream"
)

for table in tables:
    print("\n" + "=" * 70)
    print(f"HALAMAN: {table.page}")
    print(f"UKURAN: {table.df.shape}")
    print(f"PARSING REPORT: {table.parsing_report}")

    print("\nKOORDINAT KOLOM:")
    print(table.cols)

    print("\nHASIL EKSTRAKSI:")
    for row in table.df.values.tolist():
        print(row)
