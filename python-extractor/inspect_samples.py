
import pymupdf
from pathlib import Path

PDF_PATH = Path(
    "knowledge/kecamatan-kapuas-dalam-angka-2025 (1).pdf"
)

OUTPUT_PATH = Path("extracted/inspect_samples.txt")

SAMPLE_PAGES = [
    53, 69, 70, 75, 76, 77,
    81, 100, 109, 143, 186, 188, 194
]

pdf = pymupdf.open(PDF_PATH)

# Pastikan folder output tersedia
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    for page_number in SAMPLE_PAGES:
        page = pdf[page_number - 1]
        text = page.get_text("text")

        file.write("\n" + "=" * 80 + "\n")
        file.write(f"HALAMAN PDF: {page_number}\n")
        file.write("=" * 80 + "\n\n")
        file.write(text)
        file.write("\n\n")

pdf.close()

print(f"Hasil inspeksi berhasil disimpan di: {OUTPUT_PATH}")
