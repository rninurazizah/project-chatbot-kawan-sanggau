
import fitz
from pathlib import Path

PDF_PATH = Path(
    r"..\knowledge\kecamatan-kapuas-dalam-angka-2025 (1).pdf"
)

OUTPUT_DIR = Path("pdf_preview")
OUTPUT_DIR.mkdir(exist_ok=True)

doc = fitz.open(PDF_PATH)

# Halaman PDF menggunakan indeks mulai dari 0
TARGET_PAGES = [100, 101, 102, 103]

for page_number in TARGET_PAGES:
    page = doc[page_number - 1]

    pix = page.get_pixmap(
        matrix=fitz.Matrix(2, 2),
        alpha=False
    )

    output_path = OUTPUT_DIR / f"page_{page_number}.png"
    pix.save(output_path)

    print(f"Berhasil: {output_path.resolve()}")

doc.close()
