import fitz
from pathlib import Path

pdf_path = Path(
    "../knowledge/kecamatan-kapuas-dalam-angka-2025 (1).pdf"
)

output_dir = Path("debug_table_pages")
output_dir.mkdir(exist_ok=True)

doc = fitz.open(pdf_path)

for page_number in [31, 32]:
    page = doc[page_number]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    output_path = output_dir / f"page_{page_number + 1}.png"
    pix.save(output_path)
    print(f"Saved: {output_path}")