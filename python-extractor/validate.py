
from pathlib import Path
import pymupdf

PDF_PATH = Path("../knowledge/pdf_6623babcb5fea80d742728f6.pdf")
OUTPUT_DIR = Path("parser_test")

OUTPUT_DIR.mkdir(exist_ok=True)

def render_page():
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH.resolve())

    doc = pymupdf.open(PDF_PATH)

    # Halaman PDF 30 berarti index 29
    page_number = 30
    page = doc[page_number - 1]

    # Render dengan resolusi 200 DPI
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(200 / 72, 200 / 72),
        alpha=False
    )

    output_path = OUTPUT_DIR / f"page_{page_number}.png"
    pix.save(output_path)

    print("Render berhasil!")
    print("Halaman PDF:", page_number)
    print("Ukuran gambar:", pix.width, "x", pix.height)
    print("Lokasi:", output_path.resolve())

    doc.close()

if __name__ == "__main__":
    render_page()
