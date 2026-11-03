
import pymupdf
from pathlib import Path

PDF_PATH = Path(
    r"..\knowledge\kecamatan-kapuas-dalam-angka-2025 (1).pdf"
)

OUTPUT_DIR = Path("pdf_inspection")
OUTPUT_DIR.mkdir(exist_ok=True)

doc = pymupdf.open(PDF_PATH)

# Halaman fisik PDF yang akan diperiksa (dimulai dari 1)
TARGET_PAGES = [100, 101, 102, 103]

for page_number in TARGET_PAGES:
    if page_number > len(doc):
        print(f"Halaman {page_number} tidak tersedia.")
        continue

    page = doc[page_number - 1]

    # Simpan gambar halaman untuk pemeriksaan visual
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False
    )

    image_path = OUTPUT_DIR / f"page_{page_number}.png"
    pix.save(image_path)

    # Ambil teks beserta koordinatnya
    blocks = page.get_text("blocks")

    text_path = OUTPUT_DIR / f"page_{page_number}.txt"

    with open(text_path, "w", encoding="utf-8") as file:
        file.write(f"HALAMAN PDF: {page_number}\n")
        file.write("=" * 60 + "\n\n")

        for block in blocks:
            x0, y0, x1, y1, text = block[:5]

            file.write(
                f"KOORDINAT: ({x0:.1f}, {y0:.1f}, "
                f"{x1:.1f}, {y1:.1f})\n"
            )
            file.write(text.strip() + "\n")
            file.write("-" * 40 + "\n")

    print(f"Halaman {page_number} selesai diperiksa.")

print("\nPemeriksaan selesai.")
print(f"Jumlah halaman PDF: {len(doc)}")
print(f"Hasil tersimpan di: {OUTPUT_DIR.resolve()}")
