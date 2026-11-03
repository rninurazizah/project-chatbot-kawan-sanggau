
import pymupdf
from pathlib import Path

PDF_PATH = Path(
    "knowledge/kecamatan-kapuas-dalam-angka-2025 (1).pdf"
)

pdf = pymupdf.open(PDF_PATH)
page = pdf[75]  # PDF halaman 76

# Ambil seluruh kata beserta koordinatnya
words = page.get_text("words")

# Urutkan berdasarkan posisi vertikal, lalu horizontal
words = sorted(words, key=lambda w: (w[1], w[0]))

print("PDF halaman: 76")
print("Jumlah kata:", len(words))
print("\nKoordinat teks:")
print("=" * 80)

for word in words:
    x0, y0, x1, y1, text = word[:5]

    print(
        f"X: {x0:7.2f} | "
        f"Y: {y0:7.2f} | "
        f"Teks: {text}"
    )

pdf.close()
