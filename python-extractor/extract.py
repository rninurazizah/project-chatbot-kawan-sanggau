
import pymupdf
import json
from pathlib import Path


# ==========================================
# KONFIGURASI
# ==========================================

PDF_PATH = Path(
    "knowledge/kecamatan-kapuas-dalam-angka-2025 (1).pdf"
)

OUTPUT_PATH = Path(
    "extracted/raw_pages.jsonl"
)


# ==========================================
# FUNGSI EKSTRAKSI
# ==========================================

def extract_pdf(pdf_path, output_path):

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"File PDF tidak ditemukan: {pdf_path}"
        )

    # Membuat folder output jika belum tersedia
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf = pymupdf.open(pdf_path)

    total_pages = len(pdf)

    print(f"Mulai ekstraksi PDF: {pdf_path.name}")
    print(f"Total halaman: {total_pages}")

    total_words = 0

    with open(output_path, "w", encoding="utf-8") as file:

        for page_index, page in enumerate(pdf):

            # Nomor halaman PDF dimulai dari 1
            page_number = page_index + 1

            # Mengambil seluruh teks halaman
            text = page.get_text("text")

            # Mengambil kata beserta koordinatnya
            raw_words = page.get_text("words")

            words = []

            for word in raw_words:

                x0, y0, x1, y1, word_text, block_no, line_no, word_no = word

                words.append({
                    "text": word_text,
                    "x0": round(x0, 2),
                    "y0": round(y0, 2),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "block": block_no,
                    "line": line_no,
                    "word_no": word_no
                })

            # Informasi satu halaman
            page_data = {
                "page": page_number,
                "width": round(page.rect.width, 2),
                "height": round(page.rect.height, 2),
                "text": text,
                "word_count": len(words),
                "words": words
            }

            # Menyimpan satu halaman dalam satu baris JSONL
            file.write(
                json.dumps(
                    page_data,
                    ensure_ascii=False
                ) + "\n"
            )

            total_words += len(words)

            print(
                f"Halaman {page_number}/{total_pages} "
                f"| Kata: {len(words)}"
            )

    pdf.close()

    print("\n================================")
    print("EKSTRAKSI SELESAI")
    print("================================")
    print(f"Total halaman : {total_pages}")
    print(f"Total kata    : {total_words}")
    print(f"Output        : {output_path}")


# ==========================================
# PROGRAM UTAMA
# ==========================================

if __name__ == "__main__":

    extract_pdf(
        PDF_PATH,
        OUTPUT_PATH
    )
