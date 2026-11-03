
from pathlib import Path
from paddleocr import PPStructureV3
from langchain_community.document_loaders import PyMuPDFLoader

PDF_PATH = Path("../knowledge/pdf_6623babcb5fea80d742728f6.pdf")
OUTPUT_DIR = Path("parser_test")

OUTPUT_DIR.mkdir(exist_ok=True)

def test_pymupdf():
    print("\n=== TEST PYMuPDF ===")

    loader = PyMuPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print("Jumlah halaman:", len(documents))

    for doc in documents[:3]:
        print("\nHalaman:", doc.metadata.get("page"))
        print(doc.page_content[:1000])

def test_ppstructure():
    print("\n=== TEST PPStructureV3 ===", flush=True)

    image_path = OUTPUT_DIR / "page_30.png"

    if not image_path.exists():
        raise FileNotFoundError(image_path.resolve())

    print("Membuat pipeline...", flush=True)
    pipeline = PPStructureV3()

    print("Pipeline berhasil dibuat.", flush=True)
    print("Memproses halaman PDF 30...", flush=True)

    try:
        output = pipeline.predict(input=str(image_path))

        print("Predict selesai.", flush=True)
        print("Tipe output:", type(output), flush=True)

        count = 0

        for index, result in enumerate(output):
            count += 1

            print(f"Menyimpan hasil ke-{index + 1}...", flush=True)

            result.save_to_json(
                save_path=str(
                    OUTPUT_DIR / f"page_30_{index + 1}.json"
                )
            )

            result.save_to_markdown(
                save_path=str(
                    OUTPUT_DIR / f"page_30_{index + 1}.md"
                )
            )

            print(f"Hasil ke-{index + 1} berhasil disimpan.", flush=True)

        print("Jumlah hasil:", count, flush=True)

        if count == 0:
            print("PERINGATAN: Tidak ada hasil yang dikembalikan.", flush=True)

    except Exception:
        import traceback
        traceback.print_exc()

    print("Pengujian PPStructureV3 selesai.", flush=True)
    
if __name__ == "__main__":
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF tidak ditemukan: {PDF_PATH}")

    test_pymupdf()
    test_ppstructure()
