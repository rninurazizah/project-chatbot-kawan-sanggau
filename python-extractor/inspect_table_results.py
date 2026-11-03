
import pymupdf
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "knowledge"
OUTPUT_DIR = BASE_DIR / "extracted"

SAMPLE_PAGES = [32, 100, 143]


def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("PDF tidak ditemukan.")

    pdf_path = pdf_files[0]
    document = pymupdf.open(pdf_path)

    for page_number in SAMPLE_PAGES:
        page = document[page_number - 1]

        result = page.find_tables()

        extracted_tables = []

        for index, table in enumerate(result.tables, start=1):
            extracted_tables.append({
                "table_index": index,
                "bbox": list(table.bbox),
                "row_count": table.row_count,
                "column_count": table.col_count,
                "data": table.extract()
            })

        output = {
            "page": page_number,
            "table_count": len(extracted_tables),
            "tables": extracted_tables,
            "raw_text": page.get_text("text")
        }

        output_path = (
            OUTPUT_DIR / f"inspect_page_{page_number}.json"
        )

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                output,
                file,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"Halaman {page_number}: "
            f"{len(extracted_tables)} tabel ditemukan"
        )
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
