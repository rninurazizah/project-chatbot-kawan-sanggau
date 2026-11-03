
from __future__ import annotations

import json
from pathlib import Path

import pymupdf


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = (
    BASE_DIR
    / "knowledge"
    / "kecamatan-kapuas-dalam-angka-2025 (1).pdf"
)

TABLE_DIR = (
    BASE_DIR
    / "extracted"
    / "kecamatan-kapuas-dalam-angka-2025 (1)"
    / "lattice_all"
)

OUTPUT_DIR = Path("pdf_inspection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_IDS = {51, 52, 53, 54, 89, 131, 148}


def load_tables():
    path = TABLE_DIR / "tables_clean.jsonl"

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    doc = pymupdf.open(PDF_PATH)
    tables = load_tables()

    summary = []

    for table in tables:
        table_id = table.get("table_id")

        if table_id not in TARGET_IDS:
            continue

        page_number = table.get("page")
        bbox_data = (
            table.get("bbox_pymupdf")
            or table.get("bbox")
        )

        if not page_number or not bbox_data:
            print(
                f"[SKIP] ID={table_id}: "
                "nomor halaman atau bbox tidak tersedia"
            )
            continue

        page = doc[page_number - 1]
        bbox = pymupdf.Rect(bbox_data)

        items = []

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()

                    if not text:
                        continue

                    rect = pymupdf.Rect(span["bbox"])

                    if not rect.intersects(bbox):
                        continue

                    items.append({
                        "text": text,
                        "x0": round(rect.x0, 2),
                        "y0": round(rect.y0, 2),
                        "x1": round(rect.x1, 2),
                        "y1": round(rect.y1, 2),
                        "font": span["font"],
                        "size": round(span["size"], 2),
                    })

        items.sort(
            key=lambda x: (x["y0"], x["x0"])
        )

        result = {
            "table_id": table_id,
            "page": page_number,
            "bbox": list(bbox),
            "text_count": len(items),
            "items": items,
        }

        output_path = (
            OUTPUT_DIR
            / f"table_{table_id}_coordinates.json"
        )

        output_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        summary.append({
            "table_id": table_id,
            "page": page_number,
            "text_count": len(items),
            "output": str(output_path),
        })

        print(
            f"[OK] ID={table_id} "
            f"Halaman={page_number} "
            f"Teks={len(items)}"
        )

    doc.close()

    summary_path = OUTPUT_DIR / "coordinate_summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nSelesai. Hasil disimpan di:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
