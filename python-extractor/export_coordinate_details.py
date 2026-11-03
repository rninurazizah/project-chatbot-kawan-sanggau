
import json
from pathlib import Path

folder = Path("pdf_inspection")
target_ids = [52, 53]

output = []

for table_id in target_ids:
    path = folder / f"table_{table_id}_coordinates.json"

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    output.append("=" * 70)
    output.append(f"TABLE ID: {table_id} | PAGE: {data['page']}")
    output.append("=" * 70)

    for item in data["items"]:
        text = item["text"]
        x = item["x0"]
        y = item["y0"]

        output.append(
            f"Y={y:7.2f} X={x:7.2f} | {text}"
        )

output_path = Path("coordinate_details.txt")

output_path.write_text(
    "\n".join(output),
    encoding="utf-8"
)

print("Berhasil menyimpan:", output_path.resolve())
