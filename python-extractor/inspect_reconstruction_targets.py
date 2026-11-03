
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TABLE_DIR = (
    BASE_DIR
    / "extracted"
    / "kecamatan-kapuas-dalam-angka-2025 (1)"
    / "lattice_all"
)

TARGET_IDS = {51, 52, 53, 54, 89, 131, 148}

input_path = TABLE_DIR / "tables_clean.jsonl"

with input_path.open("r", encoding="utf-8") as f:
    tables = [json.loads(line) for line in f if line.strip()]

for table in tables:
    table_id = table.get("table_id")

    if table_id not in TARGET_IDS:
        continue

    rows = (
        table.get("clean_rows")
        or table.get("rows")
        or table.get("table")
        or []
    )

    print("\n" + "=" * 80)
    print(f"TABLE ID : {table_id}")
    print(f"PAGE     : {table.get('page')}")
    print(f"ENGINE   : {table.get('engine')}")
    print(f"HEADER   : {table.get('header_row_count')}")
    print(f"ROWS     : {len(rows)}")
    print(f"COLS     : {max(map(len, rows), default=0)}")
    print("=" * 80)

    for i, row in enumerate(rows):
        print(f"{i:02d}: {row}")
