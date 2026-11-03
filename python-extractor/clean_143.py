
import json
from pathlib import Path

TARGET_IDS = {51, 52, 53, 54, 89, 131, 148}

BASE_DIR = Path(
    "../extracted/kecamatan-kapuas-dalam-angka-2025 (1)/lattice_all"
)

INPUT_FILE = BASE_DIR / "tables_reconstructed.jsonl"

with INPUT_FILE.open("r", encoding="utf-8") as f:
    tables = [json.loads(line) for line in f if line.strip()]

for table in tables:
    table_id = table.get("table_id")

    if table_id not in TARGET_IDS:
        continue

    print("\n" + "=" * 80)
    print(f"TABLE ID : {table_id}")
    print(f"PAGE     : {table.get('page')}")
    print(f"METHOD   : {table.get('reconstruction_method')}")
    print("=" * 80)

    rows = table.get("reconstructed_rows", [])

    for row in rows:
        print(" | ".join(str(cell) for cell in row))

print("\nPemeriksaan selesai.")
