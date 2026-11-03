import json
from pathlib import Path
from collections import Counter

INPUT_FILE = Path(
    "../extracted/kecamatan-kapuas-dalam-angka-2025 (1)/stream/merged_chunks.jsonl"
)

chunks = []

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

print("=== DISTRIBUSI KONTEN ===")
print(Counter(c.get("content_type") for c in chunks))

print("\n=== CONTOH NARRATIVE ===")
for chunk in [c for c in chunks if c.get("content_type") == "narrative"][:3]:
    print("\nID:", chunk.get("chunk_id"))
    print("Halaman:", chunk.get("pages"))
    print("Teks:", chunk.get("text", "")[:500])

print("\n=== CONTOH TABLE ===")
for chunk in [c for c in chunks if c.get("content_type") == "table"][:3]:
    print("\nID:", chunk.get("chunk_id"))
    print("Judul:", chunk.get("title"))
    print("Halaman:", chunk.get("pages"))
    print("Teks:", chunk.get("text", "")[:500])

print("\n=== PEMERIKSAAN INDEKS ===")
keywords = ["DAFTAR ISI", "DAFTAR GAMBAR", "DAFTAR TABEL"]

for keyword in keywords:
    matches = [
        c for c in chunks
        if keyword in c.get("text", "").upper()
    ]
    print(f"{keyword}: {len(matches)} chunk")