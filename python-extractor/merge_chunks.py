
import json
import argparse
from pathlib import Path
from collections import Counter


def read_jsonl(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON tidak valid: {file_path}, "
                    f"baris {line_number}: {error}"
                )

    return records


def merge_chunks(directory):
    directory = Path(directory)

    narrative_path = directory / "narrative_chunks.jsonl"
    table_path = directory / "table_chunks.jsonl"

    output_path = directory / "merged_chunks.jsonl"

    # Validasi file input
    for path in [narrative_path, table_path]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    # Baca data
    narrative_chunks = read_jsonl(narrative_path)
    table_chunks = read_jsonl(table_path)

    merged = narrative_chunks + table_chunks

    # Validasi dan normalisasi
    seen_ids = set()
    valid_chunks = []
    empty_text = 0

    for index, chunk in enumerate(merged, start=1):
        chunk_id = chunk.get("chunk_id")
        text = str(chunk.get("text", "")).strip()

        if not chunk_id:
            raise ValueError(
                f"Chunk ke-{index} tidak memiliki chunk_id"
            )

        if chunk_id in seen_ids:
            raise ValueError(
                f"Duplicate chunk_id ditemukan: {chunk_id}"
            )

        seen_ids.add(chunk_id)

        if not text:
            empty_text += 1
            continue

        chunk["text"] = text
        valid_chunks.append(chunk)

    # Pengurutan berdasarkan halaman
    def get_page(chunk):
        if chunk.get("page") is not None:
            return chunk["page"]

        pages = chunk.get("pages", [])
        if pages:
            return min(pages)

        return float("inf")

    valid_chunks.sort(key=get_page)

    # Simpan hasil
    with open(output_path, "w", encoding="utf-8") as file:
        for chunk in valid_chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    # Statistik
    type_counts = Counter(
        chunk.get("content_type", "unknown")
        for chunk in valid_chunks
    )

    lengths = [
        len(chunk["text"])
        for chunk in valid_chunks
    ]

    print("\n=== HASIL MERGE CHUNKS ===")
    print(f"Narrative chunks dibaca : {len(narrative_chunks)}")
    print(f"Table chunks dibaca     : {len(table_chunks)}")
    print(f"Total sebelum validasi  : {len(merged)}")
    print(f"Chunk valid             : {len(valid_chunks)}")
    print(f"Chunk teks kosong       : {empty_text}")
    print(f"Total karakter          : {sum(lengths)}")

    print("\n=== BERDASARKAN JENIS ===")
    for content_type, count in type_counts.items():
        print(f"{content_type:20}: {count}")

    print(f"\nOutput: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Menggabungkan narrative dan table chunks"
    )

    parser.add_argument(
        "--dir",
        required=True,
        help="Folder yang berisi file chunks"
    )

    args = parser.parse_args()

    merge_chunks(args.dir)


if __name__ == "__main__":
    main()
