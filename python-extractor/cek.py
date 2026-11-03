
import json
import sys
from pathlib import Path
from collections import Counter


def load_jsonl(file_path):
    """Membaca file JSONL dan memvalidasi setiap record."""
    records = []
    invalid_records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                invalid_records.append({
                    "line": line_number,
                    "error": str(error)
                })

    return records, invalid_records


def inspect_records(records):
    """Memeriksa struktur dan kualitas record."""

    total = len(records)
    empty_text = 0
    missing_text = 0
    missing_page = 0

    content_types = Counter()
    documents = Counter()
    pages = []

    for record in records:
        # Periksa teks
        text = record.get("text")

        if text is None:
            missing_text += 1
        elif not str(text).strip():
            empty_text += 1

        # Periksa tipe konten
        content_types[record.get("content_type", "Tidak diketahui")] += 1

        # Periksa dokumen
        documents[record.get("document", "Tidak diketahui")] += 1

        # Periksa halaman
        page = record.get("page") or record.get("pages")

        if page is None:
            missing_page += 1
        elif isinstance(page, list):
            pages.extend(page)
        else:
            pages.append(page)

    print("\n" + "=" * 65)
    print("LAPORAN PEMERIKSAAN FILE JSONL")
    print("=" * 65)

    print(f"Total record             : {total}")
    print(f"Teks kosong              : {empty_text}")
    print(f"Field text tidak tersedia: {missing_text}")
    print(f"Metadata halaman kosong  : {missing_page}")

    print("\nDISTRIBUSI CONTENT TYPE")
    print("-" * 65)

    for content_type, count in content_types.items():
        print(f"{content_type:<25}: {count}")

    print("\nDISTRIBUSI DOKUMEN")
    print("-" * 65)

    for document, count in documents.items():
        print(f"{document:<45}: {count}")

    print("\nINFORMASI HALAMAN")
    print("-" * 65)

    if pages:
        print(f"Halaman terendah : {min(pages)}")
        print(f"Halaman tertinggi: {max(pages)}")
        print(f"Jumlah referensi halaman: {len(pages)}")
    else:
        print("Tidak ditemukan metadata halaman.")

    print("\n" + "=" * 65)


def show_samples(records, limit=3):
    """Menampilkan beberapa contoh record."""

    print("\nCONTOH RECORD")
    print("=" * 65)

    for index, record in enumerate(records[:limit], start=1):
        print(f"\nRECORD KE-{index}")
        print("-" * 65)

        print("Chunk ID     :", record.get("chunk_id"))
        print("Document     :", record.get("document"))
        print("Content type :", record.get("content_type"))
        print("Title        :", record.get("title"))

        page = record.get("page") or record.get("pages")
        print("Page         :", page)

        text = record.get("text", "")
        print("\nTEXT:")
        print(str(text)[:1500])

    print("\n" + "=" * 65)


def main():

    if len(sys.argv) < 2:
        print("Cara penggunaan:")
        print('python cek.py "lokasi_file.jsonl"')
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path

    if not file_path.exists():
        print(f"ERROR: File tidak ditemukan: {file_path}")
        sys.exit(1)

    if file_path.suffix.lower() != ".jsonl":
        print("PERINGATAN: File yang diberikan bukan berekstensi .jsonl")

    print(f"\nFile yang diperiksa: {file_path}")

    try:
        records, invalid_records = load_jsonl(file_path)

        inspect_records(records)
        show_samples(records)

        print("\nVALIDASI JSONL")
        print("=" * 65)
        print(f"Record berhasil dibaca: {len(records)}")
        print(f"Record JSON tidak valid: {len(invalid_records)}")

        if invalid_records:
            print("\nDETAIL ERROR:")
            for item in invalid_records[:10]:
                print(f"Baris {item['line']}: {item['error']}")

    except UnicodeDecodeError:
        print("ERROR: File tidak dapat dibaca menggunakan UTF-8.")

    except OSError as error:
        print(f"ERROR saat membaca file: {error}")


if __name__ == "__main__":
    main()
