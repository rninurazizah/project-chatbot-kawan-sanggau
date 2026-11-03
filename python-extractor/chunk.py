
from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "extracted_paddle" / "normalized_data.jsonl"
OUTPUT_FILE = BASE_DIR / "extracted_paddle" / "chunks.jsonl"

# Ukuran chunk dalam karakter
NARR_TARGET = 900
NARR_MAX = 1400
NARR_OVERLAP = 200
NARR_DROP_BELOW = 30

GLOSSARY_MAX = 1000

TABLE_MAX = 1800
TABLE_NOTES_MAX = 300


# ============================================================
# 2. UTILITAS
# ============================================================

def make_id(key: str) -> str:
    """ID deterministik agar proses ulang tidak membuat ID berbeda."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def read_jsonl(path: Path) -> list[dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON tidak valid pada baris {line_number}: {error}"
                )

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]]
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:

        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def clean_text(value: Any) -> str:

    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


DOC_META_FIELDS = (
    "document_id",
    "source",
    "kecamatan",
    "tahun",
    "kabupaten",
    "document_type",
)


def doc_meta(record: dict[str, Any]) -> dict[str, Any]:
    """Metadata dokumen yang harus ikut ke setiap chunk."""

    return {
        field: record.get(field)
        for field in DOC_META_FIELDS
        if record.get(field) not in (None, "")
    }


def doc_title(record: dict[str, Any]) -> str:
    return clean_text(record.get("document") or record.get("source"))


def doc_key(record: dict[str, Any]) -> str:
    return clean_text(record.get("document_id") or record.get("source"))


# ============================================================
# 3. PEMECAHAN TEKS NARASI
# ============================================================

SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"(])"
)


def split_long_text(text: str) -> list[str]:
    """Memecah teks panjang berdasarkan kalimat."""

    if len(text) <= NARR_MAX:
        return [text]

    pieces = []
    current = ""

    sentences = SENTENCE_RE.split(text)

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Jika satu kalimat terlalu panjang, pecah berdasarkan kata
        while len(sentence) > NARR_MAX:

            cut = sentence.rfind(" ", 0, NARR_MAX)

            if cut <= 0:
                cut = NARR_MAX

            if current:
                pieces.append(current)
                current = ""

            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()

        if not sentence:
            continue

        if current and len(current) + len(sentence) + 1 > NARR_TARGET:
            pieces.append(current)
            current = sentence

        else:
            current = f"{current} {sentence}".strip()

    if current:
        pieces.append(current)

    return pieces


def narrative_chunks(
    records: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    chunks = []

    for record_index, record in enumerate(records):

        content = clean_text(record.get("content"))

        if not content:
            continue

        title = clean_text(record.get("title"))
        page = record.get("page")
        page_end = record.get("page_end") or page
        document = doc_title(record)
        identity = doc_key(record)

        pieces = split_long_text(content)

        current = []
        part_index = 0

        def flush(keep_overlap: bool = False):

            nonlocal current, part_index

            if not current:
                return

            body = "\n".join(current)

            if len(body) >= NARR_DROP_BELOW:

                part_index += 1

                # Hash konten membuat ID stabil walau dokumen lain
                # ditambahkan/diurutkan ulang.
                digest = hashlib.sha1(
                    content.encode("utf-8")
                ).hexdigest()[:12]

                key = (
                    f"{identity}:narrative:"
                    f"{page}:{digest}:{part_index}"
                )

                parts = []

                if document:
                    parts.append(f"Dokumen: {document}")

                if title and title != document:
                    parts.append(title)

                parts.append(body)

                chunks.append({
                    "id": make_id(key),
                    "chunk_key": key,
                    "type": "narrative",
                    "document": document,
                    "page_start": page,
                    "page_end": page_end,
                    "title": title,
                    "text": "\n".join(parts),
                    "char_count": len(body),
                    **doc_meta(record),
                })

            previous = current[-1] if current else ""

            if keep_overlap and len(previous) <= NARR_OVERLAP:
                current = [previous]
            else:
                current = []

        for piece in pieces:

            current_size = sum(len(x) for x in current)

            if (
                current
                and current_size + len(piece) + 1 > NARR_TARGET
            ):
                flush(keep_overlap=True)

            current.append(piece)

            # Batas keras untuk menghindari chunk terlalu panjang
            if sum(len(x) for x in current) > NARR_MAX:
                flush(keep_overlap=False)

        flush()

    return chunks


# ============================================================
# 4. CHUNKING GLOSSARY
# ============================================================

def glossary_chunks(
    records: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    chunks = []

    for index, record in enumerate(records):

        content = clean_text(record.get("content"))

        if not content:
            continue

        title = clean_text(record.get("title"))
        page = record.get("page")
        page_end = record.get("page_end") or page
        document = doc_title(record)
        identity = doc_key(record)

        # Pecah berdasarkan baris agar definisi tetap berdekatan
        lines = [
            line.strip()
            for line in record.get("content", "").splitlines()
            if line.strip()
        ]

        if not lines:
            lines = [content]

        groups = []
        current = []

        for line in lines:

            if (
                current
                and sum(len(x) for x in current) + len(line) + 1
                > GLOSSARY_MAX
            ):
                groups.append(current)
                current = []

            current.append(line)

        if current:
            groups.append(current)

        for part, group in enumerate(groups, start=1):

            body = "\n".join(group)

            digest = hashlib.sha1(
                f"{title}\n{content[:500]}".encode("utf-8")
            ).hexdigest()[:12]

            key = (
                f"{identity}:glossary:"
                f"{page}:{digest}:{part}"
            )

            prefix = f"Dokumen: {document}\n{title}" if document else title

            chunks.append({
                "id": make_id(key),
                "chunk_key": key,
                "type": "glossary",
                "document": document,
                "page_start": page,
                "page_end": page_end,
                "title": title,
                "part": part,
                "parts_total": len(groups),
                "text": f"{prefix}\n{body}",
                "char_count": len(body),
                **doc_meta(record),
            })

    return chunks


# ============================================================
# 5. SERIALISASI BARIS TABEL
# ============================================================

def serialize_row(
    row: dict[str, Any],
    columns: list[str]
) -> str:

    values = []

    for column in columns:

        value = clean_text(row.get(column))

        # Konvensi BPS: "-" berarti nol/tidak ada. Jangan hilangkan sel
        # kosong karena model cenderung mengarang nilai pengganti.
        if value == "-":
            value = "0"

        values.append(
            f"{column}: {value or 'kosong'}"
        )

    return "; ".join(values)


def markdown_escape(value: Any) -> str:

    return clean_text(value).replace("|", "\\|")


def table_to_markdown(
    columns: list[str],
    rows: list[dict[str, Any]]
) -> str:

    lines = [
        "| " + " | ".join(
            markdown_escape(col) for col in columns
        ) + " |",
        "|" + "---|" * len(columns)
    ]

    for row in rows:

        values = [
            markdown_escape(row.get(col, ""))
            for col in columns
        ]

        lines.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(lines)


# ============================================================
# 6. CHUNKING TABEL
# ============================================================

def table_chunks(
    records: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    chunks = []

    for table in records:

        columns = table.get("columns") or []
        rows = table.get("rows") or []

        if not columns or not rows:
            continue

        table_id = table.get("table_id")
        table_number = table.get("table_number")
        page = table.get("page")
        page_end = table.get("page_end", page)
        caption = clean_text(table.get("title"))
        document = doc_title(table)
        identity = doc_key(table)

        if not caption:
            caption = f"Tabel {table_number or table_id}"

        # Setiap baris tetap terhubung dengan seluruh nama kolom
        serialized_rows = []

        for row in rows:

            serialized = serialize_row(row, columns)

            if serialized:
                serialized_rows.append((row, serialized))

        if not serialized_rows:
            continue

        header_lines = [f"Dokumen: {document}"]

        if table.get("kecamatan"):
            header_lines.append(f"Kecamatan: {table.get('kecamatan')}")

        if table.get("tahun"):
            header_lines.append(f"Tahun: {table.get('tahun')}")

        if re.match(r"^(tabel|table)\s", caption, re.IGNORECASE):
            table_label = caption
        else:
            table_label = f"Tabel {table_number or table_id}: {caption}"

        header_lines.extend([
            f"Halaman: {page}",
            table_label,
            f"Kolom: {'; '.join(columns)}",
            "Catatan: 0 berarti nol/tidak ada; ... berarti data belum tersedia; kosong berarti sel tidak terisi.",
        ])

        header = "\n".join(header_lines)

        groups = []
        current = []
        current_size = len(header)

        for row, serialized in serialized_rows:

            line_size = len(serialized) + 1

            if (
                current
                and current_size + line_size > TABLE_MAX
            ):
                groups.append(current)
                current = []
                current_size = len(header)

            current.append((row, serialized))
            current_size += line_size

        if current:
            groups.append(current)

        for part, group in enumerate(groups, start=1):

            part_label = (
                f"Bagian {part} dari {len(groups)}"
            )

            lines = [
                header,
                part_label
            ]

            lines.extend(
                serialized
                for _, serialized in group
            )

            text = "\n".join(lines)

            key = (
                f"{identity}:table:"
                f"{table_id}:{part}"
            )

            group_rows = [
                row for row, _ in group
            ]

            chunks.append({
                "id": make_id(key),
                "chunk_key": key,
                "type": "table",
                "document": document,
                "page_start": page,
                "page_end": page_end,
                "table_id": table_id,
                "table_number": table_number,
                "title": caption,
                "caption": caption,
                "columns": columns,
                "row_count": len(group_rows),
                "part": part,
                "parts_total": len(groups),
                "text": text,
                "markdown": table_to_markdown(
                    columns,
                    group_rows
                ),
                "char_count": len(text),
                **doc_meta(table),
            })

    return chunks


# ============================================================
# 7. VALIDASI CHUNK
# ============================================================

def validate_chunks(chunks):

    errors = []

    ids = [
        chunk.get("id")
        for chunk in chunks
    ]

    if len(ids) != len(set(ids)):
        errors.append("Ditemukan ID chunk duplikat.")

    for index, chunk in enumerate(chunks, start=1):

        if not chunk.get("text"):
            errors.append(
                f"Chunk {index}: teks kosong."
            )

        if not chunk.get("id"):
            errors.append(
                f"Chunk {index}: ID kosong."
            )

        if chunk.get("type") == "table":

            if not chunk.get("columns"):
                errors.append(
                    f"Chunk {index}: kolom tabel kosong."
                )

            if not chunk.get("table_id"):
                errors.append(
                    f"Chunk {index}: table_id kosong."
                )

    return errors


# ============================================================
# 8. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Chunking data hasil normalisasi RAG."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CHUNKING DATA UNTUK RAG")
    print("=" * 60)

    records = read_jsonl(args.input)

    narratives = [
        r for r in records
        if r.get("type") == "narrative"
    ]

    glossaries = [
        r for r in records
        if r.get("type") == "glossary"
    ]

    tables = [
        r for r in records
        if r.get("type") == "table"
    ]

    print(f"\nTotal record : {len(records)}")
    print(f"Narasi       : {len(narratives)}")
    print(f"Glossary     : {len(glossaries)}")
    print(f"Tabel        : {len(tables)}")

    print("\nMemproses narasi...")
    narrative_result = narrative_chunks(narratives)

    print("Memproses glossary...")
    glossary_result = glossary_chunks(glossaries)

    print("Memproses tabel...")
    table_result = table_chunks(tables)

    chunks = (
        narrative_result
        + glossary_result
        + table_result
    )

    chunks.sort(
        key=lambda item: (
            str(item.get("document_id") or ""),
            item.get("page_start") or 0,
            item.get("type", ""),
            item.get("chunk_key", "")
        )
    )

    print("\nValidasi chunk...")

    errors = validate_chunks(chunks)

    if errors:

        print("\nVALIDASI GAGAL:")

        for error in errors:
            print(f"- {error}")

        raise ValueError(
            "Chunking dihentikan karena validasi gagal."
        )

    write_jsonl(args.output, chunks)

    narrative_count = sum(
        c["type"] == "narrative" for c in chunks
    )

    glossary_count = sum(
        c["type"] == "glossary" for c in chunks
    )

    table_count = sum(
        c["type"] == "table" for c in chunks
    )

    print("\n" + "=" * 60)
    print("CHUNKING BERHASIL")
    print("=" * 60)

    print(f"Chunk narasi   : {narrative_count}")
    print(f"Chunk glossary : {glossary_count}")
    print(f"Chunk tabel    : {table_count}")
    print(f"Total chunk    : {len(chunks)}")
    print("Validasi       : LULUS")
    print(f"Output         : {args.output}")


if __name__ == "__main__":
    main()
