from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "extracted"


# ============================================================
# KONFIGURASI (satuan: karakter; bahasa Indonesia ~4 karakter/token)
# ============================================================

NARR_TARGET = 900      # ukuran chunk narasi yang dituju
NARR_MAX = 1400        # batas keras satu potongan teks
NARR_MIN = 200         # chunk lebih kecil dari ini digabung ke berikutnya
NARR_OVERLAP = 300     # overlap maksimum (satu potongan terakhir)
NARR_DROP_BELOW = 30   # chunk narasi lebih pendek dari ini dibuang

TABLE_MAX = 1800       # ukuran maksimum satu chunk tabel
TABLE_NOTES_MAX = 300  # catatan/sumber tabel dipotong sampai sini
HEADER_ROWS_MAX = 3

SKIP_PAGE_HINTS = {"toc"}  # halaman daftar isi tidak di-chunk


# ============================================================
# UTIL
# ============================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_id(key: str) -> str:
    """UUID deterministik: jalankan ulang tidak membuat duplikat di vector DB."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


# ============================================================
# NARASI
# ============================================================

SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
HEADING_RE = re.compile(r"^(BAB|SUBBAB|LAMPIRAN)\b|^\d+(\.\d+)*\.?\s+\S", re.I)


def is_heading(text: str) -> bool:
    if len(text) > 100 or text[-1] in ".;,:":
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    return upper_ratio > 0.8 or bool(HEADING_RE.match(text))


def split_long(text: str) -> list[str]:
    """Pecah paragraf panjang per kalimat; potong per kata bila perlu."""
    if len(text) <= NARR_MAX:
        return [text]

    pieces: list[str] = []
    current = ""
    for sentence in SENT_RE.split(text):
        while len(sentence) > NARR_MAX:
            cut = sentence.rfind(" ", 0, NARR_MAX)
            cut = cut if cut > 0 else NARR_MAX
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if current and len(current) + 1 + len(sentence) > NARR_TARGET:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def narrative_chunks(
    pages: list[dict[str, Any]], document: str, source: str
) -> list[dict[str, Any]]:
    title = document.replace("_", " ")
    chunks: list[dict[str, Any]] = []
    buf: list[dict[str, Any]] = []   # {"page", "text", "overlap"}
    heading = ""

    def size() -> int:
        return sum(len(p["text"]) for p in buf) + max(0, len(buf) - 1)

    def flush(keep_overlap: bool) -> None:
        nonlocal buf
        if any(not p["overlap"] for p in buf):
            body = "\n".join(p["text"] for p in buf)
            if len(body) >= NARR_DROP_BELOW:
                p0, p1 = buf[0]["page"], buf[-1]["page"]
                label = f"Halaman {p0}" if p0 == p1 else f"Halaman {p0}-{p1}"
                head = f"{title} | {label}" + (f" | {heading}" if heading else "")
                key = f"{document}:n{len(chunks)}"
                chunks.append({
                    "id": make_id(key),
                    "chunk_key": key,
                    "type": "narrative",
                    "document": document,
                    "source": source,
                    "page_start": p0,
                    "page_end": p1,
                    "heading": heading,
                    "text": f"{head}\n{body}",
                    "char_count": len(body),
                })
        last = buf[-1] if buf else None
        if keep_overlap and last and len(last["text"]) <= NARR_OVERLAP:
            buf = [{**last, "overlap": True}]
        else:
            buf = []

    for rec in pages:
        if SKIP_PAGE_HINTS & set(rec.get("page_hints") or []):
            continue
        for para in rec.get("paragraphs") or []:
            para = para.strip()
            if not para:
                continue
            if is_heading(para):
                flush(keep_overlap=False)   # bagian baru: tanpa overlap
                heading = para
                continue
            for piece in split_long(para):
                if buf and size() + 1 + len(piece) > NARR_TARGET and size() >= NARR_MIN:
                    flush(keep_overlap=True)
                buf.append({"page": rec["page"], "text": piece, "overlap": False})
    flush(keep_overlap=False)
    return chunks


# ============================================================
# TABEL
# ============================================================

NUM_RE = re.compile(r"^[\s(+-]*\d[\d.,/%\- ]*[\s)]*$")


def is_colnum_row(row: list[str]) -> bool:
    """Baris nomor kolom BPS: (1) (2) (3) ... atau 1 2 3 ..."""
    cells = [c.strip("() ") for c in row if c.strip()]
    if len(cells) < 3 or not all(c.isdigit() for c in cells):
        return False
    nums = [int(c) for c in cells]
    return nums in (
        list(range(1, len(nums) + 1)),
        list(range(0, len(nums))),
    ) or all(c.startswith("(") or c.endswith(")") for c in [x.strip() for x in row if x.strip()])


def numeric_ratio(values: list[str]) -> float:
    values = [v for v in values if v.strip()]
    if not values:
        return 0.0
    return sum(bool(NUM_RE.match(v.strip())) for v in values) / len(values)


def split_header(rows: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    header: list[list[str]] = []
    i = 0
    while i < len(rows) and len(header) < HEADER_ROWS_MAX:
        row = rows[i]
        if is_colnum_row(row):
            i += 1
            continue
        if numeric_ratio(row[1:]) >= 0.5:
            break
        header.append(row)
        i += 1
    data = rows[i:]
    if not data and header:            # semuanya terlihat seperti header
        data, header = rows[1:], rows[:1]
    return header, data


def column_names(header: list[list[str]], ncols: int) -> list[str]:
    names = [[] for _ in range(ncols)]
    for row in header:
        filled = list(row)
        last = ""
        for j in range(1, ncols):          # isi sel header gabungan ke kanan
            if filled[j]:
                last = filled[j]
            elif last:
                filled[j] = last
        for j in range(ncols):
            part = filled[j]
            if part and (not names[j] or names[j][-1] != part):
                names[j].append(part)
    result = []
    for j, parts in enumerate(names):
        if parts:
            result.append(" > ".join(parts))
        else:
            result.append("Uraian" if j == 0 else f"Kolom {j + 1}")
    return result


def serialize_row(row: list[str], names: list[str]) -> str:
    label = row[0] or "(tanpa label)"
    pairs = [f"{names[j]} = {v}" for j, v in enumerate(row) if j > 0 and v]
    return f"{label}: " + "; ".join(pairs) if pairs else f"[{label}]"


def md_escape(cell: str) -> str:
    return cell.replace("|", "\\|")


def to_markdown(names: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(md_escape(n) for n in names) + " |",
        "|" + "---|" * len(names),
    ]
    lines += ["| " + " | ".join(md_escape(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def table_chunks(
    tables: list[dict[str, Any]], document: str, source: str
) -> list[dict[str, Any]]:
    title = document.replace("_", " ")
    chunks: list[dict[str, Any]] = []

    for table in tables:
        rows = table.get("clean_rows") or []
        if not rows:
            continue
        ncols = max(len(r) for r in rows)
        rows = [r + [""] * (ncols - len(r)) for r in rows]

        header, data = split_header(rows)
        names = column_names(header, ncols)
        if not data:
            continue

        caption = (table.get("caption") or "").strip()
        notes = table.get("notes") or ""
        if isinstance(notes, list):
            notes = " ".join(str(n) for n in notes)
        notes = notes.strip()[:TABLE_NOTES_MAX]

        page = table["page"]
        head = f"{title} | Halaman {page} | Tabel: {caption or 'tanpa judul'}"
        cols_line = "Kolom: " + " | ".join(names)
        overhead = len(head) + len(cols_line) + len(notes) + 40

        # kelompokkan baris; header kolom diulang di setiap chunk
        groups: list[list[list[str]]] = []
        current: list[list[str]] = []
        used = overhead
        for row in data:
            line_len = len(serialize_row(row, names)) + 1
            if current and used + line_len > TABLE_MAX:
                groups.append(current)
                current, used = [], overhead
            current.append(row)
            used += line_len
        if current:
            groups.append(current)

        for part, group in enumerate(groups, start=1):
            part_label = f" [bagian {part}/{len(groups)}]" if len(groups) > 1 else ""
            lines = [head + part_label, cols_line]
            lines += [serialize_row(r, names) for r in group]
            if notes:
                lines.append(f"Catatan: {notes}")

            key = f"{document}:t{table['table_id']}:{part}"
            chunks.append({
                "id": make_id(key),
                "chunk_key": key,
                "type": "table",
                "document": document,
                "source": source,
                "page_start": page,
                "page_end": page,
                "table_id": table["table_id"],
                "caption": caption,
                "columns": names,
                "row_count": len(group),
                "part": part,
                "parts_total": len(groups),
                "needs_review": table.get("status") == "needs_review",
                "text": "\n".join(lines),
                "markdown": to_markdown(names, group),
                "char_count": sum(len(l) for l in lines),
            })

    return chunks


# ============================================================
# MAIN
# ============================================================

def chunk_document(doc_dir: Path) -> list[dict[str, Any]]:
    narratives = read_jsonl(doc_dir / "narrative.jsonl")
    tables = read_jsonl(doc_dir / "tables.jsonl")
    if not narratives and not tables:
        return []

    sample = (narratives or tables)[0]
    document, source = sample["document"], sample["source"]

    chunks = narrative_chunks(narratives, document, source)
    chunks += table_chunks(tables, document, source)
    write_jsonl(doc_dir / "chunks.jsonl", chunks)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunking narasi dan tabel.")
    parser.add_argument("--input", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--label",
        default="lattice",
        help="Subfolder hasil pipeline: lattice, stream, atau pymupdf.",
    )
    args = parser.parse_args()

    all_chunks: list[dict[str, Any]] = []
    for doc_dir in sorted(p for p in args.input.iterdir() if p.is_dir()):
        target = doc_dir / args.label
        if not target.exists():
            continue
        chunks = chunk_document(target)
        n_narr = sum(c["type"] == "narrative" for c in chunks)
        n_tab = len(chunks) - n_narr
        print(f"{doc_dir.name}: {n_narr} chunk narasi, {n_tab} chunk tabel")
        all_chunks += chunks

    out = args.input / f"all_chunks_{args.label}.jsonl"
    write_jsonl(out, all_chunks)
    print(f"Total {len(all_chunks)} chunk -> {out}")


if __name__ == "__main__":
    main()