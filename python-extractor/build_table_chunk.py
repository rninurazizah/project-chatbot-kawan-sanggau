
"""
Langkah 2 + 3:
  2. Gabungkan tabel lanjutan ("Lanjutan Tabel X") menjadi satu tabel logis.
  3. Ubah tiap tabel logis menjadi chunk teks (header/judul diulang).

Pakai:
  python build_table_chunk.py --dir "../extracted/<nama-pdf>/stream"

Masukan:
  tables.jsonl
  rejected_tables.jsonl

Keluaran:
  logical_tables.jsonl
  table_chunks.jsonl
  rescued_pages.json
  filtered_noise_tables.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from content_separation import (
    CAPTION_RE,
    COLNUM_CELL,
    is_noise_cell,
    parse_caption,
    strip_invisible,
)

MAX_CHARS = 1600

NOTE_ROW_RE = re.compile(
    r"^(sumber|c\s?atatan|keterangan)\b",
    re.IGNORECASE,
)
YEAR_END_RE = re.compile(r"20\d{2}(/\d{2,4})?\s*$")
FOOTER_RE = re.compile(r"^[A-Z0-9 ,.&/()-]{12,}$")
TITLE_TAIL_RE = re.compile(r",\s*20\d{2}")


# ============================================================
# MEMUAT BARIS
# ============================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def rebuild_rows_from_raw(
    raw_rows: list[list[str]]
) -> list[list[str]]:
    """
    Camelot kadang menaruh seluruh baris dalam satu sel
    dengan newline. Pecah berdasarkan newline.
    """
    rows = []

    for row in raw_rows:
        tokens = []

        for cell in row:
            for part in str(cell).split("\n"):
                part = strip_invisible(part)
                part = re.sub(r"\s+", " ", part).strip()

                if part:
                    tokens.append(part)

        if tokens:
            rows.append(tokens)

    return rows


def usable_rows(
    record: dict[str, Any]
) -> list[list[str]]:
    rows = record["clean_rows"]

    columns = max(
        (len(r) for r in rows),
        default=0,
    )

    has_newline = any(
        "\n" in str(c)
        for r in record.get("raw_rows", [])
        for c in r
    )

    if columns <= 2 and has_newline:
        return rebuild_rows_from_raw(
            record["raw_rows"]
        )

    return [
        [strip_invisible(c) for c in row]
        for row in rows
    ]


# ============================================================
# STRUKTUR SATU RECORD
# ============================================================

def row_text(row: list[str]) -> str:
    text = re.sub(
        r"\s+",
        " ",
        " ".join(c for c in row if c),
    ).strip()

    return re.sub(
        r"^C\s+atatan",
        "Catatan",
        text,
    )


def colnum_columns(row: list[str]) -> list[int]:
    return [
        i
        for i, c in enumerate(row)
        if COLNUM_CELL.fullmatch(c)
    ]


def parse_structure(
    record: dict[str, Any]
) -> dict[str, Any]:

    rows = usable_rows(record)

    colnum_idx = next(
        (
            i
            for i, row in enumerate(rows)
            if len(colnum_columns(row)) >= 2
        ),
        None,
    )

    caption_idx = next(
        (
            i
            for i, row in enumerate(rows[:4])
            if any(CAPTION_RE.match(c) for c in row)
        ),
        None,
    )

    section = ""
    title_end = -1
    title = ""

    number = record.get("table_number")
    is_continuation = bool(
        record.get("is_continuation")
    )

    if caption_idx is not None:

        section = " ".join(
            row_text(r)
            for r in rows[:caption_idx]
        )

        caption_info = parse_caption(
            row_text(rows[caption_idx])
        )

        is_continuation = (
            is_continuation
            or caption_info["is_continuation"]
        )

        limit = caption_idx + 4

        if colnum_idx is not None:
            limit = min(
                limit,
                max(caption_idx, colnum_idx - 1),
            )

        k = caption_idx

        if not caption_info["is_continuation"]:

            while k < min(limit, len(rows) - 1):

                text = row_text(rows[k])
                nxt = row_text(rows[k + 1]).lower()

                if (
                    YEAR_END_RE.search(text)
                    and not nxt.startswith("dan ")
                ):
                    break

                if len([
                    c for c in rows[k + 1] if c
                ]) > 2:
                    break

                k += 1

        title_end = k

        title_text = " ".join(
            row_text(r)
            for r in rows[
                caption_idx:title_end + 1
            ]
        )

        info = parse_caption(title_text)

        number = number or info["table_number"]

        title = CAPTION_RE.sub(
            "",
            title_text,
            count=1,
        ).strip()

        title = re.sub(
            r"^\s*[:\-–]\s*",
            "",
            title,
        )

    if not title and record.get("caption"):

        cap = parse_caption(record["caption"])

        number = number or cap["table_number"]

        if not cap["is_continuation"]:
            title = CAPTION_RE.sub(
                "",
                record["caption"],
                count=1,
            ).strip()

    head_rows: list[list[str]] = []
    colnum_cols: list[int] = []

    body_start = title_end + 1

    if (
        colnum_idx is not None
        and colnum_idx >= title_end
    ):

        head_rows = rows[
            title_end + 1:colnum_idx
        ]

        colnum_cols = colnum_columns(
            rows[colnum_idx]
        )

        while head_rows:

            text = row_text(head_rows[0])

            few_cells = (
                len([
                    c for c in head_rows[0] if c
                ]) <= 2
            )

            if (
                (few_cells and TITLE_TAIL_RE.search(text))
                or text.lower().startswith("menurut")
            ):

                if not is_continuation:
                    title = (
                        title + " " + text
                    ).strip()

                head_rows = head_rows[1:]

            else:
                break

        body_start = colnum_idx + 1

    body: list[list[str]] = []
    notes: list[str] = []
    in_notes = False

    for row in rows[body_start:]:

        text = row_text(row)

        if not text:
            continue

        if NOTE_ROW_RE.match(text):
            in_notes = True

        if in_notes:

            if (
                FOOTER_RE.match(
                    row[0] if row else ""
                )
                and len(row) <= 2
                and row[-1].isdigit()
            ):
                continue

            notes.append(text)
            continue

        if (
            len([c for c in row if c]) <= 2
            and row[-1].isdigit()
            and FOOTER_RE.match(row[0])
        ):
            continue

        body.append(row)

    return {
        "section": section,
        "number": number,
        "is_continuation": is_continuation,
        "title": title,
        "head_rows": head_rows,
        "colnum_cols": colnum_cols,
        "body_rows": body,
        "notes": notes,
    }


def column_labels(
    head_rows: list[list[str]],
    cols: list[int],
) -> list[str]:

    labels = []

    for c in cols:

        parts = []

        for candidate in (c, c - 1, c + 1):

            parts = [
                row[candidate]
                for row in head_rows
                if (
                    0 <= candidate < len(row)
                    and row[candidate]
                )
            ]

            if parts:
                break

        labels.append(
            re.sub(
                r"\s+",
                " ",
                " ".join(parts),
            ).strip()
        )

    return labels


# ============================================================
# BARIS LOGIS
# ============================================================

def map_row(
    row: list[str],
    cols: list[int],
) -> list[str]:

    cells = [""] * len(cols)

    for position, text in enumerate(row):

        if not text or is_noise_cell(text):
            continue

        j = min(
            range(len(cols)),
            key=lambda k: abs(position - cols[k]),
        )

        cells[j] = (
            cells[j] + " " + text
        ).strip()

    return cells


def logical_rows(
    body: list[list[str]],
    cols: list[int],
) -> list[dict[str, Any]]:

    if not cols:

        return [
            {
                "section": "",
                "cells": [row_text(r)],
            }
            for r in body
            if row_text(r)
        ]

    mapped = [
        map_row(r, cols)
        for r in body
    ]

    keyed_width = min(2, len(cols))

    def keyed(cells):
        return any(cells[:keyed_width])

    def has_values(cells):
        return any(cells[keyed_width:])

    result: list[dict[str, Any]] = []
    pending: list[list[str]] = []
    section = ""

    def merge(target, source):

        for j, v in enumerate(source):

            if not v:
                continue

            target[j] = (
                (target[j] + " " + v).strip()
                if target[j]
                else v
            )

    for cells in mapped:

        if not keyed(cells):
            pending.append(cells)
            continue

        if not has_values(cells):

            label = " ".join(
                c for c in cells[:keyed_width] if c
            )

            starts_lower = label[:1].islower()

            if result and (
                starts_lower
                or len(label.split()) > 4
            ):
                merge(result[-1]["cells"], cells)
                continue

            if (
                not starts_lower
                and len(label.split()) <= 4
            ):
                section = label
                continue

        row = {
            "section": section,
            "cells": list(cells),
        }

        for orphan in pending:

            conflict = any(
                v and row["cells"][j]
                for j, v in enumerate(orphan)
            )

            if conflict and result:
                merge(
                    result[-1]["cells"],
                    orphan,
                )
            else:
                merge(
                    row["cells"],
                    orphan,
                )

        pending = []
        result.append(row)

    for orphan in pending:

        if result:
            merge(
                result[-1]["cells"],
                orphan,
            )

    return result


# ============================================================
# LANGKAH 2: GABUNGKAN TABEL LANJUTAN
# ============================================================

def labels_compatible(
    a: list[str],
    b: list[str],
) -> bool:

    wa = set(
        " ".join(a).lower().split()
    )

    wb = set(
        " ".join(b).lower().split()
    )

    if not wa or not wb:
        return True

    return (
        len(wa & wb) / len(wa | wb)
        >= 0.5
    )


def sort_key(record: dict[str, Any]):

    box = (
        record.get("bbox_pymupdf")
        or [0, 0, 0, 0]
    )

    return (
        record["page"],
        box[1],
        record["table_id"],
    )


def build_logical_tables(
    records: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    groups: list[dict[str, Any]] = []

    for record in sorted(records, key=sort_key):

        s = parse_structure(record)

        n_cols = len(s["colnum_cols"])
        page = record["page"]

        target = None

        if s["number"]:

            for g in reversed(groups):

                if (
                    g["number"] == s["number"]
                    and page - g["last_page"] <= 3
                ):
                    target = g
                    break

        elif groups:

            last = groups[-1]

            own_labels = (
                column_labels(
                    s["head_rows"],
                    s["colnum_cols"],
                )
                if n_cols
                else []
            )

            consecutive = (
                page == last["last_page"] + 1
                and (
                    n_cols == 0
                    or n_cols == last["n_cols"]
                )
                and labels_compatible(
                    own_labels,
                    last["labels"],
                )
            )

            title_only_same_page = (
                page == last["last_page"]
                and last["n_cols"] == 0
                and not last["parts"][-1]["body_rows"]
            )

            if consecutive or title_only_same_page:
                target = last

        if target is None:

            target = {
                "number": s["number"],
                "title": s["title"],
                "section": s["section"],
                "labels": [],
                "n_cols": 0,
                "last_page": page,
                "pages": [],
                "parts": [],
                "rescued": False,
            }

            groups.append(target)

        target["parts"].append(s)
        target["pages"].append(page)
        target["last_page"] = page

        target["rescued"] = (
            target["rescued"]
            or bool(record.get("_rescued"))
        )

        if s["title"] and not target["title"]:
            target["title"] = s["title"]

        if s["section"] and not target["section"]:
            target["section"] = s["section"]

        if n_cols and not target["labels"]:

            target["labels"] = column_labels(
                s["head_rows"],
                s["colnum_cols"],
            )

            target["n_cols"] = n_cols
            target["cols"] = s["colnum_cols"]

    logical = []

    for g in groups:

        rows: list[dict[str, Any]] = []
        notes: list[str] = []

        for part in g["parts"]:

            cols = (
                part["colnum_cols"]
                or g.get("cols", [])
            )

            if (
                part["colnum_cols"]
                and len(part["colnum_cols"]) != g["n_cols"]
            ):
                cols = g.get(
                    "cols",
                    part["colnum_cols"],
                )

            rows.extend(
                logical_rows(
                    part["body_rows"],
                    cols,
                )
            )

            for n in part["notes"]:

                if n not in notes:
                    notes.append(n)

        logical.append({
            "table_number": g["number"],
            "title": g["title"],
            "section_heading": g["section"],
            "pages": sorted(set(g["pages"])),
            "column_labels": g["labels"],
            "row_count": len(rows),
            "rows": rows,
            "notes": notes,
            "rescued_from_rejected": g["rescued"],
        })

    return logical


# ============================================================
# FILTER NOISE TEKS VERTIKAL
# ============================================================

def is_vertical_text_noise(
    table: dict[str, Any]
) -> bool:

    rows = table.get("rows", [])
    labels = table.get("column_labels", [])

    if not rows or labels:
        return False

    cells = [
        str(cell).strip()
        for row in rows
        for cell in row.get("cells", [])
        if str(cell).strip()
    ]

    if len(cells) < 10:
        return False

    single_column = all(
        len(row.get("cells", [])) <= 1
        for row in rows
    )

    if not single_column:
        return False

    short_cells = sum(
        len(cell) <= 2
        for cell in cells
    )

    short_ratio = short_cells / len(cells)

    return short_ratio >= 0.80


# ============================================================
# LANGKAH 3: CHUNKING TABEL
# ============================================================

def clean_value(value: str) -> str:

    value = strip_invisible(value)

    return re.sub(
        r"^[–—−]+$",
        "-",
        value,
    )


def format_row(
    row: dict[str, Any],
    labels: list[str],
) -> str:

    parts = []

    for j, value in enumerate(row["cells"]):

        value = clean_value(value)

        if not value:
            continue

        label = (
            labels[j]
            if j < len(labels) and labels[j]
            else ""
        )

        parts.append(
            f"{label}: {value}"
            if label
            else value
        )

    text = " | ".join(parts)

    return (
        f"[{row['section']}] {text}"
        if row["section"]
        else text
    )


def chunk_table(
    table: dict[str, Any],
    doc: str,
    start_id: int,
) -> list[dict[str, Any]]:

    number = (
        table["table_number"]
        or "tanpa-nomor"
    )

    heading = (
        f"Tabel {number}"
        if table["table_number"]
        else "Tabel"
    )

    if table["title"]:
        heading += f" - {table['title']}"

    pages = table["pages"]

    page_label = (
        f"halaman {pages[0]}"
        if len(pages) == 1
        else f"halaman {pages[0]}-{pages[-1]}"
    )

    header = f"{heading} ({page_label})"

    lines = [
        format_row(
            r,
            table["column_labels"],
        )
        for r in table["rows"]
    ]

    lines = [
        line
        for line in lines
        if line
    ]

    notes_text = " ".join(
        table["notes"]
    ).strip()

    chunks = []
    current = []
    size = len(header)
    first_row = 0

    def flush(last_index):

        nonlocal current, size, first_row

        if not current:
            return

        chunks.append((
            first_row,
            last_index,
            list(current),
        ))

        current = []
        size = len(header)
        first_row = last_index + 1

    for i, line in enumerate(lines):

        if (
            current
            and size + len(line) + 1 > MAX_CHARS
        ):
            flush(i - 1)

        current.append(line)
        size += len(line) + 1

    flush(len(lines) - 1)

    if not chunks and notes_text:
        chunks = [(0, 0, [])]

    records = []

    for n, (a, b, body) in enumerate(chunks):

        text_lines = [header] + body

        if (
            n == len(chunks) - 1
            and notes_text
        ):
            text_lines.append(
                f"Catatan/Sumber: {notes_text}"
            )

        records.append({
            "chunk_id": (
                f"{doc}-tbl-{start_id + n}"
            ),
            "document": doc,
            "content_type": "table",
            "table_number": table["table_number"],
            "title": table["title"],
            "pages": pages,
            "row_range": [a, b],
            "part": f"{n + 1}/{len(chunks)}",
            "rescued_from_rejected": (
                table["rescued_from_rejected"]
            ),
            "text": "\n".join(text_lines),
        })

    return records


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help=(
            "Folder berisi tables.jsonl "
            "dan rejected_tables.jsonl"
        ),
    )

    args = parser.parse_args()
    d: Path = args.dir.resolve()

    if not d.exists():
        raise FileNotFoundError(
            f"Folder input tidak ditemukan: {d}"
        )

    tables = load_jsonl(
        d / "tables.jsonl"
    )

    rejected = load_jsonl(
        d / "rejected_tables.jsonl"
    )

    rescued = [
        dict(r, _rescued=True)
        for r in rejected
        if (
            r.get("classification", {}).get("kind")
            == "data_table"
        )
    ]

    rescued_pages = sorted({
        r["page"]
        for r in rescued
        if (
            r.get("quality", {}).get(
                "bbox_area_ratio", 0
            ) >= 0.9
        )
    })

    records = tables + rescued

    doc = (
        records[0]["document"]
        if records
        else d.parent.name
    )

    # Bangun tabel logis
    logical_all = build_logical_tables(records)

    # Filter noise
    logical = []
    filtered_noise = []

    for table in logical_all:

        if is_vertical_text_noise(table):
            filtered_noise.append(table)
        else:
            logical.append(table)

    # Buat table chunks
    chunks: list[dict[str, Any]] = []

    for table in logical:

        chunks.extend(
            chunk_table(
                table,
                doc,
                len(chunks) + 1,
            )
        )

    # Simpan logical tables yang lolos filter
    with (
        d / "logical_tables.jsonl"
    ).open("w", encoding="utf-8") as f:

        for table in logical:
            f.write(
                json.dumps(
                    table,
                    ensure_ascii=False,
                ) + "\n"
            )

    # Simpan table chunks
    with (
        d / "table_chunks.jsonl"
    ).open("w", encoding="utf-8") as f:

        for chunk in chunks:
            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                ) + "\n"
            )

    # Simpan tabel yang difilter untuk audit
    with (
        d / "filtered_noise_tables.jsonl"
    ).open("w", encoding="utf-8") as f:

        for table in filtered_noise:
            f.write(
                json.dumps(
                    table,
                    ensure_ascii=False,
                ) + "\n"
            )

    # Simpan halaman tabel hasil rescue
    (d / "rescued_pages.json").write_text(
        json.dumps(
            {
                "skip_narrative_pages": rescued_pages
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Ringkasan
    print("\n=== HASIL TABLE PROCESSING ===")
    print(
        f"Record tabel dimuat        : {len(tables)}"
    )
    print(
        f"Record diselamatkan        : {len(rescued)}"
    )
    print(
        f"Tabel logis sebelum filter : {len(logical_all)}"
    )
    print(
        f"Tabel noise difilter       : {len(filtered_noise)}"
    )
    print(
        f"Tabel logis dipertahankan  : {len(logical)}"
    )
    print(
        f"Table chunks               : {len(chunks)}"
    )
    print(
        f"Halaman narasi dilewati    : {rescued_pages}"
    )
    print(f"Output                     : {d}")


if __name__ == "__main__":
    main()