from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import camelot
import pymupdf

from common import read_jsonl
from pipeline import (
    MANUAL_DIR,
    NUMERIC_PATTERN,
    OUTPUT_DIR,
    PDF_DIR,
    camelot_bbox_to_pymupdf,
    clean_table_rows,
    read_page_lines,
)

CAPTION_RE = re.compile(r"^(?:Tabel|Table)\s+(\d+(?:\.\d+)*)", re.I)
LEADER_RE = re.compile(r"(?:\.\s?){4,}")


# ============================================================
# UTIL
# ============================================================

def resolve_pdf(name: str) -> Path:
    path = Path(name)
    if path.exists():
        return path
    candidate = PDF_DIR / name
    if candidate.suffix.lower() != ".pdf":
        candidate = candidate.with_suffix(".pdf")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"PDF tidak ditemukan: {name} (dicari juga di {PDF_DIR})")


def pdf_captions(pdf_path: Path) -> list[dict[str, Any]]:
    """Caption 'Tabel x.y ...' yang ada di PDF (daftar tabel dan rujukan kalimat dilewati)."""
    found: list[dict[str, Any]] = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            lines = [
                line for line in read_page_lines(page)
                if CAPTION_RE.match(line["text"])
                and not LEADER_RE.search(line["text"])
                and not line["text"].rstrip().endswith(".")
            ]
            if len(lines) >= 6:      # halaman Daftar Tabel
                continue
            for line in lines:
                found.append({
                    "page": page_no,
                    "number": CAPTION_RE.match(line["text"]).group(1),
                    "text": line["text"],
                    "y0": line["y0"],
                })
    return found


# ============================================================
# REPORT: tabel mana yang hilang
# ============================================================

def cmd_report(args: argparse.Namespace) -> None:
    pdf = resolve_pdf(args.pdf)
    doc_dir = OUTPUT_DIR / pdf.stem / args.label

    tables = read_jsonl(doc_dir / "tables.jsonl")
    rejected = read_jsonl(doc_dir / "rejected_tables.jsonl")
    if not tables and not rejected:
        print(f"Tidak ada hasil di {doc_dir}. Jalankan pipeline.py dulu.")
        return

    have_numbers: set[str] = set()
    captionless_pages: set[int] = set()
    for table in tables:
        match = CAPTION_RE.match((table.get("caption") or "").strip())
        if match:
            have_numbers.add(match.group(1))
        else:
            captionless_pages.add(table["page"])

    captions = pdf_captions(pdf)
    missing = [c for c in captions if c["number"] not in have_numbers]

    print(f"Caption tabel di PDF     : {len(captions)}")
    print(f"Tabel terekstrak         : {len(tables)}")
    print(f"Caption tanpa hasil      : {len(missing)}\n")

    for cap in missing:
        page = cap["page"]
        print(f"Tabel {cap['number']}  (hal. {page})  {cap['text'][:80]}")
        if page in captionless_pages:
            print("  catatan: ada tabel terekstrak di halaman ini tanpa caption; "
                  "mungkin sudah masuk, hanya caption-nya tidak terbaca.")
        on_page = [r for r in rejected if r["page"] == page]
        if not on_page:
            print("  tidak ada deteksi sama sekali di halaman ini "
                  "(engine tidak melihat garis/kolom tabel).")
        for rec in on_page:
            q = rec.get("quality", {})
            print(f"  deteksi ditolak: status={rec.get('status')} "
                  f"ukuran={q.get('row_count')}x{q.get('column_count')} "
                  f"alasan={q.get('rejection_reasons')} "
                  f"jenis={rec.get('classification', {}).get('kind')}")
        print(f"  coba: python rescue_tables.py try --pdf {pdf.name} --table {cap['number']}\n")


# ============================================================
# TRY: coba semua strategi pada satu halaman
# ============================================================

def numeric_ratio(rows: list[list[str]]) -> float:
    cells = [c for r in rows for c in r if c.strip()]
    if not cells:
        return 0.0
    return sum(bool(NUMERIC_PATTERN.match(c.strip())) for c in cells) / len(cells)


def collect_results(
    pdf_path: Path, page_no: int, caption_y0: float | None
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def add(name: str, index: int, raw_rows: list[list[Any]], bbox, rect) -> None:
        raw = [["" if c is None else str(c) for c in row] for row in raw_rows]
        rows, _ = clean_table_rows(raw)
        if len(rows) < 2:
            return
        cells = [c for r in rows for c in r if c]
        results.append({
            "id": f"{name}#{index}",
            "raw_rows": raw,
            "rows": rows,
            "bbox": [float(v) for v in bbox],
            "rect": [float(v) for v in rect],
            "nrows": len(rows),
            "ncols": max(len(r) for r in rows),
            "nonempty": len(cells),
            "numeric": numeric_ratio(rows),
        })

    with pymupdf.open(pdf_path) as doc:
        page = doc[page_no - 1]
        width, height = page.rect.width, page.rect.height
        top = max(0.0, caption_y0 - 2) if caption_y0 is not None else height * 0.05
        clip = pymupdf.Rect(0, top, width, height * 0.95)

        # --- Camelot
        configs = [
            ("camelot_lattice", dict(flavor="lattice", line_scale=40,
                                     copy_text=["h", "v"], split_text=True)),
            ("camelot_lattice_halus", dict(flavor="lattice", line_scale=15,
                                           copy_text=["h", "v"], split_text=True)),
            ("camelot_stream", dict(flavor="stream", split_text=True)),
            ("camelot_stream_lebar", dict(flavor="stream", split_text=True,
                                          edge_tol=500, row_tol=8)),
        ]
        for name, kwargs in configs:
            try:
                found = camelot.read_pdf(
                    str(pdf_path), pages=str(page_no), suppress_stdout=True, **kwargs
                )
            except Exception as exc:
                print(f"  ({name} gagal: {exc})")
                continue
            for i, table in enumerate(found, start=1):
                bbox = tuple(float(v) for v in table._bbox)
                rect = camelot_bbox_to_pymupdf(bbox, page)
                add(name, i, table.df.values.tolist(), bbox, rect)

        # --- PyMuPDF
        pm_configs = [
            ("pymupdf_garis", {}),
            ("pymupdf_garis_clip", {"clip": clip}),
            ("pymupdf_teks", {"strategy": "text"}),
            ("pymupdf_teks_clip", {"strategy": "text", "clip": clip}),
        ]
        for name, kwargs in pm_configs:
            try:
                found = page.find_tables(**kwargs)
            except Exception as exc:
                print(f"  ({name} gagal: {exc})")
                continue
            for i, tab in enumerate(found.tables, start=1):
                add(name, i, tab.extract(), tab.bbox, tab.bbox)

    return results


def score(result: dict[str, Any]) -> float:
    if result["nrows"] < 3 or result["ncols"] < 2:
        return 0.0
    return result["nonempty"] * (0.3 + result["numeric"])


def preview(rows: list[list[str]], limit: int = 4, width: int = 14) -> str:
    lines = []
    for row in rows[:limit]:
        lines.append("      | " + " | ".join(c[:width].ljust(width) for c in row[:6]))
    return "\n".join(lines)


def cmd_try(args: argparse.Namespace) -> None:
    pdf = resolve_pdf(args.pdf)
    captions = pdf_captions(pdf)

    page_no = args.page
    caption_y0 = None
    if args.table:
        match = [c for c in captions if c["number"] == args.table]
        if not match:
            raise SystemExit(f"Caption 'Tabel {args.table}' tidak ditemukan; pakai --page.")
        page_no, caption_y0 = match[0]["page"], match[0]["y0"]
        print(f"Tabel {args.table} ada di halaman {page_no}: {match[0]['text'][:80]}")
    elif page_no is None:
        raise SystemExit("Isi --page atau --table.")
    else:
        on_page = [c for c in captions if c["page"] == page_no]
        if on_page:
            caption_y0 = on_page[0]["y0"]

    results = collect_results(pdf, page_no, caption_y0)
    if not results:
        print("Tidak ada strategi yang menghasilkan tabel di halaman ini. "
              "Kemungkinan halaman berupa gambar (scan); butuh OCR.")
        return

    best = max(results, key=score)
    print(f"\nHasil per strategi (halaman {page_no}):\n")
    for r in results:
        mark = "  <-- saran" if r is best and score(r) > 0 else ""
        print(f"  {r['id']:<28} {r['nrows']:>3} baris x {r['ncols']:<2} kolom  "
              f"terisi={r['nonempty']:<4} angka={r['numeric']:.0%}{mark}")
        print(preview(r["rows"]))

    if not args.use:
        print(f"\nPilih satu dan simpan:\n  python rescue_tables.py try --pdf {pdf.name} "
              f"--page {page_no} --use {best['id']}")
        return

    chosen = next((r for r in results if r["id"] == args.use), None)
    if chosen is None:
        raise SystemExit(f"ID '{args.use}' tidak ada. Pilihan: {[r['id'] for r in results]}")

    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    path = MANUAL_DIR / f"{pdf.stem}.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    # satu tabel manual per halaman dan caption
    existing = [
        c for c in existing
        if not (c["page"] == page_no and c.get("table") == args.table)
    ]
    existing.append({
        "page": page_no,
        "table": args.table,
        "strategy": chosen["id"],
        "raw_rows": chosen["raw_rows"],
        "bbox": chosen["bbox"],
        "rect": chosen["rect"],
        "accuracy": 0.0,
        "whitespace": 0.0,
    })
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDisimpan ke {path}")
    print("Jalankan ulang: python pipeline.py && python cleaner.py && python chunk.py")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Cari dan selamatkan tabel yang gagal diekstrak.")
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser("report", help="Daftar caption tabel yang tidak punya hasil.")
    rep.add_argument("--pdf", required=True)
    rep.add_argument("--label", default="lattice", help="lattice, stream, atau pymupdf")
    rep.set_defaults(func=cmd_report)

    tr = sub.add_parser("try", help="Coba semua strategi pada satu halaman/tabel.")
    tr.add_argument("--pdf", required=True)
    tr.add_argument("--page", type=int)
    tr.add_argument("--table", help="Nomor tabel, mis. 3.1.2")
    tr.add_argument("--use", help="ID strategi yang disimpan, mis. camelot_stream#1")
    tr.set_defaults(func=cmd_try)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()