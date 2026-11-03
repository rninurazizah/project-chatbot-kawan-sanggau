"""
Menjalankan seluruh pipeline Paddle hybrid untuk semua PDF di knowledge/.

Urutan per dokumen:
  0. pipeline.py        (deteksi tabel; otomatis jika belum ada)
  1. paddle_extract.py  (OCR tabel, resumable per tabel)
  2. paddle_build.py    (narasi text layer + tabel)
  3. cleaner.py         (cleaning + gabung tabel lanjutan)

Setelah semua dokumen:
  4. normalisasi_v2.py  (gabungan multi-dokumen)
  5. chunk.py
  6. embedder.py        (opsional, ke collection NVIDIA_QDRANT_COLLECTION)

Menambah PDF baru cukup dengan meletakkan file di knowledge/, lalu:
    python batch_paddle.py --docs nama_file --no-ocr --embed

Deteksi tabel akan dijalankan otomatis bila
extracted/<nama_file>/lattice_all/tables.jsonl belum tersedia.

Contoh:
    python batch_paddle.py                       # semua dokumen unik
    python batch_paddle.py --docs pdf_4846        # sebagian
    python batch_paddle.py --limit 3 --docs pdf_4846
    python batch_paddle.py --skip-extract         # hanya build/clean/normalize
    python batch_paddle.py --embed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
KNOWLEDGE = BASE_DIR / "knowledge"
EXTRACTED = BASE_DIR / "extracted"
OUTPUT = BASE_DIR / "extracted_paddle"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def unique_pdfs() -> list[Path]:
    seen: set[str] = set()
    pdfs: list[Path] = []

    for path in sorted(KNOWLEDGE.glob("*.pdf")):
        digest = file_hash(path)

        if digest in seen:
            print(f"Lewati duplikat: {path.name}")
            continue

        seen.add(digest)
        pdfs.append(path)

    return pdfs


def run(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))

    result = subprocess.run(command)

    if result.returncode != 0:
        raise SystemExit(f"Gagal: {label} (exit {result.returncode})")


def seed_tables_from_detection(detection_path: Path, tables_path: Path) -> None:
    """
    Buat tabel_paddle.jsonl awal dari hasil deteksi saja.

    Nilai sel akan diisi dari text layer oleh paddle_build.py. OCR
    PPStructureV3 hanya dijalankan untuk tabel yang gagal.
    """

    if tables_path.exists():
        return

    records = []

    with detection_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            detection = json.loads(line)

            records.append({
                **detection,
                "engine": "detection_only",
                "flavor": "textlayer_seed",
                "raw_rows": [],
                "clean_rows": [],
                "status": "needs_review",
            })

    tables_path.parent.mkdir(parents=True, exist_ok=True)

    with tables_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Seed tabel dari deteksi: {tables_path} ({len(records)} tabel)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch pipeline Paddle hybrid untuk semua dokumen."
    )
    parser.add_argument("--docs", default="", help="Filter nama file (substring), pisah koma.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Batasi jumlah tabel per dokumen (untuk uji coba).")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--skip-extract", action="store_true",
                        help="Jangan jalankan OCR tabel.")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Bangun tabel dari text layer; OCR hanya untuk tabel yang gagal.")
    parser.add_argument("--force", action="store_true",
                        help="OCR ulang walau cache sudah ada.")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Jangan jalankan pipeline.py otomatis saat deteksi tabel belum ada.")
    parser.add_argument("--embed", action="store_true",
                        help="Jalankan embedder setelah chunking.")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Lewati normalisasi/chunking/embedding.")
    args = parser.parse_args()

    filters = [item.strip() for item in args.docs.split(",") if item.strip()]

    pdfs = unique_pdfs()

    if filters:
        pdfs = [
            path for path in pdfs
            if any(item.lower() in path.name.lower() for item in filters)
        ]

    if not pdfs:
        raise SystemExit("Tidak ada PDF yang cocok.")

    print(f"Dokumen diproses: {len(pdfs)}")

    for index, pdf in enumerate(pdfs, start=1):
        stem = pdf.stem
        detection = EXTRACTED / stem / "lattice_all" / "tables.jsonl"
        out_dir = OUTPUT / stem

        print(f"\n{'#' * 70}")
        print(f"[{index}/{len(pdfs)}] {pdf.name}")
        print(f"{'#' * 70}")

        if not detection.exists():
            if args.skip_detection:
                print(f"Lewati: deteksi tabel tidak ditemukan ({detection})")
                continue

            run(
                f"Deteksi tabel: {pdf.name}",
                [
                    PYTHON, "pipeline.py",
                    "--input", str(KNOWLEDGE),
                    "--output", str(EXTRACTED),
                    "--engine", "all",
                    "--flavor", "lattice",
                    "--docs", pdf.name,
                ],
            )

        if not detection.exists():
            print(f"Lewati: deteksi tabel tetap tidak ditemukan ({detection})")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        tables_file = out_dir / "tables_paddle.jsonl"

        should_extract = (
            not args.no_ocr
            and not args.skip_extract
            and (args.force or not tables_file.exists())
        )

        if args.no_ocr and not tables_file.exists():
            seed_tables_from_detection(detection, tables_file)

        if should_extract:
            command = [
                PYTHON, "paddle_extract.py",
                "--pdf", str(pdf),
                "--detection", str(detection),
                "--out", str(out_dir),
                "--dpi", str(args.dpi),
            ]

            if args.limit:
                command.extend(["--limit", str(args.limit)])

            if args.force:
                command.append("--force")

            run(f"OCR tabel: {pdf.name}", command)
        else:
            print(f"OCR dilewati (cache tersedia): {tables_file}")

        run(
            f"Build narasi+tabel: {pdf.name}",
            [
                PYTHON, "paddle_build.py",
                "--pdf", str(pdf),
                "--detection", str(detection),
                "--tables", str(tables_file),
                "--out", str(out_dir),
            ],
        )

        run(
            f"Cleaning: {pdf.name}",
            [PYTHON, "cleaner.py", "--dir", str(out_dir)],
        )

    if args.no_normalize:
        print("\nSelesai (normalisasi dilewati).")
        return

    run("Normalisasi multi-dokumen", [PYTHON, "normalisasi_v2.py"])
    run("Chunking", [PYTHON, "chunk.py"])

    if args.embed:
        run(
            "Embedding NVIDIA",
            [
                PYTHON, "embedder.py",
                "--input", str(OUTPUT / "chunks.jsonl"),
            ],
        )

    print("\nBatch selesai.")


if __name__ == "__main__":
    main()
