# Pipeline Ekstraksi PDF BPS (Python + PaddleOCR + NVIDIA)

Dokumen ini menjelaskan cara menambahkan publikasi BPS baru dan mengekstraknya
menjadi chunk yang siap di-embedding.

Kode berada di `python-extractor/`. Semua perintah dijalankan dari folder itu.

## Ringkasan alur

```
knowledge/*.pdf
      │
      ├─ pipeline.py        deteksi halaman + bbox tabel
      │                     → extracted/<nama-pdf>/lattice_all/tables.jsonl
      │
      ├─ paddle_extract.py  OCR PPStructureV3 (hanya fallback)
      ├─ paddle_build.py    narasi + nilai sel dari text layer PDF
      ├─ cleaner.py         cleaning, dedup, gabung tabel lanjutan
      │
      ├─ normalisasi_v2.py  normalisasi semua dokumen → normalized_data.jsonl
      ├─ chunk.py           serialisasi → chunks.jsonl
      └─ embedder.py        embedding NVIDIA → Qdrant
```

`batch_paddle.py` menjalankan seluruh alur di atas. Sejak versi ini, deteksi
tabel juga dijalankan otomatis bila `lattice_all/tables.jsonl` belum ada.

## Menambah satu dokumen baru

1. Letakkan PDF di `knowledge/`.
2. Jalankan:

```bash
../../.venv/Scripts/python.exe batch_paddle.py --docs "nama_file" --no-ocr --embed
```

`--docs` menerima substring nama file dan boleh dipisah koma:

```bash
--docs "pdf_3240,pdf_4846"
```

Untuk PDF digital (punya text layer), `--no-ocr` adalah jalur yang disarankan:
nilai sel diambil dari text layer, OCR hanya dipakai untuk tabel yang gagal.

Untuk PDF hasil scan (tidak punya text layer), jangan pakai `--no-ocr`:

```bash
../../.venv/Scripts/python.exe batch_paddle.py --docs "nama_file" --embed
```

Tabel masih bisa diekstrak lewat OCR, tetapi narasi dan glosarium akan kosong
karena keduanya dibangun dari text layer.

## Jika ingin menjalankan tahap secara terpisah

```bash
# 1. Deteksi tabel untuk satu PDF saja
../../.venv/Scripts/python.exe pipeline.py \
  --input ../knowledge --output ../extracted \
  --engine all --flavor lattice --docs "nama_file"

# 2. Build + cleaning
../../.venv/Scripts/python.exe batch_paddle.py --docs "nama_file" --no-ocr --skip-detection

# 3. Normalisasi + chunk + embedding
../../.venv/Scripts/python.exe normalisasi_v2.py
../../.venv/Scripts/python.exe chunk.py
../../.venv/Scripts/python.exe embedder.py --input ../extracted_paddle/chunks.jsonl
```

## Opsi penting

| Opsi | Fungsi |
|---|---|
| `--docs` | Batasi pemrosesan ke PDF tertentu (substring, pisah koma) |
| `--no-ocr` | Gunakan text layer; OCR hanya fallback |
| `--skip-extract` | Jangan jalankan OCR tabel |
| `--skip-detection` | Jangan jalankan `pipeline.py` otomatis |
| `--force` | OCR ulang walau cache sudah ada |
| `--limit N` | Batasi jumlah tabel per dokumen (untuk uji coba) |
| `--embed` | Langsung embedding setelah chunking |

## Sifat inkremental

- PDF dengan isi identik dilewati berdasarkan hash.
- Chunk yang sudah ada di Qdrant dilewati berdasarkan `chunk_id`.
- Menambah PDF baru tidak mengubah chunk dokumen lama karena ID chunk
  deterministik.

Jika sebuah PDF **diganti** dengan versi revisi, chunk lama bisa tertinggal
apabila struktur halaman berubah. Untuk kasus itu, hapus dulu points dengan
`document_id` tersebut, atau buat collection baru lewat
`NVIDIA_QDRANT_COLLECTION`, lalu embed ulang.

## Asumsi format

Pipeline ini dirancang untuk publikasi **BPS "Kecamatan Dalam Angka"**:

- Judul memuat pola `KECAMATAN ... DALAM ANGKA ... 20xx`.
- Caption tabel memakai pola `Tabel X.Y.Z`.
- Glosarium dikenali dari judul `PENJELASAN UMUM`, `SINGKATAN`,
  `ISTILAH DAN DEFINISI`.
- Kabupaten dibaca dari halaman judul; fallback `Sanggau`.
- Nilai sel memakai text layer PDF, bukan OCR penuh.

Untuk jenis publikasi lain, periksa kembali pola metadata, caption tabel, dan
deteksi glosarium di `paddle_build.py` serta `normalisasi_v2.py`.

## Keluaran

| File | Isi |
|---|---|
| `extracted_paddle/<doc>/narrative_clean.jsonl` | Narasi bersih per dokumen |
| `extracted_paddle/<doc>/glossary.jsonl` | Glosarium |
| `extracted_paddle/<doc>/tables_clean.jsonl` | Tabel bersih |
| `extracted_paddle/normalized_data.jsonl` | Gabungan semua dokumen |
| `extracted_paddle/chunks.jsonl` | Chunk final untuk embedding |
| `extracted_paddle/normalization_report.json` | Laporan validasi |

## Pemeriksaan setelah ekstraksi

1. `normalization_report.json` → `validation.all_valid` harus `true`.
2. Jumlah chunk di `chunks.jsonl` masuk akal.
3. Jumlah points di Qdrant bertambah sesuai jumlah chunk baru.
4. Uji beberapa pertanyaan tentang kecamatan baru:

```bash
../../.venv/Scripts/python.exe rag.py --question "Berapa luas wilayah Kecamatan X?"
```

## Catatan lingkungan

- Gunakan Python dari `.venv`: `../../.venv/Scripts/python.exe`.
- `paddlepaddle 3.2.2` adalah versi yang stabil di CPU/Windows.
- Qdrant lokal: `docker compose up -d` dari folder proyek.
- Jangan commit `.env`; gunakan `.env.example` sebagai template.
