# Kawan Sanggau — Gemini RAG + Qdrant + WhatsApp

Versi perbaikan berdasarkan file yang dikirim. Arsitektur memakai Gemini untuk embedding dan generasi jawaban, Qdrant lokal untuk vector index, dan `whatsapp-web.js` untuk kanal WhatsApp.

## 1. Persiapan

- Node.js 20 atau lebih baru disarankan.
- Docker Desktop untuk menjalankan Qdrant lokal.
- API key Google Gemini dari Google AI Studio.
- PDF yang memang boleh diproses, letakkan di folder `knowledge/`.

## 2. Pasang dependensi dan konfigurasi

Di terminal PowerShell pada folder proyek:

```powershell
npm install
Copy-Item .env.example .env
```

Edit `.env`, isi `GOOGLE_API_KEY` dengan API key milikmu. Jangan unggah `.env` ke Git atau membagikan API key.

## 3. Jalankan Qdrant

```powershell
docker compose up -d
```

Dashboard Qdrant biasanya tersedia di `http://localhost:6333/dashboard`.

## 4. Masukkan dokumen dan buat indeks

Taruh file PDF di `knowledge/`, kemudian jalankan:

```powershell
npm run ingest
```

Proses ini membaca PDF, memecah teks, membuat embedding dengan Gemini, dan menyimpan vektor di Qdrant. File yang tidak berubah dilewati menggunakan hash. Jika file berubah, kode mencoba menghapus chunk lama dengan metadata nama file sebelum menambahkan versi baru.

**Penting:** jika koleksi Qdrant sebelumnya berisi data dari model embedding Hugging Face, jangan gunakan koleksi lama bersama embedding Gemini karena dimensi/ruang vektornya bisa berbeda. Untuk migrasi pertama, hentikan bot dan hapus koleksi lama melalui Qdrant Dashboard, atau ubah `QDRANT_COLLECTION` menjadi nama koleksi baru di `.env`, lalu jalankan `npm run ingest` lagi.

## 4b. Pipeline alternatif: Python + PaddleOCR + NVIDIA

Selain pipeline JavaScript di atas, proyek ini memiliki pipeline ekstraksi PDF
yang lebih rinci di `python-extractor/`. Pipeline ini memakai PaddleOCR
`PPStructureV3` untuk deteksi tabel, text layer PDF untuk nilai sel, dan model
embedding NVIDIA untuk Qdrant.

Untuk menambah publikasi BPS baru:

```powershell
# dari folder python-extractor
..\..\.venv\Scripts\python.exe batch_paddle.py --docs "nama_file" --no-ocr --embed
```

Deteksi tabel akan dijalankan otomatis bila belum tersedia. Panduan lengkap,
opsi, dan asumsi format ada di
[`python-extractor/README_PIPELINE.md`](python-extractor/README_PIPELINE.md).

## 5. Uji satu pertanyaan sebelum WhatsApp

```powershell
npm run ask -- "Berapa luas wilayah menurut dokumen?"
```

Jika belum ada indeks, jalankan ingestion terlebih dahulu. Skor retrieval dan ambang `RETRIEVAL_SCORE_THRESHOLD` perlu diuji dengan pertanyaan yang jawabannya sudah diketahui. Nilai awal `0.35` hanya titik awal, bukan jaminan akurasi.

## 6. Jalankan WhatsApp

```powershell
npm start
```

Pindai QR di terminal melalui WhatsApp > Perangkat tertaut. Sesi disimpan di `wa_session/`. Jangan bagikan folder tersebut karena berisi data autentikasi.

## Catatan implementasi

- Sapaan, menu bantuan, ucapan terima kasih, cooldown, pesan nonteks, dan penanganan error dari `bot.js` tetap dipertahankan.
- `cek.json` tidak digunakan sebagai sumber jawaban RAG; ia berisi jawaban statis dan data tentang Kabupaten Sanggau. Chatbot RAG menjawab dari PDF di `knowledge/`.
- Pastikan cakupan dokumen konsisten. Nama PDF pada tangkapan layar tampak merujuk Kecamatan Kapuas, sedangkan `cek.json` berisi beberapa data Kabupaten Sanggau. Jangan mencampur cakupan kecamatan dan kabupaten tanpa metadata serta instruksi yang jelas.
- PDF hasil scan mungkin membutuhkan OCR terlebih dahulu.
- Ini prototipe lokal. Untuk deployment publik, pertimbangkan autentikasi, batas penggunaan, logging yang tidak menyimpan data pribadi, dan kebijakan privasi WhatsApp.
