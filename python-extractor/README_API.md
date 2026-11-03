# API RAG Kawan Sanggau (Python + NVIDIA NIM)

Layanan HTTP yang menghubungkan chatbot WhatsApp (Node.js) dengan pipeline
RAG Python: embedding NVIDIA, Qdrant `kcda_knowledge_nvidia_v4`, dan
generation NVIDIA NIM.

Pipeline lama di Node.js (`src/rag.js`, `src/gemini-ai.js`) tidak dipakai lagi
oleh bot. Embedding dan retrieval tidak diubah.

## 1. Konfigurasi

Isi `.env` di root proyek:

```env
QDRANT_URL=http://localhost:6333
NVIDIA_QDRANT_COLLECTION=kcda_knowledge_nvidia_v4
NVIDIA_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b

NVIDIA_API_KEY=...
NVIDIA_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_TIMEOUT_SECONDS=60
NVIDIA_MAX_RETRIES=3
NVIDIA_DISABLE_THINKING=true

RETRIEVAL_SCORE_THRESHOLD=0.35
RETRIEVAL_GLOSSARY_THRESHOLD=0.25
RAG_TOP_K=5

PYTHON_API_URL=http://127.0.0.1:8000
PYTHON_API_TIMEOUT_MS=60000
```

`NVIDIA_MODEL` didahulukan; `NVIDIA_CHAT_MODEL` tetap didukung sebagai alias.

## 2. Menjalankan

```bash
# dari folder python-extractor
../../.venv/Scripts/python.exe -m pip install -r requirements.txt
../../.venv/Scripts/python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Qdrant harus aktif lebih dulu (`docker compose up -d`).

## 3. Endpoint

| Metode | Path | Fungsi |
|---|---|---|
| GET | `/health` | Status konfigurasi NIM, Qdrant, collection, dimensi |
| POST | `/ask` | Retrieval + generation (pertanyaan statistik) |
| POST | `/ask/general` | Generation tanpa retrieval (sapaan/umum) |

Contoh:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Berapa produksi karet Kecamatan Jangkang?\"}"
```

Respons sukses:

```json
{
  "ok": true,
  "answer": "...",
  "in_scope": true,
  "top_score": 0.62,
  "pages": [123],
  "sources": [{"document": "Kecamatan Jangkang Dalam Angka 2026"}],
  "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
  "latency_ms": 4200
}
```

Respons gagal **selalu** `ok: false` dengan HTTP non-200 dan `error.code`
(`rate_limit`, `timeout`, `model_not_found`, `auth`, `upstream_error`,
`empty_response`, `retrieval_error`, `config`). Jawaban tidak pernah
dikembalikan seolah-olah berhasil.

## 4. Uji tanpa WhatsApp

```bash
# generation sederhana
curl -X POST http://127.0.0.1:8000/ask/general \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Halo, kamu siapa?\"}"

# CLI Python
../../.venv/Scripts/python.exe rag.py --question "Berapa jumlah penduduk Kecamatan Kapuas?"

# dari Node.js
npm run ask "Berapa jumlah penduduk Kecamatan Kapuas?"
```

## 5. Catatan

- Jangan menjalankan `embedder.py` atau `npm run ingest` untuk pengujian ini.
- Koleksi, model embedding, dan dimensi 2048 tidak diubah.
- `NVIDIA_DISABLE_THINKING=true` diperlukan untuk
  `nvidia/nemotron-3.5-super-vl-120b-a12b`; tanpa itu jawaban bisa kosong.
