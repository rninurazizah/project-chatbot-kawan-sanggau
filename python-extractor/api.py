"""
API HTTP untuk Kawan Sanggau.

Node.js (WhatsApp) memanggil layanan ini. Seluruh proses retrieval dan
embedding tetap berada di retriever.py; modul ini hanya membungkusnya
menjadi HTTP dan meneruskan generation ke NVIDIA NIM.

Jalankan:
    uvicorn api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
EXPECTED_VECTOR_SIZE = 2048

STATUS_BY_CODE = {
    "config": 503,
    "auth": 502,
    "model_not_found": 502,
    "bad_request": 502,
    "rate_limit": 429,
    "upstream_error": 502,
    "connection": 502,
    "timeout": 504,
    "empty_response": 502,
    "retrieval_error": 503,
    "context_empty": 502,
    "empty_question": 400,
}

app = FastAPI(
    title="Kawan Sanggau RAG API",
    description=(
        "Retrieval Qdrant (NVIDIA embedding) + generation NVIDIA NIM "
        "untuk chatbot WhatsApp Kawan Sanggau."
    ),
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., description="Pertanyaan pengguna")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Jumlah chunk yang diambil dari Qdrant",
    )


class GeneralRequest(BaseModel):
    question: str = Field(..., description="Pertanyaan umum")


def _error(code: str, message: str, *, retry_after=None):
    return {
        "code": code,
        "message": message,
        "retry_after": retry_after,
    }


def _qdrant_health() -> dict:
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:  # pragma: no cover - dependency hilang
        return {"ok": False, "message": f"qdrant-client tidak tersedia: {exc}"}

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv(
        "NVIDIA_QDRANT_COLLECTION", "kcda_knowledge_nvidia_v4"
    )

    try:
        client = QdrantClient(url=url, timeout=5)
        if not client.collection_exists(collection):
            return {
                "ok": False,
                "collection": collection,
                "message": f"Collection {collection} tidak ditemukan.",
            }

        info = client.get_collection(collection)
        size = info.config.params.vectors.size
        count = client.count(collection, exact=True).count

        return {
            "ok": size == EXPECTED_VECTOR_SIZE,
            "collection": collection,
            "vector_size": size,
            "points": count,
            "message": (
                None
                if size == EXPECTED_VECTOR_SIZE
                else f"Dimensi {size} tidak sama dengan {EXPECTED_VECTOR_SIZE}."
            ),
        }
    except Exception as exc:
        return {"ok": False, "message": f"Qdrant tidak dapat dihubungi: {exc}"}


def _nim_health() -> dict:
    try:
        import nim
    except Exception as exc:
        return {"ok": False, "message": f"Modul nim gagal dimuat: {exc}"}

    try:
        info = nim.describe_config()
    except nim.NimError as exc:
        return {"ok": False, "message": str(exc), "code": exc.code}

    return {"ok": True, **info}


@app.get("/health")
def health():
    nim_status = _nim_health()
    qdrant_status = _qdrant_health()
    healthy = bool(nim_status.get("ok")) and bool(qdrant_status.get("ok"))

    payload = {
        "status": "ok" if healthy else "degraded",
        "checks": {
            "generation": nim_status,
            "retrieval": qdrant_status,
        },
    }

    return JSONResponse(status_code=200 if healthy else 503, content=payload)


@app.post("/ask")
def ask(request: AskRequest):
    question = str(request.question or "").strip()

    if not question:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": _error("empty_question", "Pertanyaan kosong.")},
        )

    # Import lambat agar API tetap hidup dan bisa melaporkan error
    # konfigurasi dengan rapi saat .env belum lengkap.
    try:
        from rag import ask_rag
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": _error(
                    "config",
                    f"RAG Python tidak dapat dimuat: {exc}",
                ),
            },
        )

    try:
        result = ask_rag(
            question=question,
            top_k=request.top_k or DEFAULT_TOP_K,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "error": _error("upstream_error", f"Kesalahan tak terduga: {exc}"),
            },
        )

    if result.get("error"):
        error = result["error"]
        status = STATUS_BY_CODE.get(error.get("code"), 502)
        return JSONResponse(
            status_code=status,
            content={"ok": False, "question": question, "error": error},
        )

    return {
        "ok": True,
        "question": result["question"],
        "answer": result["answer"],
        "in_scope": result["in_scope"],
        "top_score": result.get("top_score"),
        "pages": result.get("pages", []),
        "sources": result.get("sources", []),
        "model": result.get("model"),
        "latency_ms": result.get("latency_ms"),
    }


GENERAL_SYSTEM_PROMPT = (
    "Kamu adalah admin digital Kawan Sanggau, asisten ramah untuk layanan "
    "BPS Kabupaten Sanggau. Jawab dalam bahasa Indonesia yang jelas dan "
    "ringkas, maksimal 40 kata. Jangan mengarang angka atau data statistik. "
    "Jika pengguna menanyakan angka, arahkan untuk menanyakan data kecamatan "
    "yang tersedia di dokumen."
)


@app.post("/ask/general")
def ask_general(request: GeneralRequest):
    question = str(request.question or "").strip()

    if not question:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": _error("empty_question", "Pertanyaan kosong.")},
        )

    try:
        import nim
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": _error("config", f"Modul nim gagal dimuat: {exc}"),
            },
        )

    try:
        generated = nim.chat(
            [
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        )
    except nim.NimError as exc:
        return JSONResponse(
            status_code=exc.http_status,
            content={"ok": False, "error": exc.to_dict()},
        )

    return {
        "ok": True,
        "question": question,
        "answer": generated["content"],
        "model": generated["model"],
        "finish_reason": generated["finish_reason"],
        "latency_ms": generated["latency_ms"],
    }
