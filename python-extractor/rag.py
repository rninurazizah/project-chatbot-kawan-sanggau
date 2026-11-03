
import os
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from retriever import retrieve, is_definition_question


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=True)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()

NVIDIA_CHAT_MODEL = os.getenv(
    "NVIDIA_CHAT_MODEL",
    "nvidia/nemotron-3.5-super-vl-120b-a12b"
).strip()

TOP_K = int(os.getenv("RAG_TOP_K", "5"))


# =========================================================
# 2. VALIDATE CONFIGURATION
# =========================================================

if not ENV_FILE.exists():
    raise FileNotFoundError(
        f"File .env tidak ditemukan: {ENV_FILE}"
    )

if not NVIDIA_API_KEY:
    raise ValueError(
        "NVIDIA_API_KEY tidak ditemukan atau kosong. "
        "Periksa file .env."
    )

if not NVIDIA_CHAT_MODEL:
    raise ValueError(
        "NVIDIA_CHAT_MODEL tidak boleh kosong."
    )

print("========================================")
print("KAWAN SANGGAU - RAG CONFIGURATION")
print("========================================")
print(f"ENV FILE       : {ENV_FILE}")
print(f"API KEY LOADED : {bool(NVIDIA_API_KEY)}")
print(f"NVIDIA MODEL   : {NVIDIA_CHAT_MODEL}")
print(f"TOP K          : {TOP_K}")
print("========================================")


# Inisialisasi NVIDIA NIM
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
    timeout=120.0,
    max_retries=2
)


# =========================================================
# 3. BUILD CONTEXT
# =========================================================

def build_context(retrieved_documents):
    """
    Mengubah hasil retrieval Qdrant menjadi context
    untuk diberikan kepada NVIDIA NIM.
    """

    context_parts = []
    sources = []

    for index, point in enumerate(
        retrieved_documents,
        start=1
    ):

        payload = point.payload or {}

        text = str(payload.get("text", "")).strip()

        if not text:
            continue

        document = payload.get(
            "document",
            "Dokumen tidak diketahui"
        )

        title = (
            payload.get("title")
            or payload.get("caption")
            or "Tanpa judul"
        )

        content_type = payload.get(
            "type",
            "unknown"
        )

        page_start = payload.get("page_start")
        page_end = payload.get("page_end")

        # Format halaman
        if page_start is not None and page_end is not None:

            if page_start == page_end:
                page_info = f"Halaman {page_start}"
            else:
                page_info = (
                    f"Halaman {page_start}-{page_end}"
                )

        elif page_start is not None:
            page_info = f"Halaman {page_start}"

        else:
            page_info = "Halaman tidak diketahui"

        # Context untuk NVIDIA NIM
        context_parts.append(
            f"""
[Sumber {index}]
Dokumen: {document}
Judul: {title}
Jenis konten: {content_type}
Halaman: {page_info}

Isi:
{text}
"""
        )

        sources.append({
            "document": document,
            "title": title,
            "type": content_type,
            "page_start": page_start,
            "page_end": page_end,
            "score": getattr(point, "score", None)
        })

    context = "\n".join(context_parts)

    return context, sources


# =========================================================
# 4. GENERATE ANSWER WITH NVIDIA NIM
# =========================================================

def generate_answer(question, context):

    prompt = f"""
Kamu adalah asisten BPS Kawan Sanggau.

Tugasmu adalah menjawab pertanyaan pengguna
berdasarkan informasi dari dokumen BPS yang diberikan.

Aturan:
1. Gunakan konteks sebagai sumber utama jawaban.
2. Jangan mengarang atau menambahkan data yang tidak tersedia.
3. Jika informasi tidak ditemukan dalam konteks,
   sampaikan dengan jujur.
4. Gunakan bahasa Indonesia yang jelas dan mudah dipahami.
5. Jawab secara langsung dan tidak bertele-tele.
6. Jika memungkinkan, sertakan informasi halaman atau dokumen sumber.

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
"""

    print(
        "[INFO] Mengirim permintaan ke NVIDIA NIM...",
        flush=True
    )

    start_time = time.time()

    
    response = client.chat.completions.create(
        model=NVIDIA_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Kamu adalah asisten statistik "
                    "BPS Kawan Sanggau. Jawab berdasarkan "
                    "konteks dokumen secara akurat dan ringkas."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1200,
        stream=False,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )

    elapsed = time.time() - start_time

    print(
        f"[INFO] Waktu generation NVIDIA NIM: "
        f"{elapsed:.2f} detik",
        flush=True
    )
    print(
        "[INFO] Finish reason:",
        response.choices[0].finish_reason
    )

    answer = response.choices[0].message.content

    if answer is None or not answer.strip():
        print("[ERROR] NVIDIA NIM tidak menghasilkan konten jawaban.")
        return None

    return answer.strip()


# =========================================================
# 5. MAIN RAG FUNCTION
# =========================================================

def ask_rag(question, top_k=5):
    """
    Menjalankan seluruh proses RAG:
    1. Retrieval dokumen dari Qdrant.
    2. Penyusunan context.
    3. Generation jawaban menggunakan NVIDIA NIM.
    4. Pengembalian jawaban, pertanyaan, dan sumber.
    """

    try:
        print("\n[1] Melakukan retrieval dari Qdrant...")

        retrieved_documents = retrieve(
            question=question,
            top_k=top_k,
            content_type=(
                "glossary"
                if is_definition_question(question)
                else None
            )
        )

        print(
            f"[INFO] Jumlah chunk ditemukan: "
            f"{len(retrieved_documents)}"
        )

        if not retrieved_documents:
            return {
                "question": question,
                "answer": (
                    "Maaf, informasi yang kamu tanyakan "
                    "tidak ditemukan dalam dokumen."
                ),
                "sources": []
            }

        print("[2] Menyusun context...")

        context, sources = build_context(
            retrieved_documents
        )

        if not context or not context.strip():
            return {
                "question": question,
                "answer": (
                    "Maaf, konteks informasi tidak dapat "
                    "disusun dari dokumen."
                ),
                "sources": sources
            }

        print("[3] Menghasilkan jawaban dengan NVIDIA NIM...")

        answer = generate_answer(
            question=question,
            context=context
        )

        if answer is None or not answer.strip():
            answer = "Maaf, saya belum dapat menghasilkan jawaban."

        print("[4] Generation selesai.")

        return {
            "question": question,
            "answer": answer.strip(),
            "sources": sources
        }

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan pada ask_rag(): {e}")

        return {
            "question": question,
            "answer": (
                "Maaf, terjadi kesalahan saat memproses "
                "pertanyaan."
            ),
            "sources": []
        }

# =========================================================
# 6. DISPLAY RESULT
# =========================================================

def display_result(result):

    print("\n" + "=" * 60)
    print("PERTANYAAN")
    print("=" * 60)

    print(result["question"])

    print("\n" + "=" * 60)
    print("JAWABAN KAWAN SANGGAU")
    print("=" * 60)

    print(result["answer"])

    print("\n" + "=" * 60)
    print("SUMBER RETRIEVAL")
    print("=" * 60)

    if not result["sources"]:
        print("Tidak ada sumber ditemukan.")
        return

    for index, source in enumerate(
        result["sources"],
        start=1
    ):

        print(f"\nSumber {index}")
        print(f"Dokumen : {source['document']}")
        print(f"Judul   : {source['title']}")
        print(f"Jenis   : {source['type']}")

        page_start = source["page_start"]
        page_end = source["page_end"]

        if page_start is not None and page_end is not None:

            if page_start == page_end:
                print(f"Halaman : {page_start}")
            else:
                print(
                    f"Halaman : {page_start}-{page_end}"
                )

        elif page_start is not None:
            print(f"Halaman : {page_start}")

        else:
            print("Halaman : Tidak diketahui")

        print(f"Score   : {source['score']}")


# =========================================================
# 7. COMMAND LINE
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Kawan Sanggau RAG"
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Pertanyaan pengguna"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Jumlah chunk yang diambil dari Qdrant"
    )

    args = parser.parse_args()

    try:

        result = ask_rag(
            question=args.question,
            top_k=args.top_k
        )

        display_result(result)

    except Exception as e:

        print("\n[ERROR]")
        print(f"{type(e).__name__}: {e}")
