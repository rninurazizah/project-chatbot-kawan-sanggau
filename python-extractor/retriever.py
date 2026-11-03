
import os
import re
import argparse
import requests

from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


# ==================================================
# KONFIGURASI
# ==================================================

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.getenv("NVIDIA_API_KEY")

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

COLLECTION = os.getenv(
    "NVIDIA_QDRANT_COLLECTION",
    "kcda_knowledge_nvidia_v3"
)

MODEL = os.getenv(
    "NVIDIA_EMBEDDING_MODEL",
    "nvidia/nemotron-3-embed-1b"
)

# Ambang minimal skor kemiripan. Hasil di bawah ambang dianggap tidak
# relevan sehingga model tidak dipaksa menjawab dari konteks yang salah.
SCORE_THRESHOLD = float(
    os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.35")
)

# Pertanyaan definisi/glosarium punya skor lebih rendah karena teksnya
# berupa daftar istilah, bukan kalimat. Ambangnya dipisah.
GLOSSARY_SCORE_THRESHOLD = float(
    os.getenv("RETRIEVAL_GLOSSARY_THRESHOLD", "0.25")
)

DEFINITION_RE = re.compile(
    r"\b(apa\s+itu|apa\s+arti|arti|definisi|pengertian|singkatan|"
    r"maksud|istilah|glosarium|penjelasan)\b",
    re.IGNORECASE,
)


def is_definition_question(question: str) -> bool:
    return bool(DEFINITION_RE.search(question or ""))


EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"

VECTOR_SIZE = 2048

if not API_KEY:
    raise ValueError("NVIDIA_API_KEY tidak ditemukan di .env")


# ==================================================
# INISIALISASI CLIENT
# ==================================================

qdrant = QdrantClient(url=QDRANT_URL)

session = requests.Session()

session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json"
})


# ==================================================
# EMBEDDING PERTANYAAN
# ==================================================

def generate_query_embedding(question):

    payload = {
        "model": MODEL,
        "input": [question],
        "input_type": "query",
        "encoding_format": "float",
        "truncate": "NONE"
    }

    response = session.post(
        EMBEDDING_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    data = result.get("data", [])

    if len(data) != 1:
        raise ValueError(
            "NVIDIA tidak mengembalikan satu embedding."
        )

    vector = data[0]["embedding"]

    if len(vector) != VECTOR_SIZE:
        raise ValueError(
            f"Dimensi query tidak sesuai: {len(vector)}"
        )

    return vector


# ==================================================
# RETRIEVAL QDRANT
# ==================================================

def retrieve(question, top_k=5, score_threshold=None, content_type=None):

    if not qdrant.collection_exists(COLLECTION):
        raise ValueError(
            f"Collection {COLLECTION} tidak ditemukan."
        )

    info = qdrant.get_collection(COLLECTION)

    if info.config.params.vectors.size != VECTOR_SIZE:
        raise ValueError(
            "Dimensi vektor Qdrant tidak sesuai."
        )

    query_vector = generate_query_embedding(question)

    query_filter = None

    if content_type:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=content_type)
                )
            ]
        )

    result = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False
    )

    if score_threshold is not None:
        threshold = float(score_threshold)
    elif content_type == "glossary":
        threshold = GLOSSARY_SCORE_THRESHOLD
    else:
        threshold = SCORE_THRESHOLD

    return [
        point
        for point in result.points
        if point.score >= threshold
    ]


# ==================================================
# MENAMPILKAN HASIL
# ==================================================

def display_results(question, results):

    print("\n" + "=" * 70)
    print("PERTANYAAN:", question)
    print("=" * 70)

    if not results:
        print("Tidak ada dokumen yang ditemukan.")
        return

    for index, point in enumerate(results, start=1):

        payload = point.payload or {}

        print(f"\nHASIL KE-{index}")
        print("-" * 70)

        page_start = payload.get("page_start")
        page_end = payload.get("page_end")
        pages = (
            f"{page_start}-{page_end}"
            if page_start != page_end
            else str(page_start)
        )

        print(f"Score       : {point.score:.4f}")
        print(f"Chunk ID    : {payload.get('id')}")
        print(f"Document    : {payload.get('document')}")
        print(f"Content     : {payload.get('type')}")
        print(f"Title       : {payload.get('title') or payload.get('caption')}")
        print(f"Pages       : {pages}")

        print("\nTEXT:")
        print(payload.get("text", ""))

    print("\n" + "=" * 70)


# ==================================================
# MAIN
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description="Retrieval dokumen BPS menggunakan NVIDIA dan Qdrant"
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Pertanyaan yang ingin dicari"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Jumlah hasil retrieval"
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k harus minimal 1")

    results = retrieve(
        question=args.question,
        top_k=args.top_k
    )

    display_results(
        question=args.question,
        results=results
    )


if __name__ == "__main__":
    main()
