
import json
import os
import uuid
import time
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
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Gunakan collection baru khusus NVIDIA
COLLECTION = os.getenv(
    "NVIDIA_QDRANT_COLLECTION",
    "kcda_knowledge_nvidia_v3"
)

MODEL = os.getenv(
    "NVIDIA_EMBEDDING_MODEL",
    "nvidia/nemotron-3-embed-1b"
)

EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"

VECTOR_SIZE = 2048
BATCH_SIZE = 10

# Jeda antarbatch
BATCH_DELAY = 2

# Pengaturan retry
MAX_RETRIES = 5
RETRY_DELAY = 20


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
# MEMBACA CHUNKS
# ==================================================

def read_chunks(file_path):
    chunks = []
    seen_ids = set()

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            item = json.loads(line)

            # Gunakan id sebagai chunk_id
            chunk_id = item.get("chunk_id") or item.get("id")

            if not chunk_id:
                raise ValueError(
                    f"ID chunk tidak ditemukan pada baris {line_number}"
                )

            # Pastikan teks tersedia
            text = str(item.get("text", "")).strip()

            if not text:
                print(
                    f"Chunk kosong dilewati: baris {line_number}"
                )
                continue

            # Periksa ID duplikat
            if chunk_id in seen_ids:
                raise ValueError(
                    f"ID duplikat ditemukan: {chunk_id}"
                )

            seen_ids.add(chunk_id)

            # Samakan format ID
            item["chunk_id"] = str(chunk_id)

            chunks.append(item)

    return chunks
# ==================================================
# MEMBUAT EMBEDDING NVIDIA
# ==================================================

def generate_embeddings(texts):

    payload = {
        "model": MODEL,
        "input": texts,
        "input_type": "passage",
        "encoding_format": "float",
        "truncate": "NONE"
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = session.post(
                EMBEDDING_URL,
                json=payload,
                timeout=120
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                try:
                    wait_time = (
                        float(retry_after)
                        if retry_after else RETRY_DELAY * attempt
                    )
                except ValueError:
                    wait_time = RETRY_DELAY * attempt

                print(
                    f"Rate limit NVIDIA. "
                    f"Menunggu {wait_time:.1f} detik..."
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

            result = response.json()

            data = result.get("data", [])

            # Pastikan urutan embedding sesuai input
            data.sort(key=lambda item: item["index"])

            vectors = [
                item["embedding"]
                for item in data
            ]

            if len(vectors) != len(texts):
                raise ValueError(
                    "Jumlah embedding tidak sama dengan jumlah teks"
                )

            for vector in vectors:
                if len(vector) != VECTOR_SIZE:
                    raise ValueError(
                        f"Dimensi embedding tidak valid: {len(vector)}"
                    )

            return vectors

        except requests.exceptions.HTTPError as error:

            status = (
                error.response.status_code
                if error.response is not None else None
            )

            # Error selain rate limit tidak perlu diulang
            if status != 429:
                raise

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError
        ) as error:

            print(f"Koneksi bermasalah: {error}")

        if attempt < MAX_RETRIES:
            wait_time = RETRY_DELAY * attempt

            print(
                f"Percobaan {attempt}/{MAX_RETRIES} gagal. "
                f"Retry dalam {wait_time} detik..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "Gagal mendapatkan embedding setelah beberapa percobaan."
    )


# ==================================================
# MEMBUAT COLLECTION QDRANT
# ==================================================

def create_collection():

    if not qdrant.collection_exists(COLLECTION):

        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

        print(f"Collection dibuat: {COLLECTION}")

    else:

        info = qdrant.get_collection(COLLECTION)

        print(f"Collection sudah tersedia: {COLLECTION}")

        if info.config.params.vectors.size != VECTOR_SIZE:
            raise ValueError(
                "Dimensi collection tidak sesuai dengan model NVIDIA."
            )


# ==================================================
# MENGAMBIL CHUNK YANG SUDAH TERSIMPAN
# ==================================================

def get_existing_chunk_ids():

    existing_ids = set()

    offset = None

    while True:

        points, next_offset = qdrant.scroll(
            collection_name=COLLECTION,
            limit=500,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False
        )

        for point in points:

            chunk_id = point.payload.get("chunk_id")

            if chunk_id:
                existing_ids.add(chunk_id)

        if next_offset is None:
            break

        offset = next_offset

    return existing_ids


# ==================================================
# MEMPROSES DAN MENYIMPAN BATCH
# ==================================================

def process_chunks(chunks):

    existing_ids = get_existing_chunk_ids()

    print(f"Chunk sudah tersedia: {len(existing_ids)}")

    pending_chunks = [
        item for item in chunks
        if item["chunk_id"] not in existing_ids
    ]

    total = len(pending_chunks)

    print(f"Chunk belum di-embedding: {total}")

    if total == 0:
        print("Semua chunk sudah tersimpan. Tidak perlu embedding ulang.")
        return

    total_batches = (
        total + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for start in range(0, total, BATCH_SIZE):

        batch = pending_chunks[start:start + BATCH_SIZE]

        batch_number = start // BATCH_SIZE + 1

        texts = [
            item["text"]
            for item in batch
        ]

        print(
            f"\nEmbedding batch {batch_number}/{total_batches}"
        )

        vectors = generate_embeddings(texts)

        points = []

        for item, vector in zip(batch, vectors):

            chunk_id = item["chunk_id"]

            # UUID deterministik
            point_id = str(uuid.UUID(chunk_id))
            

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=dict(item)
                )
            )

        qdrant.upsert(
            collection_name=COLLECTION,
            points=points,
            wait=True
        )

        print(
            f"Berhasil disimpan: {len(points)} chunks"
        )

        if start + BATCH_SIZE < total:
            time.sleep(BATCH_DELAY)


# ==================================================
# MAIN
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Lokasi merged_chunks.jsonl"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    chunks = read_chunks(input_path)

    if not chunks:
        raise ValueError("Tidak ada chunks untuk diproses")

    print(f"Total chunks: {len(chunks)}")

    create_collection()

    process_chunks(chunks)

    print("\n=== EMBEDDING SELESAI ===")

    print(f"Collection: {COLLECTION}")

    print(
        "Jumlah points:",
        qdrant.count(
            collection_name=COLLECTION,
            exact=True
        ).count
    )


if __name__ == "__main__":
    main()
