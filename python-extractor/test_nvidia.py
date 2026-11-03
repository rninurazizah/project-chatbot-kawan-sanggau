
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Membaca .env dari folder utama proyek
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

api_key = os.getenv("NVIDIA_API_KEY")
model = os.getenv(
    "NVIDIA_EMBEDDING_MODEL",
    "nvidia/nemotron-3-embed-1b"
)

if not api_key:
    raise ValueError("NVIDIA_API_KEY belum ditemukan di .env")

url = "https://integrate.api.nvidia.com/v1/embeddings"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

payload = {
    "model": model,
    "input": [
        "Kecamatan Kapuas Dalam memiliki data statistik penduduk.",
        "Jumlah penduduk berdasarkan kelompok umur."
    ],
    "input_type": "passage",
    "encoding_format": "float",
    "truncate": "NONE"
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=120
)

response.raise_for_status()
result = response.json()

print("Model:", result["model"])
print("Jumlah embedding:", len(result["data"]))

for item in result["data"]:
    vector = item["embedding"]
    print(
        f"Index: {item['index']} | "
        f"Dimensi: {len(vector)} | "
        f"Contoh nilai: {vector[:5]}"
    )
