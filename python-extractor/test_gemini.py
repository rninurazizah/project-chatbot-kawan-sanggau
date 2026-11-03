
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Membaca konfigurasi .env
env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env, override=True)

api_key = os.getenv("NVIDIA_API_KEY", "").strip()
model = os.getenv("NVIDIA_CHAT_MODEL", "").strip()

if not api_key:
    raise ValueError("NVIDIA_API_KEY tidak ditemukan.")

if not model:
    raise ValueError("NVIDIA_CHAT_MODEL tidak ditemukan.")

# Koneksi NVIDIA NIM
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
    timeout=120.0,
    max_retries=2
)

print("Model:", model)
print("Menghubungi NVIDIA NIM...", flush=True)

try:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Kamu adalah asisten statistik BPS."
            },
            {
                "role": "user",
                "content": "Jelaskan secara singkat apa itu Badan Pusat Statistik."
            }
        ],
        temperature=0.2,
        top_p=0.95,
        max_tokens=500,
        stream=False
    )

    print("\nBERHASIL")
    print(completion.choices[0].message.content)

except Exception as e:
    print("\nGAGAL")
    print(type(e).__name__)
    print(e)
