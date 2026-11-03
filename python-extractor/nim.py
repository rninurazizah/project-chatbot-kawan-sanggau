"""
Klien generation NVIDIA NIM untuk Kawan Sanggau.

Modul ini hanya menangani generation (chat completions). Embedding dan
retrieval tetap berada di modul masing-masing dan tidak diubah.

Konfigurasi dibaca dari .env:
    NVIDIA_API_KEY
    NVIDIA_MODEL            (alias lama: NVIDIA_CHAT_MODEL)
    NVIDIA_BASE_URL         (default https://integrate.api.nvidia.com/v1)
    NVIDIA_TIMEOUT_SECONDS  (default 60)
    NVIDIA_MAX_RETRIES      (default 3; jumlah percobaan ulang)
    NVIDIA_TEMPERATURE      (default 0.2)
    NVIDIA_MAX_TOKENS       (default 1200)
    NVIDIA_DISABLE_THINKING (default true)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


# =========================================================
# ERROR TERSTRUKTUR
# =========================================================

class NimError(RuntimeError):
    """Kesalahan generation yang bisa dipetakan ke HTTP oleh api.py."""

    code = "nim_error"
    http_status = 502

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "retry_after": self.retry_after,
        }


class NimConfigError(NimError):
    code = "config"
    http_status = 503


class NimAuthError(NimError):
    code = "auth"
    http_status = 502


class NimModelError(NimError):
    code = "model_not_found"
    http_status = 502


class NimBadRequest(NimError):
    code = "bad_request"
    http_status = 502


class NimRateLimit(NimError):
    code = "rate_limit"
    http_status = 429


class NimUpstreamError(NimError):
    code = "upstream_error"
    http_status = 502


class NimConnectionError(NimError):
    code = "connection"
    http_status = 502


class NimTimeout(NimError):
    code = "timeout"
    http_status = 504


class NimEmptyResponse(NimError):
    code = "empty_response"
    http_status = 502


# =========================================================
# KONFIGURASI
# =========================================================

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "ya", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default
    return value


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default
    return value


@dataclass(frozen=True)
class NimConfig:
    api_key: str
    model: str
    base_url: str
    timeout: float
    max_retries: int
    temperature: float
    max_tokens: int
    disable_thinking: bool


def load_config() -> NimConfig:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    model = (
        os.getenv("NVIDIA_MODEL", "").strip()
        or os.getenv("NVIDIA_CHAT_MODEL", "").strip()
        or DEFAULT_MODEL
    )
    base_url = (
        os.getenv("NVIDIA_BASE_URL", "").strip()
        or DEFAULT_BASE_URL
    ).rstrip("/")

    if not api_key:
        raise NimConfigError(
            "NVIDIA_API_KEY tidak ditemukan atau kosong. Periksa file .env."
        )

    if not model:
        raise NimConfigError("NVIDIA_MODEL tidak boleh kosong.")

    return NimConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=_env_float("NVIDIA_TIMEOUT_SECONDS", 60.0),
        max_retries=max(0, _env_int("NVIDIA_MAX_RETRIES", 3)),
        temperature=_env_float("NVIDIA_TEMPERATURE", 0.2),
        max_tokens=max(1, _env_int("NVIDIA_MAX_TOKENS", 1200)),
        disable_thinking=_env_bool("NVIDIA_DISABLE_THINKING", True),
    )


def describe_config() -> dict[str, Any]:
    """Ringkasan konfigurasi tanpa pernah menampilkan API key."""
    cfg = load_config()
    return {
        "api_key_loaded": bool(cfg.api_key),
        "model": cfg.model,
        "base_url": cfg.base_url,
        "timeout_seconds": cfg.timeout,
        "max_retries": cfg.max_retries,
        "disable_thinking": cfg.disable_thinking,
    }


@lru_cache(maxsize=4)
def _client_for(base_url: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _extract_content(message: Any) -> str:
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()

    return ""


# =========================================================
# GENERATION
# =========================================================

def chat(
    messages: list[dict[str, Any]],
    *,
    config: NimConfig | None = None,
) -> dict[str, Any]:
    """
    Memanggil NVIDIA NIM dan mengembalikan dict:
        content, model, finish_reason, latency_ms, usage
    Semua kegagalan dilempar sebagai NimError, tidak pernah
    dikembalikan sebagai jawaban palsu.
    """
    cfg = config or load_config()
    client = _client_for(cfg.base_url, cfg.api_key, cfg.timeout)

    extra_body = None
    if cfg.disable_thinking:
        # Wajib untuk nemotron-3.5-super-vl: tanpa ini konten bisa kosong.
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    allow_without_extra = extra_body is not None
    attempt = 0
    started = time.time()

    while True:
        attempt += 1

        try:
            response = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                stream=False,
                **({"extra_body": extra_body} if extra_body else {}),
            )
        except openai.BadRequestError as error:
            # Sebagian model menolak chat_template_kwargs. Coba sekali
            # tanpa parameter itu agar tetap bisa dipakai.
            if allow_without_extra:
                allow_without_extra = False
                extra_body = None
                continue
            raise NimBadRequest(
                f"Permintaan ditolak NVIDIA NIM (400): {_short(error)}"
            ) from error
        except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
            raise NimAuthError(
                "NVIDIA menolak API key (401/403). Periksa NVIDIA_API_KEY."
            ) from error
        except openai.NotFoundError as error:
            raise NimModelError(
                f"Model '{cfg.model}' tidak ditemukan pada endpoint "
                f"{cfg.base_url} (404)."
            ) from error
        except openai.RateLimitError as error:
            wait = _retry_after_seconds(error)
            if attempt > cfg.max_retries:
                raise NimRateLimit(
                    "NVIDIA NIM sedang membatasi permintaan (429). "
                    "Coba lagi beberapa saat.",
                    retry_after=wait,
                ) from error
            _sleep(wait, attempt)
            continue
        except openai.APITimeoutError as error:
            if attempt > cfg.max_retries:
                raise NimTimeout(
                    f"Permintaan ke NVIDIA NIM melewati batas waktu "
                    f"{cfg.timeout:.1f} detik."
                ) from error
            _sleep(None, attempt)
            continue
        except openai.APIConnectionError as error:
            if attempt > cfg.max_retries:
                raise NimConnectionError(
                    f"Tidak dapat terhubung ke NVIDIA NIM: {_short(error)}"
                ) from error
            _sleep(None, attempt)
            continue
        except openai.APIStatusError as error:
            status = getattr(error, "status_code", None)
            if status is not None and status >= 500:
                if attempt > cfg.max_retries:
                    raise NimUpstreamError(
                        f"NVIDIA NIM mengembalikan HTTP {status}."
                    ) from error
                _sleep(None, attempt)
                continue
            raise NimUpstreamError(
                f"NVIDIA NIM mengembalikan HTTP {status}: {_short(error)}"
            ) from error
        except openai.OpenAIError as error:
            raise NimUpstreamError(
                f"Kesalahan klien NVIDIA NIM: {_short(error)}"
            ) from error

        latency_ms = int((time.time() - started) * 1000)
        choice = response.choices[0] if response.choices else None
        finish_reason = getattr(choice, "finish_reason", None)
        content = _extract_content(choice.message if choice else None)

        if not content:
            raise NimEmptyResponse(
                "NVIDIA NIM tidak menghasilkan jawaban "
                f"(finish_reason={finish_reason})."
            )

        usage = getattr(response, "usage", None)

        return {
            "content": content,
            "model": getattr(response, "model", None) or cfg.model,
            "finish_reason": finish_reason,
            "latency_ms": latency_ms,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
        }


def _sleep(retry_after: float | None, attempt: int) -> None:
    wait = retry_after if retry_after is not None else min(30.0, 2.0 ** attempt)
    time.sleep(wait)


def _short(error: Exception, limit: int = 200) -> str:
    text = str(error).replace("\n", " ").strip()
    return text[:limit]
