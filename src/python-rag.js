import 'dotenv/config';

/**
 * Klien HTTP ke API Python (python-extractor/api.py).
 *
 * Node.js tidak melakukan embedding maupun retrieval sendiri. Seluruh
 * proses RAG (embedding NVIDIA, Qdrant, generation NVIDIA NIM) berada
 * di Python agar hanya ada satu pipeline.
 */

const BASE_URL = (process.env.PYTHON_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
const TIMEOUT_MS = Number.parseInt(process.env.PYTHON_API_TIMEOUT_MS || '60000', 10);

export class PythonApiError extends Error {
  constructor(message, { status = 0, code = 'python_api_error', retryAfter = null } = {}) {
    super(message);
    this.name = 'PythonApiError';
    this.status = status;
    this.code = code;
    this.retryAfter = retryAfter;
  }
}

async function request(path, { method = 'GET', body, timeoutMs = TIMEOUT_MS } = {}) {
  let response;

  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    if (error?.name === 'TimeoutError' || error?.name === 'AbortError') {
      throw new PythonApiError(
        `API Python tidak merespons dalam ${timeoutMs} ms (${BASE_URL}${path}).`,
        { status: 504, code: 'timeout' }
      );
    }

    throw new PythonApiError(
      `Tidak dapat menghubungi API Python di ${BASE_URL}. Pastikan server berjalan.`,
      { status: 503, code: 'connection' }
    );
  }

  const raw = await response.text();
  let data = null;

  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch {
      data = null;
    }
  }

  if (!response.ok || data?.ok === false) {
    const detail = data?.error ?? {};
    throw new PythonApiError(
      detail.message || `API Python mengembalikan HTTP ${response.status}.`,
      {
        status: response.status,
        code: detail.code || 'http_error',
        retryAfter: detail.retry_after ?? null,
      }
    );
  }

  if (!data) {
    throw new PythonApiError('API Python mengembalikan respons kosong.', {
      status: 502,
      code: 'empty_response',
    });
  }

  return data;
}

function toLegacyShape(data) {
  return {
    answer: data.answer,
    pages: Array.isArray(data.pages) ? data.pages : [],
    sources: [...new Set((data.sources ?? []).map((item) => item.document).filter(Boolean))],
    topScore: data.top_score ?? 0,
    inScope: Boolean(data.in_scope),
    model: data.model ?? null,
    latencyMs: data.latency_ms ?? null,
    rawSources: data.sources ?? [],
  };
}

/**
 * Pertanyaan statistik: retrieval Qdrant + generation NVIDIA NIM di Python.
 * Mengembalikan bentuk yang sama dengan rag.js lama agar bot.js tidak berubah.
 */
export async function answerQuestion(question, { topK } = {}) {
  const body = { question };

  if (Number.isInteger(topK) && topK > 0) {
    body.top_k = topK;
  }

  return toLegacyShape(await request('/ask', { method: 'POST', body }));
}

/**
 * Pertanyaan umum (sapaan/administratif) tanpa retrieval dokumen.
 */
export async function runGeneralAnswer(question) {
  const data = await request('/ask/general', {
    method: 'POST',
    body: { question },
  });

  return data.answer;
}

/**
 * Untuk pemeriksaan konfigurasi saat bot dinyalakan.
 */
export async function checkPythonApi() {
  return request('/health', { timeoutMs: 5000 });
}

export const pythonApiUrl = BASE_URL;
