import 'dotenv/config';
import { createHash } from 'node:crypto';
import { QdrantVectorStore } from '@langchain/qdrant';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { markFilesAsProcessed } from './extractor.js';

export const COLLECTION_NAME = process.env.QDRANT_COLLECTION || 'kcda_knowledge';
export const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333';
export const EMBEDDING_MODEL = process.env.GEMINI_EMBEDDING_MODEL || 'gemini-embedding-001';

const BATCH_SIZE = 32;

// Parameter opsional (tidak wajib ada di .env). Nilai 0 diperbolehkan.
function envInt(name, fallback) {
  const n = Number.parseInt(process.env[name] ?? '', 10);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}
const MAX_EMBED_ATTEMPTS = Math.max(1, envInt('EMBED_MAX_ATTEMPTS', 5));
const BASE_BACKOFF_MS = envInt('EMBED_BACKOFF_MS', 15000);
// Free tier gemini-embedding-001 dilaporkan 100 RPM; 32 teks per batch => jeda ~20 dtk.
// Jika akun Anda berbayar, set EMBED_DELAY_MS=0 di .env agar lebih cepat.
const DELAY_BETWEEN_BATCHES_MS = envInt('EMBED_DELAY_MS', 20000);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function createEmbeddings() {
  if (!process.env.GOOGLE_API_KEY) {
    throw new Error('GOOGLE_API_KEY belum diatur. Salin .env.example menjadi .env dan isi API key Gemini.');
  }
  return new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_API_KEY,
    modelName: EMBEDDING_MODEL,
  });
}

export function createVectorStore() {
  return new QdrantVectorStore(createEmbeddings(), {
    url: QDRANT_URL,
    collectionName: COLLECTION_NAME,
  });
}

/* ------------------------------------------------------------------ */
/* Helper                                                              */
/* ------------------------------------------------------------------ */

function isValidVector(v) {
  return Array.isArray(v) && v.length > 0 && v.every(Number.isFinite);
}

// ID deterministik: chunk yang sama => ID yang sama, sehingga upsert ulang menimpa
// (idempotent) dan tidak menggandakan data. Formatnya UUID yang valid untuk Qdrant
// (bukan UUIDv5 sungguhan, hanya format 8-4-4-4-12 dengan bit versi/varian).
function makePointId(fileName, indexInFile, text) {
  const hex = createHash('sha256')
    .update(`${fileName}\u0000${indexInFile}\u0000${text}`)
    .digest('hex');
  const variant = ((parseInt(hex[16], 16) & 0x3) | 0x8).toString(16);
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    `5${hex.slice(13, 16)}`,
    `${variant}${hex.slice(17, 20)}`,
    hex.slice(20, 32),
  ].join('-');
}

function buildItems(chunks) {
  const counters = new Map();
  const items = [];
  let skipped = 0;

  for (const doc of chunks) {
    const text = typeof doc?.pageContent === 'string' ? doc.pageContent : '';
    if (!text.trim()) {
      skipped += 1;
      continue;
    }
    const fileName = doc.metadata?.fileName ?? null;
    const key = fileName ?? '__tanpa_nama_file__';
    const indexInFile = counters.get(key) ?? 0;
    counters.set(key, indexInFile + 1);
    items.push({ doc, fileName, id: makePointId(key, indexInFile, text) });
  }

  if (skipped > 0) {
    console.warn(`[PERINGATAN] ${skipped} chunk kosong dilewati.`);
  }
  return items;
}

async function getExistingVectorSize(client) {
  const { collections } = await client.getCollections();
  if (!collections.some((c) => c.name === COLLECTION_NAME)) return null;
  const info = await client.getCollection(COLLECTION_NAME);
  const vectors = info?.config?.params?.vectors;
  return typeof vectors?.size === 'number' ? vectors.size : null;
}

/**
 * Membuat embedding untuk satu batch dan MEMVALIDASI hasilnya.
 *
 * Latar belakang: GoogleGenerativeAIEmbeddings.embedDocuments() (versi 2.3.4) memakai
 * Promise.allSettled dan mengisi `[]` untuk teks yang request-nya gagal, tanpa melempar
 * error. Di sini setiap vektor dicek; yang kosong diulang dengan backoff. Jika tetap
 * gagal, error dilempar SEBELUM ada data yang ditulis ke Qdrant.
 */
async function embedBatchStrict(embeddings, texts, label, expectedDim) {
  const vectors = new Array(texts.length).fill(null);
  let pending = texts.map((_, i) => i);

  for (let attempt = 1; attempt <= MAX_EMBED_ATTEMPTS && pending.length > 0; attempt += 1) {
    let result = [];
    try {
      result = await embeddings.embedDocuments(pending.map((i) => texts[i]));
    } catch (error) {
      console.warn(`[EMBEDDING] ${label}: embedDocuments melempar error: ${error.message}`);
    }

    const stillPending = [];
    pending.forEach((originalIndex, pos) => {
      const v = result?.[pos];
      if (!isValidVector(v)) {
        stillPending.push(originalIndex);
        return;
      }
      if (expectedDim && v.length !== expectedDim) {
        throw new Error(
          `${label}: dimensi vektor ${v.length} tidak sama dengan dimensi collection ${expectedDim}.`
        );
      }
      vectors[originalIndex] = v;
    });
    pending = stillPending;

    if (pending.length > 0 && attempt < MAX_EMBED_ATTEMPTS) {
      const wait = Math.min(BASE_BACKOFF_MS * 2 ** (attempt - 1), 120000);
      console.warn(
        `[EMBEDDING] ${label}: ${pending.length}/${texts.length} vektor kosong. ` +
          `Mencoba lagi dalam ${Math.round(wait / 1000)} dtk (percobaan ${attempt}/${MAX_EMBED_ATTEMPTS}).`
      );
      await sleep(wait);
    }
  }

  if (pending.length > 0) {
    // embedQuery() memakai embedContent yang MELEMPAR error asli (mis. 429), jadi
    // dipakai untuk menampilkan penyebab sebenarnya.
    let cause;
    try {
      await embeddings.embedQuery(texts[pending[0]]);
      cause = 'embedQuery tunggal berhasil (kegagalan tampaknya terkait laju/kuota request batch)';
    } catch (error) {
      cause = error.message;
    }
    throw new Error(
      `${label}: ${pending.length} dari ${texts.length} teks tetap tanpa vektor valid ` +
        `setelah ${MAX_EMBED_ATTEMPTS} percobaan. Penyebab dari Gemini: ${cause}`
    );
  }

  return vectors;
}

async function fileFilter(fileName) {
  return { must: [{ key: 'metadata.fileName', match: { value: fileName } }] };
}

// Hapus poin lama milik file yang sama yang ID-nya TIDAK ada di himpunan ID baru.
async function deleteStalePoints(client, fileName, keepIds) {
  const filter = await fileFilter(fileName);
  const stale = [];
  let offset;

  do {
    const page = await client.scroll(COLLECTION_NAME, {
      filter,
      limit: 256,
      offset,
      with_payload: false,
      with_vector: false,
    });
    for (const point of page.points) {
      if (!keepIds.has(String(point.id))) stale.push(point.id);
    }
    offset = page.next_page_offset ?? undefined;
  } while (offset !== undefined && offset !== null);

  for (let i = 0; i < stale.length; i += 500) {
    await client.delete(COLLECTION_NAME, { wait: true, points: stale.slice(i, i + 500) });
  }
  return stale.length;
}

/* ------------------------------------------------------------------ */
/* Pipeline utama                                                      */
/* ------------------------------------------------------------------ */

export async function processEmbeddings(newChunks) {
  if (!Array.isArray(newChunks) || newChunks.length === 0) {
    console.log('[INFO] Tidak ada PDF baru/berubah untuk di-embed.');
    return 0;
  }

  const items = buildItems(newChunks);
  if (items.length === 0) {
    console.log('[INFO] Semua chunk kosong, tidak ada yang di-embed.');
    return 0;
  }

  const embeddings = createEmbeddings();
  const vectorStore = new QdrantVectorStore(embeddings, {
    url: QDRANT_URL,
    collectionName: COLLECTION_NAME,
  });
  const client = vectorStore.client;
  const totalBatches = Math.ceil(items.length / BATCH_SIZE);

  // TAHAP 1: embedding SEMUA chunk di memori. Belum ada yang ditulis/dihapus di Qdrant.
  console.log(`[EMBEDDING] Membuat embedding Gemini untuk ${items.length} chunks...`);
  let expectedDim = await getExistingVectorSize(client);
  const vectors = [];

  for (let i = 0; i < items.length; i += BATCH_SIZE) {
    const batchNo = Math.floor(i / BATCH_SIZE) + 1;
    const label = `Batch ${batchNo}/${totalBatches}`;
    const texts = items.slice(i, i + BATCH_SIZE).map((it) => it.doc.pageContent);

    const batchVectors = await embedBatchStrict(embeddings, texts, label, expectedDim);
    expectedDim ??= batchVectors[0].length;
    vectors.push(...batchVectors);
    console.log(`[EMBEDDING] ${label} embedding valid (${batchVectors.length} vektor, dimensi ${expectedDim}).`);

    if (i + BATCH_SIZE < items.length && DELAY_BETWEEN_BATCHES_MS > 0) {
      await sleep(DELAY_BETWEEN_BATCHES_MS);
    }
  }

  // TAHAP 2: upsert ke Qdrant. ID deterministik => aman diulang jika gagal di tengah.
  for (let i = 0; i < items.length; i += BATCH_SIZE) {
    const batchNo = Math.floor(i / BATCH_SIZE) + 1;
    const slice = items.slice(i, i + BATCH_SIZE);
    const docs = slice.map((it) => ({
      pageContent: it.doc.pageContent,
      metadata: it.doc.metadata,
      id: it.id,
    }));

    await vectorStore.addVectors(vectors.slice(i, i + BATCH_SIZE), docs, {
      ids: slice.map((it) => it.id),
    });
    console.log(`[QDRANT] Batch ${batchNo}/${totalBatches} tersimpan.`);
  }

  // TAHAP 3: setelah SEMUA chunk baru tersimpan, hapus chunk lama yang sudah tidak dipakai,
  // lalu verifikasi jumlah poin per file sebelum menandai file selesai.
  const idsByFile = new Map();
  for (const it of items) {
    if (!it.fileName) continue;
    if (!idsByFile.has(it.fileName)) idsByFile.set(it.fileName, new Set());
    idsByFile.get(it.fileName).add(it.id);
  }

  for (const [fileName, keepIds] of idsByFile) {
    const removed = await deleteStalePoints(client, fileName, keepIds);
    console.log(`[INDEX] ${fileName}: ${removed} chunk lama dihapus setelah versi baru lengkap.`);

    const { count } = await client.count(COLLECTION_NAME, {
      filter: await fileFilter(fileName),
      exact: true,
    });
    if (count !== keepIds.size) {
      throw new Error(
        `Verifikasi gagal untuk ${fileName}: Qdrant berisi ${count} poin, seharusnya ${keepIds.size}. ` +
          'File TIDAK ditandai selesai.'
      );
    }
    console.log(`[INDEX] ${fileName}: verifikasi OK (${count} poin).`);
  }

  // Hash baru dicatat hanya setelah semua tahap di atas berhasil.
  await markFilesAsProcessed(newChunks);
  console.log('[SUKSES] Embedding dan indexing selesai.');
  return items.length;
}