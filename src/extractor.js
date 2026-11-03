import 'dotenv/config';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

const KNOWLEDGE_DIR = './knowledge';
const STORE_FILE = './data_store/processed_files.json';
const CHUNK_SIZE = 700;
const CHUNK_OVERLAP = 100;

async function getFileHash(filePath) {
  const buffer = await fs.readFile(filePath);
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

async function loadProcessedFiles() {
  try {
    return JSON.parse(await fs.readFile(STORE_FILE, 'utf-8'));
  } catch (error) {
    if (error.code === 'ENOENT') return {};
    console.warn(`[PERINGATAN] Catatan proses tidak bisa dibaca: ${error.message}.`);
    return {};
  }
}

async function saveProcessedFiles(data) {
  await fs.mkdir(path.dirname(STORE_FILE), { recursive: true });
  await fs.writeFile(STORE_FILE, JSON.stringify(data, null, 2), 'utf-8');
}

async function extractChunks(filePath, fileName, fileHash) {
  const loader = new PDFLoader(filePath);
  const pages = await loader.load();
  const nonEmptyPages = pages.filter((page) => page.pageContent?.trim());
  if (!nonEmptyPages.length) {
    throw new Error('Tidak ada teks yang berhasil diekstrak. PDF mungkin hasil scan dan memerlukan OCR.');
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
  });
  const chunks = await splitter.splitDocuments(nonEmptyPages);

  return chunks.map((chunk) => ({
    ...chunk,
    metadata: {
      ...chunk.metadata,
      fileName,
      fileHash,
      // PDFLoader biasanya menyimpan nomor halaman pada metadata.loc.pageNumber.
      pageNumber: chunk.metadata?.loc?.pageNumber ?? chunk.metadata?.pageNumber ?? chunk.metadata?.page,
    },
  }));
}

export async function extractNewPDFs() {
  await fs.mkdir(KNOWLEDGE_DIR, { recursive: true });
  const processedFiles = await loadProcessedFiles();
  const entries = await fs.readdir(KNOWLEDGE_DIR, { withFileTypes: true });
  const pdfFiles = entries.filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.pdf')).map((entry) => entry.name);

  if (!pdfFiles.length) {
    console.log(`[INFO] Tidak ada file PDF di ${KNOWLEDGE_DIR}. Masukkan PDF terlebih dahulu.`);
    return [];
  }

  const newChunks = [];
  for (const fileName of pdfFiles) {
    const filePath = path.join(KNOWLEDGE_DIR, fileName);
    try {
      const fileHash = await getFileHash(filePath);
      if (processedFiles[fileName] === fileHash) {
        console.log(`[SKIP] ${fileName} tidak berubah.`);
        continue;
      }
      console.log(`[EKSTRAKSI] ${fileName}...`);
      const chunks = await extractChunks(filePath, fileName, fileHash);
      console.log(`[EKSTRAKSI] ${chunks.length} chunks dibuat.`);
      newChunks.push(...chunks);
    } catch (error) {
      console.error(`[ERROR] Gagal memproses ${fileName}: ${error.message}`);
    }
  }
  return newChunks;
}

export async function markFilesAsProcessed(chunks) {
  if (!chunks?.length) return;
  const processedFiles = await loadProcessedFiles();
  for (const chunk of chunks) {
    const { fileName, fileHash } = chunk.metadata ?? {};
    if (fileName && fileHash) processedFiles[fileName] = fileHash;
  }
  await saveProcessedFiles(processedFiles);
}
