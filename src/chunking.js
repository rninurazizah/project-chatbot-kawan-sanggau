
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { encodingForModel } from "js-tiktoken";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const INPUT_FILE = path.join(
  __dirname,
  "../extracted/normalized_data.jsonl"
);

const OUTPUT_FILE = path.join(
  __dirname,
  "../extracted/chunks.jsonl"
);

// Konfigurasi ukuran chunk narasi
const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 50;

// Konfigurasi chunk tabel
const TABLE_ROWS = 10;
const TABLE_OVERLAP = 2;

// Tokenizer untuk estimasi jumlah token
const tokenizer = encodingForModel("gpt-4");

// Membaca file JSONL
function readJsonl(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File tidak ditemukan: ${filePath}`);
  }

  return fs.readFileSync(filePath, "utf-8")
    .split(/\r?\n/)
    .filter(line => line.trim())
    .map(line => JSON.parse(line));
}

// Menghitung jumlah token
function countTokens(text) {
  return tokenizer.encode(text).length;
}

// Membuat splitter narasi
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 2500,
  chunkOverlap: 250,
  separators: ["\n\n", "\n", ". ", " ", ""]
});

// Chunking narasi
async function chunkNarrative(item) {
  const text = item.text ?? "";

  if (!text.trim()) return [];

  const pieces = await splitter.splitText(text);
  const results = [];

  let currentChunk = "";

  for (const piece of pieces) {
    const candidate = currentChunk
      ? `${currentChunk}\n${piece}`
      : piece;

    if (
      currentChunk &&
      countTokens(candidate) > CHUNK_SIZE
    ) {
      results.push(currentChunk);

      // Mengambil bagian akhir sebagai overlap
      const tokens = tokenizer.encode(currentChunk);
      const overlapTokens = tokens.slice(-CHUNK_OVERLAP);

      currentChunk = tokenizer.decode(overlapTokens);
      
      if (countTokens(`${currentChunk}\n${piece}`) <= CHUNK_SIZE) {
        currentChunk += `\n${piece}`;
      } else {
        currentChunk = piece;
      }
    } else {
      currentChunk = candidate;
    }
  }

  if (currentChunk.trim()) {
    results.push(currentChunk);
  }

  return results.map((content, index) => ({
    type: "narrative",
    content,
    metadata: {
      page: item.page,
      chunk_index: index,
      token_count: countTokens(content),
      source: item.source
    }
  }));
}

// Mengubah baris tabel menjadi teks
function formatTableRow(row, columns) {
  return columns
    .map(column => `${column}: ${row[column] ?? ""}`)
    .join(" | ");
}

// Chunking tabel berdasarkan kelompok baris
function chunkTable(item) {
  const results = [];

  for (
    let start = 0;
    start < item.rows.length;
    start += TABLE_ROWS - TABLE_OVERLAP
  ) {
    const rows = item.rows.slice(
      start,
      start + TABLE_ROWS
    );

    const header = [
      `Judul tabel: ${item.title}`,
      `Kolom: ${item.columns.join(" | ")}`
    ].join("\n");

    const tableRows = rows
      .map(row => formatTableRow(row, item.columns))
      .join("\n");

    const content = `${header}\n${tableRows}`;

    results.push({
      type: "table",
      content,
      metadata: {
        table_id: item.table_id,
        page: item.page,
        row_start: start,
        row_end: start + rows.length - 1,
        row_count: rows.length,
        token_count: countTokens(content),
        source: item.source
      }
    });

    if (start + TABLE_ROWS >= item.rows.length) {
      break;
    }
  }

  return results;
}

// Proses utama
async function main() {
  const documents = readJsonl(INPUT_FILE);
  const allChunks = [];

  for (const item of documents) {
    if (item.type === "narrative") {
      allChunks.push(...await chunkNarrative(item));
    } else if (item.type === "table") {
      allChunks.push(...chunkTable(item));
    } else {
      console.warn(
        `Tipe dokumen tidak dikenali: ${item.type}`
      );
    }
  }

  fs.writeFileSync(
    OUTPUT_FILE,
    allChunks.map(chunk => JSON.stringify(chunk)).join("\n"),
    "utf-8"
  );

  const narrativeChunks = allChunks.filter(
    chunk => chunk.type === "narrative"
  );

  const tableChunks = allChunks.filter(
    chunk => chunk.type === "table"
  );

  console.log("\n=== HASIL CHUNKING ===");
  console.log(`Total dokumen: ${documents.length}`);
  console.log(`Chunk narasi: ${narrativeChunks.length}`);
  console.log(`Chunk tabel: ${tableChunks.length}`);
  console.log(`Total chunk: ${allChunks.length}`);
  console.log(`Output: ${OUTPUT_FILE}`);

console.log(`Output: ${OUTPUT_FILE}`);
}

main().catch(error => {
  console.error("Chunking gagal:", error);
  process.exit(1);
});
