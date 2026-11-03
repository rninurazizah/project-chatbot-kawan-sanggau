import 'dotenv/config';
import { extractNewPDFs } from './extractor.js';
import { processEmbeddings } from './embedder.js';

async function main() {
  console.log('=== INGEST: PDF -> Gemini Embeddings -> Qdrant ===');
  if (!process.env.GOOGLE_API_KEY) throw new Error('GOOGLE_API_KEY belum diatur dalam file .env.');
  const chunks = await extractNewPDFs();
  const count = await processEmbeddings(chunks);
  console.log(`=== SELESAI: ${count} chunks baru/berubah diproses ===`);
}

main().catch((error) => {
  console.error('[FATAL] Ingest gagal:', error);
  console.error(error.stack);
  process.exit(1);
});