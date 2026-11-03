import 'dotenv/config';
import { createVectorStore } from './src/embedder.js';

const store = createVectorStore();

const results = await store.similaritySearch(
  'Berapa jumlah desa atau kelurahan di Kecamatan Kapuas?',
  5
);

for (const [i, doc] of results.entries()) {
  console.log(`\nHasil ${i + 1}`);
  console.log('Metadata:', doc.metadata);
  console.log('Isi:', doc.pageContent);
}