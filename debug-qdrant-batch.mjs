import 'dotenv/config';
import { extractNewPDFs } from './src/extractor.js';
import { createEmbeddings } from './src/embedder.js';
import { QdrantVectorStore } from '@langchain/qdrant';

const chunks = await extractNewPDFs();
const batch = chunks.slice(96, 128);
const name = 'kcda_debug_batch';

const embeddings = createEmbeddings();
const vectors = await embeddings.embedDocuments(
  batch.map((d) => d.pageContent)
);

console.log('Chunks:', batch.length);
console.log('Dimensi vektor:', vectors.map(v => v.length));

try {
  const store = await QdrantVectorStore.fromDocuments(
    batch,
    embeddings,
    {
      url: process.env.QDRANT_URL || 'http://localhost:6333',
      collectionName: name,
    },
  );

  console.log('BERHASIL: batch keempat tersimpan.');
  await store.client.deleteCollection(name);
  console.log('Collection uji dibersihkan.');
} catch (error) {
  console.error('GAGAL:', error.message);
  console.error(error.stack);
}
