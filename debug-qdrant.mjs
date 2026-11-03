import 'dotenv/config';
import { Document } from '@langchain/core/documents';
import { QdrantVectorStore } from '@langchain/qdrant';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  modelName: process.env.GEMINI_EMBEDDING_MODEL || 'gemini-embedding-001',
});

try {
  const store = await QdrantVectorStore.fromDocuments(
    [
      new Document({
        pageContent: 'Dokumen uji statistik Kecamatan Kapuas.',
        metadata: { source: 'diagnostic-test' },
      }),
    ],
    embeddings,
    {
      url: process.env.QDRANT_URL || 'http://localhost:6333',
      collectionName: 'kcda_debug_test',
    },
  );

  console.log('BERHASIL: satu dokumen uji tersimpan.');
  await store.client.deleteCollection('kcda_debug_test');
  console.log('Collection uji dibersihkan.');
} catch (error) {
  console.error('UJI GAGAL:', error.message);
  console.error(error.stack);
}
