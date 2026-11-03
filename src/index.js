import 'dotenv/config';
import { checkPythonApi, pythonApiUrl } from './python-rag.js';

// Generation dan retrieval kini berada di API Python (NVIDIA NIM + Qdrant).
// Google API key tidak lagi diperlukan untuk menjalankan bot.
console.log(`[KONFIGURASI] API Python RAG: ${pythonApiUrl}`);

checkPythonApi()
  .then((health) => {
    const retrieval = health?.checks?.retrieval;
    console.log(
      `[KONFIGURASI] API Python siap. Collection: ${retrieval?.collection ?? '-'} ` +
      `(${retrieval?.points ?? '?'} points, dim ${retrieval?.vector_size ?? '?'}).`
    );
  })
  .catch((error) => {
    // Jangan hentikan bot: QR WhatsApp tetap bisa muncul, dan pesan masuk
    // akan dibalas dengan pesan fallback sampai API Python dinyalakan.
    console.error(`[KONFIGURASI] API Python belum siap: ${error.message}`);
  });

// bot.js menangani QR, sesi WhatsApp, cooldown, sapaan, dan pesan pengguna.
await import('./bot.js');
