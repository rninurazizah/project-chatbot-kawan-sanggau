import 'dotenv/config';

if (!process.env.GOOGLE_API_KEY) {
  console.error('[KONFIGURASI] GOOGLE_API_KEY belum diatur. Salin .env.example menjadi .env dan isi API key Gemini.');
  process.exit(1);
}

// bot.js menangani QR, sesi WhatsApp, cooldown, sapaan, dan pesan pengguna.
await import('./bot.js');
