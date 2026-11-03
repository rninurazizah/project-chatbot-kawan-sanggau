import pkg from 'whatsapp-web.js';
import qrcode from 'qrcode-terminal';
import { answerQuestion, runGeneralAnswer } from './python-rag.js';

const { Client, LocalAuth } = pkg;

const COOLDOWN_MS = 5000;
const FALLBACK_MESSAGE = 'Maaf, terjadi kendala saat memproses pertanyaan kamu. Silakan coba lagi sebentar lagi.';
const NON_TEXT_MESSAGE = 'Maaf, saat ini saya hanya bisa membaca pesan teks.';

const HELP_MESSAGE =
  'Halo! Saya Kawan Sanggau, asisten informasi berbasis dokumen statistik yang tersedia.\n\n' +
  'Silakan tanyakan data seputar kecamatan ini, contohnya:\n' +
  '- Berapa jumlah penduduk Kecamatan Kapuas?\n' +
  '- Berapa luas wilayah Kecamatan Kapuas?\n' +
  '- Berapa jumlah sekolah dasar di Kecamatan Kapuas?';
const THANKS_MESSAGE = 'Sama-sama! Kalau ada pertanyaan lain seputar data Kecamatan Kapuas, silakan tanya lagi.';

const GREETING_PATTERN = /^(halo|hai|hi|hello|p|assalamu'?alaikum|selamat (pagi|siang|sore|malam))[\s!.,]*$/i;
const HELP_PATTERN = /^!?(bantuan|help|menu)[\s!.,]*$/i;
const THANKS_PATTERN = /^(terima kasih|makasih|thanks|thx|oke|ok)[\s!.,]*$/i;

// Pertanyaan tentang data/indikator statistik harus tetap dijawab melalui RAG.
const STATISTICAL_PATTERN = /\b(data|statistik|jumlah|berapa|persentase|persen|angka|penduduk|populasi|luas|wilayah|sekolah|pendidikan|kesehatan|pertanian|perkebunan|produksi|kemiskinan|pengangguran|pdrb|inflasi|sensus|tahun|tabel|indikator|kecamatan|kabupaten|desa|kelurahan|bps dalam angka)\b/i;

const lastRequestAt = new Map();

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: './wa_session' }),
  puppeteer: {
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  },
});

function shouldIgnore(msg) {
  return msg.fromMe || msg.from === 'status@broadcast' || msg.from.endsWith('@g.us');
}

function isOnCooldown(userId) {
  const now = Date.now();
  const last = lastRequestAt.get(userId) ?? 0;
  if (now - last < COOLDOWN_MS) return true;
  lastRequestAt.set(userId, now);
  return false;
}

/**
 * Balasan tetap untuk sapaan, ucapan terima kasih, dan permintaan bantuan,
 * sehingga tidak perlu memanggil pencarian maupun LLM.
 */
function getStaticReply(text) {
  if (GREETING_PATTERN.test(text) || HELP_PATTERN.test(text)) return HELP_MESSAGE;
  if (THANKS_PATTERN.test(text)) return THANKS_MESSAGE;
  return null;
}

function formatReply({ answer, pages, sources = [] }) {
  if (!pages.length && !sources.length) return answer;
  const sourceText = sources.length ? sources.join(', ') : 'dokumen pengetahuan';
  const pageText = pages.length ? `, hlm. ${pages.join(', ')}` : '';
  return `${answer}\n\n_Sumber: ${sourceText}${pageText}_`;
}

async function handleMessage(msg) {
  if (shouldIgnore(msg)) return;

  if (msg.type !== 'chat') {
    await msg.reply(NON_TEXT_MESSAGE);
    return;
  }

  const question = msg.body.trim();
  if (!question) return;

  const staticReply = getStaticReply(question);
  if (staticReply) {
    await msg.reply(staticReply);
    return;
  }

  if (isOnCooldown(msg.from)) return;

  try {
    const chat = await msg.getChat();
    await chat.sendStateTyping();

    const result = await answerQuestion(question);

    // Prioritaskan jawaban berbasis dokumen jika RAG menemukan konteks relevan.
    if (result.inScope) {
      await msg.reply(formatReply(result));
      return;
    }

    // Jika pertanyaan meminta data statistik tetapi dokumen tidak mendukungnya,
    // jangan menebak dengan model umum. Untuk pertanyaan umum, gunakan
    // generation NVIDIA melalui API Python.
    if (!STATISTICAL_PATTERN.test(question)) {
      await msg.reply(await runGeneralAnswer(question));
      return;
    }

    await msg.reply(formatReply(result));
  } catch (error) {
    console.error(`[ERROR] Gagal membalas ${msg.from}: ${error.message}`);
    await msg.reply(FALLBACK_MESSAGE);
  }
}

client.on('qr', (qr) => {
  console.log('[WA] Scan QR berikut dengan WhatsApp (Perangkat tertaut):');
  qrcode.generate(qr, { small: true });
});

client.on('authenticated', () => console.log('[WA] Autentikasi berhasil.'));
client.on('auth_failure', (reason) => console.error(`[WA] Autentikasi gagal: ${reason}`));
client.on('ready', () => console.log('[WA] Bot siap menerima pesan.'));
client.on('disconnected', (reason) => console.warn(`[WA] Terputus: ${reason}`));
client.on('message', handleMessage);

process.on('SIGINT', async () => {
  console.log('\n[WA] Menutup bot...');
  await client.destroy();
  process.exit(0);
});

client.initialize();