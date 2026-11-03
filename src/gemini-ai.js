import 'dotenv/config';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';

const CHAT_MODEL = process.env.GEMINI_CHAT_MODEL || 'gemini-2.5-flash';
let llm;

function getLLM() {
  if (!process.env.GOOGLE_API_KEY) {
    throw new Error('GOOGLE_API_KEY belum diatur dalam file .env.');
  }
  if (!llm) {
    llm = new ChatGoogleGenerativeAI({
      apiKey: process.env.GOOGLE_API_KEY,
      model: CHAT_MODEL,
      temperature: 0.3,
    });
  }
  return llm;
}

/**
 * Menjawab sapaan/pertanyaan umum sebagai admin digital BPS.
 * Fungsi ini tidak melakukan pencarian web; jangan gunakan sebagai sumber angka statistik.
 */
export async function runGeminiAi(message) {
  const cleanMessage = String(message ?? '').trim();
  if (!cleanMessage) return 'Silakan tuliskan pertanyaan Anda.';

  const response = await getLLM().invoke([
    ['system', `Kamu adalah admin digital Kawan Sanggau, asisten yang ramah untuk membantu pertanyaan umum tentang layanan BPS Kabupaten Sanggau. Jawab dalam bahasa Indonesia, maksimal 30 kata sebelum disclaimer. Jangan mengaku telah membuka situs web atau melakukan pencarian internet. Jangan mengarang informasi kontak, jadwal, angka statistik, atau kebijakan yang tidak diketahui. Jika pertanyaan membutuhkan data statistik spesifik, arahkan pengguna untuk menanyakan data dari dokumen resmi yang tersedia.`],
    ['human', cleanMessage],
  ]);

  const answer = typeof response.content === 'string'
    ? response.content
    : Array.isArray(response.content)
      ? response.content.map((part) => part.type === 'text' ? part.text : '').filter(Boolean).join('\n')
      : '';

  const disclaimer = `\n\n*Disclaimer:*\nJawaban ini dihasilkan oleh asisten digital (Gemini AI). Kami sangat menghargai saran dan kritik Anda untuk pengembangan yang lebih baik. Terima kasih! 😁`;
  return `${answer.trim() || 'Maaf, saya belum dapat menjawab pertanyaan tersebut.'}${disclaimer}`;
}
