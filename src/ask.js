
import fs from 'node:fs/promises';
import { answerQuestion } from './python-rag.js';

const QUESTIONS_FILE = './questions.json';

async function loadQuestions() {
  const question = process.argv.slice(2).join(' ').trim();

  if (question) {
    return [{ question }];
  }

  const raw = await fs.readFile(QUESTIONS_FILE, 'utf-8');
  return JSON.parse(raw);
}

function printResult(item, result, elapsedMs) {
  console.log('\n' + '='.repeat(60));
  console.log(`Pertanyaan : ${item.question}`);
  console.log(`Status     : ${result.inScope ? 'Konteks ditemukan' : 'Tidak dijawab dari dokumen'}`);
  console.log(`Skor       : ${Number(result.topScore ?? 0).toFixed(3)}`);

  console.log('\nJAWABAN:');
  console.log(result.answer || 'Tidak ada jawaban yang dihasilkan.');

  if (result.sources?.length) {
    console.log('\nSUMBER:');
    for (const source of result.sources) {
      console.log(`- ${source}`);
    }
  }

  if (result.pages?.length) {
    console.log(`Halaman    : ${result.pages.join(', ')}`);
  }

  console.log(`Total waktu: ${(elapsedMs / 1000).toFixed(2)} detik`);
  console.log('='.repeat(60));
}

async function main() {
  let questions;

  try {
    questions = await loadQuestions();
  } catch (error) {
    console.error('[ERROR] Gagal membaca pertanyaan:', error.message);
    process.exitCode = 1;
    return;
  }

  for (const item of questions) {
    const start = Date.now();

    try {
      const result = await answerQuestion(item.question);
      printResult(item, result, Date.now() - start);
    } catch (error) {
      console.error(`\n[ERROR] Pertanyaan: ${item.question}`);
      console.error(error?.message ?? error);
    }
  }
}

main().catch((error) => {
  console.error('[FATAL]', error?.message ?? error);
  process.exitCode = 1;
});


