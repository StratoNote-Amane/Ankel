/**
 * llm.worker.js
 * ブラウザ内で完結する Wllama の Wasm エンジンを使う Web Worker。
 *
 * ── 受信メッセージ ──────────────────────────────────────────────────
 *   { type: 'LOAD_MODEL' }
 *      → Wllama モデルをロードする。完了後 LLM_READY を返す
 *   { type: 'GENERATE', data: { prompt } }
 *      → 応答を生成し LLM_CHUNK / LLM_DONE を返す
 *
 * 旧互換:
 *   { type: 'INIT' }  → LOAD_MODEL
 *   { type: 'CHAT', messages } → GENERATE
 *
 * ── 送信メッセージ ──────────────────────────────────────────────────
 *   { type: 'LLM_PROGRESS', pct: number, text: string }
 *   { type: 'LLM_READY', modelLabel: string }
 *   { type: 'LLM_CHUNK',  delta: string }
 *   { type: 'LLM_DONE' }
 *   { type: 'ERROR',      message: string, fatal: boolean }
 */

import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@1.22.4/dist/index.js';

let wllama = null;
let modelLoaded = false;

async function ensureEngine() {
  if (!wllama) {
    wllama = new Wllama({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@wllama/wllama@1.22.4/dist/${file}`,
    });
  }

  if (!modelLoaded) {
    const modelUrl = 'https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-v2/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf';
    await wllama.loadModelFromUrl(modelUrl, {
      parallel: 1,
      useCache: true,
    });
    modelLoaded = true;
  }
}

async function handleLoadModel() {
  self.postMessage({ type: 'LLM_PROGRESS', pct: 0, text: 'モデルを読み込み中…' });
  await ensureEngine();
  self.postMessage({ type: 'LLM_READY', modelLabel: 'Llama-3.2-1B-Instruct-Q4_K_M' });
}

function buildPromptFromMessages(messages) {
  if (!Array.isArray(messages)) return String(messages ?? '');
  return messages.map(msg => `${(msg.role ?? 'UNKNOWN').toUpperCase()}: ${msg.content ?? ''}`).join('\n');
}

async function handleGenerate(prompt) {
  if (!modelLoaded) {
    await handleLoadModel();
  }

  self.postMessage({ type: 'LLM_PROGRESS', pct: 0, text: '応答を生成中…' });
  const response = await wllama.createCompletion(prompt, {
    nPredict: 512,
    sampling: { temp: 0.7 },
  });

  self.postMessage({ type: 'LLM_CHUNK', delta: response });
  self.postMessage({ type: 'LLM_DONE' });
}

self.addEventListener('message', async (e) => {
  const { type, data, messages } = e.data;

  try {
    if (type === 'LOAD_MODEL' || type === 'INIT') {
      await handleLoadModel();
    } else if (type === 'GENERATE') {
      const prompt = data?.prompt ?? '';
      await handleGenerate(prompt);
    } else if (type === 'CHAT') {
      const prompt = buildPromptFromMessages(messages);
      await handleGenerate(prompt);
    }
  } catch (err) {
    self.postMessage({ type: 'ERROR', message: err.message ?? String(err), fatal: false });
  }
});
