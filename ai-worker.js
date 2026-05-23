/**
 * ai-worker.js - Ankel AI Worker
 *
 * 注意: Wllama は内部で自前Workerを起動するため、
 *       このWorkerからWllamaを呼ぶことはできない（document is not defined）。
 * 
 * このWorkerはTransformers.js (Embedding) 専用。
 * LLM推論はmain.jsがWllamaをメインスレッドで直接呼ぶ。
 */

// ════════════════════════════════════════════
//  状態
// ════════════════════════════════════════════

let pipelineFn    = null;
let embedPipeline = null;

// ════════════════════════════════════════════
//  Transformers.js 遅延ロード
// ════════════════════════════════════════════

async function loadTransformers() {
  if (pipelineFn) return;
  console.log('[Ankel EmbedWorker] Transformers.js import 開始...');
  try {
    const mod = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.esm.js');
    pipelineFn = mod.pipeline ?? mod.default?.pipeline;
    if (typeof pipelineFn !== 'function') {
      throw new Error('pipeline 関数が見つかりません。keys: ' + Object.keys(mod).join(', '));
    }
    const env = mod.env ?? mod.default?.env;
    if (env) { env.allowRemoteModels = true; env.useBrowserCache = true; }
    console.log('[Ankel EmbedWorker] Transformers.js ロード完了');
  } catch (e) {
    console.error('[Ankel EmbedWorker] Transformers.js ロード失敗:', e);
    throw new Error('Transformers.jsのロードに失敗: ' + e.message);
  }
}

// ════════════════════════════════════════════
//  メッセージハンドラ（EMBED のみ）
// ════════════════════════════════════════════

self.onmessage = async (e) => {
  const { type, payload, id } = e.data;
  try {
    if (type === 'EMBED') {
      await handleEmbed(payload, id);
    } else {
      postReply(id, 'ERROR', { message: 'EmbedWorkerはEMBEDのみ対応: ' + type });
    }
  } catch (err) {
    console.error('[Ankel EmbedWorker]', type, 'エラー:', err);
    postReply(id, 'ERROR', { message: err.message || String(err) });
  }
};

async function handleEmbed({ text }, id) {
  await loadTransformers();
  if (!embedPipeline) {
    postReply(id, 'EMBED_LOADING', { status: 'Embeddingモデル読み込み中...' });
    console.log('[Ankel EmbedWorker] Embeddingモデル初期化...');
    embedPipeline = await pipelineFn('feature-extraction', 'Xenova/multilingual-e5-small', {
      progress_callback: (p) => {
        if (p.status === 'progress') {
          const pct = Math.round(p.progress || 0);
          postReply(id, 'EMBED_PROGRESS', { progress: pct });
        }
      },
    });
    console.log('[Ankel EmbedWorker] Embeddingモデル準備完了');
  }
  const out = await embedPipeline(text, { pooling: 'mean', normalize: true });
  const vec = out.data instanceof Float32Array ? out.data : new Float32Array(out.data);
  postReply(id, 'EMBED_RESULT', { embedding: Array.from(vec) });
}

function postReply(id, type, payload) {
  self.postMessage({ id, type, payload });
}
