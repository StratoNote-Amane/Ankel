/**
 * embed.worker.js
 * Transformers.js の埋め込みモデルをメインスレッドから完全分離する Web Worker。
 *
 * ── 受信メッセージ ──────────────────────────────────────────────────
 *   { type: 'START',       chunks: [{pageTitle,year,tag,title,text}] }
 *      → バッチで埋め込みを生成し DONE を返す
 *   { type: 'EMBED_QUERY', text: string }
 *      → クエリ1件を埋め込んで QUERY_RESULT を返す
 *
 * ── 送信メッセージ ──────────────────────────────────────────────────
 *   { type: 'MODEL_PROGRESS', pct: number }
 *   { type: 'EMBED_PROGRESS',  done: number, total: number }
 *   { type: 'DONE',            data: EmbeddedChunk[] }
 *   { type: 'QUERY_RESULT',    embedding: number[] }
 *   { type: 'ERROR',           message: string }
 */

import { pipeline, env }
  from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';

env.allowLocalModels = false;
env.useBrowserCache  = true;

const EMBED_MODEL  = 'Xenova/all-MiniLM-L6-v2';
const BATCH_SIZE   = 32;   // 一度に処理するチャンク数
const YIELD_EVERY  = 8;    // N バッチごとに GC・UI の余地を与える

let extractor = null;

/* ── モデル初期化（遅延・シングルトン） ── */
async function getExtractor() {
  if (extractor) return extractor;

  extractor = await pipeline('feature-extraction', EMBED_MODEL, {
    progress_callback: (p) => {
      self.postMessage({ type: 'MODEL_PROGRESS', pct: p.progress ?? 0 });
    },
  });

  self.postMessage({ type: 'MODEL_PROGRESS', pct: 100 });
  return extractor;
}

/* ── Float32Array → 通常配列変換（JSON シリアライズのため） ── */
function toArray(tensor) {
  return Array.from(tensor.data);
}

/* ── バッチ埋め込み生成 ── */
async function embedChunks(chunks) {
  const ext    = await getExtractor();
  const result = [];
  const total  = chunks.length;

  for (let i = 0; i < total; i += BATCH_SIZE) {
    const batch     = chunks.slice(i, i + BATCH_SIZE);
    const texts     = batch.map(c => c.text);

    // Transformers.js はテキスト配列を受け付ける
    const out = await ext(texts, { pooling: 'mean', normalize: true });

    // out.data は [batchSize × dim] の Float32Array（flat）
    const dim = out.dims[1];
    for (let b = 0; b < batch.length; b++) {
      const start = b * dim;
      result.push({
        pageTitle: batch[b].pageTitle,
        year:      batch[b].year,
        tag:       batch[b].tag,
        title:     batch[b].title,
        text:      batch[b].text,
        embedding: Array.from(out.data.slice(start, start + dim)),
      });
    }

    const done = Math.min(i + BATCH_SIZE, total);
    self.postMessage({ type: 'EMBED_PROGRESS', done, total });

    // N バッチごとにイベントループを解放
    const batchIdx = Math.floor(i / BATCH_SIZE);
    if (batchIdx % YIELD_EVERY === YIELD_EVERY - 1) {
      await new Promise(r => setTimeout(r, 0));
    }
  }

  return result;
}

/* ── クエリ 1 件を埋め込む ── */
async function embedQuery(text) {
  const ext = await getExtractor();
  const out  = await ext(text, { pooling: 'mean', normalize: true });
  return Array.from(out.data);
}

/* ── メッセージハンドラ ── */
self.addEventListener('message', async (e) => {
  const { type } = e.data;

  try {
    if (type === 'START') {
      const data = await embedChunks(e.data.chunks);
      self.postMessage({ type: 'DONE', data });

    } else if (type === 'EMBED_QUERY') {
      const embedding = await embedQuery(e.data.text);
      self.postMessage({ type: 'QUERY_RESULT', embedding });
    }
  } catch (err) {
    self.postMessage({ type: 'ERROR', message: err.message ?? String(err) });
  }
});
