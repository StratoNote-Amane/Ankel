/**
 * ai-worker.js - Ankel AI Worker
 *
 * ★ このファイルは必ず type:"module" Worker として起動すること (main.js側)
 *    new Worker('./ai-worker.js', { type: 'module' })
 *
 * importScripts は一切使用しない。
 * ライブラリはすべて ESM dynamic import で遅延ロードする。
 *
 * CDN が COEP ブロックされる問題について:
 *   coi-serviceworker.js が外部オリジンをパススルーするよう修正済みのため、
 *   jsdelivr / HuggingFace への fetch は正常に通る。
 */

// ════════════════════════════════════════════
//  状態
// ════════════════════════════════════════════

let WllamaClass   = null;   // Wllama コンストラクタ
let pipelineFn    = null;   // Transformers.js pipeline 関数
let wllamaInst    = null;   // Wllama インスタンス
let modelLoaded   = false;
let embedPipeline = null;

const SYSTEM_PROMPT =
  'あなたは生徒会総務の業務補佐システム「Ankel（アンケル）」です。' +
  'Scrapboxの過去資料を徹底的に分析し、校閲、資料作成、問題解決の思考を' +
  '誠実かつ冷静にサポートしてください。回答は簡潔かつ具体的に、' +
  '必要に応じて過去の事例を引用してください。';

// ════════════════════════════════════════════
//  モデル定義
// ════════════════════════════════════════════

const MODELS = {
  standard: {
    label: 'Llama-3.2-1B',
    url:   'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf',
  },
  writing: {
    label: 'Qwen2.5-1.5B',
    url:   'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
  },
  reasoning: {
    label: 'Gemma-2-2B',
    url:   'https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf',
  },
};

const WASM_PATHS = {
  'single-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.0/dist/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm':  'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.0/dist/multi-thread/wllama.wasm',
};

// ════════════════════════════════════════════
//  ライブラリ 遅延ロード
// ════════════════════════════════════════════

async function loadWllama() {
  if (WllamaClass) return;
  console.log('[Ankel Worker] Wllama dynamic import 開始...');
  try {
    const mod = await import('https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.0/dist/wllama.js');
    WllamaClass = mod.Wllama ?? mod.default?.Wllama ?? mod.default;
    if (typeof WllamaClass !== 'function') {
      throw new Error('Wllama クラスが見つかりません。export keys: ' + Object.keys(mod).join(', '));
    }
    console.log('[Ankel Worker] Wllama ロード完了');
  } catch (e) {
    console.error('[Ankel Worker] Wllama ロード失敗:', e);
    throw new Error('Wllamaのロードに失敗: ' + e.message);
  }
}

async function loadTransformers() {
  if (pipelineFn) return;
  console.log('[Ankel Worker] Transformers.js dynamic import 開始...');
  try {
    const mod = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.esm.js');
    pipelineFn = mod.pipeline ?? mod.default?.pipeline;
    if (typeof pipelineFn !== 'function') {
      throw new Error('pipeline 関数が見つかりません。export keys: ' + Object.keys(mod).join(', '));
    }
    const env = mod.env ?? mod.default?.env;
    if (env) { env.allowRemoteModels = true; env.useBrowserCache = true; }
    console.log('[Ankel Worker] Transformers.js ロード完了');
  } catch (e) {
    console.error('[Ankel Worker] Transformers.js ロード失敗:', e);
    throw new Error('Transformers.jsのロードに失敗: ' + e.message);
  }
}

// ════════════════════════════════════════════
//  メッセージハンドラ
// ════════════════════════════════════════════

self.onmessage = async (e) => {
  const { type, payload, id } = e.data;
  try {
    switch (type) {
      case 'LOAD_MODEL': await handleLoadModel(payload, id); break;
      case 'GENERATE':   await handleGenerate(payload, id);  break;
      case 'EMBED':      await handleEmbed(payload, id);     break;
      case 'UNLOAD':     await handleUnload(id);             break;
      default: postReply(id, 'ERROR', { message: 'Unknown type: ' + type });
    }
  } catch (err) {
    console.error('[Ankel Worker]', type, 'エラー:', err);
    postReply(id, 'ERROR', { message: err.message || String(err) });
  }
};

// ════════════════════════════════════════════
//  モデルロード
// ════════════════════════════════════════════

async function handleLoadModel({ modelKey }, id) {
  const modelDef = MODELS[modelKey] || MODELS.standard;

  postReply(id, 'PROGRESS', { status: 'ライブラリ読み込み中...', progress: 0 });
  await loadWllama();

  postReply(id, 'PROGRESS', { status: 'モデル初期化中: ' + modelDef.label, progress: 2 });

  if (wllamaInst) {
    try { await wllamaInst.exit(); } catch (_) {}
    wllamaInst = null; modelLoaded = false;
  }

  wllamaInst = new WllamaClass(WASM_PATHS, {
    n_threads: Math.min(navigator.hardwareConcurrency || 2, 4),
  });

  await wllamaInst.loadModelFromUrl(modelDef.url, {
    n_ctx:   2048,
    n_batch: 512,
    progressCallback: ({ loaded, total }) => {
      const pct = total > 0 ? Math.round((loaded / total) * 100) : 0;
      postReply(id, 'PROGRESS', { status: 'モデルダウンロード中... ' + pct + '%', progress: pct });
      if (pct % 10 === 0) console.log('[Ankel Worker] モデルロード:', pct + '%');
    },
  });

  modelLoaded = true;
  console.log('[Ankel Worker] モデルロード完了:', modelDef.label);
  postReply(id, 'SUCCESS', { label: modelDef.label });
}

// ════════════════════════════════════════════
//  テキスト生成
// ════════════════════════════════════════════

async function handleGenerate({ messages, context }, id) {
  if (!modelLoaded || !wllamaInst) throw new Error('モデルが読み込まれていません');

  const prompt = buildPrompt(messages, context);
  postReply(id, 'START', {});

  let generated = '';
  await wllamaInst.createCompletion(prompt, {
    nPredict:    800,
    temperature: 0.7,
    topP:        0.9,
    onNewToken: (_t, _p, text) => {
      generated += text;
      postReply(id, 'TOKEN', { token: text });
    },
  });

  postReply(id, 'DONE', { text: generated });
}

// ════════════════════════════════════════════
//  Embedding
// ════════════════════════════════════════════

async function handleEmbed({ text }, id) {
  await loadTransformers();

  if (!embedPipeline) {
    postReply(id, 'EMBED_LOADING', { status: 'Embeddingモデル読み込み中...' });
    console.log('[Ankel Worker] Embeddingモデル初期化...');
    embedPipeline = await pipelineFn('feature-extraction', 'Xenova/multilingual-e5-small', {
      progress_callback: (p) => {
        if (p.status === 'progress') {
          const pct = Math.round(p.progress || 0);
          postReply(id, 'EMBED_PROGRESS', { progress: pct });
          if (pct % 20 === 0) console.log('[Ankel Worker] Embeddingモデル:', pct + '%');
        }
      },
    });
    console.log('[Ankel Worker] Embeddingモデル準備完了');
  }

  const out = await embedPipeline(text, { pooling: 'mean', normalize: true });
  const vec = out.data instanceof Float32Array ? out.data : new Float32Array(out.data);
  postReply(id, 'EMBED_RESULT', { embedding: Array.from(vec) });
}

// ════════════════════════════════════════════
//  モデル解放
// ════════════════════════════════════════════

async function handleUnload(id) {
  if (wllamaInst) {
    try { await wllamaInst.exit(); } catch (_) {}
    wllamaInst = null; modelLoaded = false;
  }
  postReply(id, 'SUCCESS', {});
}

// ════════════════════════════════════════════
//  プロンプトビルダー
// ════════════════════════════════════════════

function buildPrompt(messages, context) {
  let sys = SYSTEM_PROMPT;
  if (context && context.length > 0) {
    sys += '\n\n【参考資料（過去のScrapboxページ）】\n';
    context.forEach((doc, i) => {
      sys += '\n--- 資料' + (i + 1) + ': ' + doc.title + ' ---\n';
      sys += (doc.body || '').slice(0, 600) + '\n';
    });
    sys += '\n上記の資料を参考にして回答してください。';
  }
  let prompt = '<|system|>\n' + sys + '\n';
  for (const m of messages) {
    if (m.role === 'user')      prompt += '<|user|>\n'      + m.content + '\n';
    if (m.role === 'assistant') prompt += '<|assistant|>\n' + m.content + '\n';
  }
  return prompt + '<|assistant|>\n';
}

// ════════════════════════════════════════════
//  ユーティリティ
// ════════════════════════════════════════════

function postReply(id, type, payload) {
  self.postMessage({ id, type, payload });
}
