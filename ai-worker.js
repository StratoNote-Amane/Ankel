/**
 * ai-worker.js - Ankel AI Worker
 * Wllama (llama.cpp WASM) による CPU推論コア
 * Transformers.js による Embedding も担当
 */

// ── Wllama ────────────────────────────────────
// CDN から wllama をインポート（ESM）
importScripts('https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.0/dist/wllama.min.js');

// ── Transformers.js ───────────────────────────
importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers@3.5.0/dist/transformers.min.js');

const { pipeline, env: tfEnv } = self.Transformers ?? self.TransformersJS ?? {};

// Transformers.js の設定
if (tfEnv) {
  tfEnv.allowRemoteModels   = true;
  tfEnv.useBrowserCache     = true;
}

// ── 状態 ──────────────────────────────────────
let wllama       = null;
let modelLoaded  = false;
let currentModel = null;
let embedPipeline = null;

const SYSTEM_PROMPT = `あなたは生徒会総務の業務補佐システム「Ankel（アンケル）」です。Scrapboxの過去資料を徹底的に分析し、校閲、資料作成、問題解決の思考を誠実かつ冷静にサポートしてください。回答は簡潔かつ具体的に、必要に応じて過去の事例を引用してください。`;

// モデル定義
const MODELS = {
  standard: {
    name:  '標準モード（校閲・高速）',
    url:   'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf',
    label: 'Llama-3.2-1B',
  },
  writing: {
    name:  '文章作成特化',
    url:   'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
    label: 'Qwen2.5-1.5B',
  },
  reasoning: {
    name:  '論理・解決特化',
    url:   'https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf',
    label: 'Gemma-2-2B',
  },
};

// ── メッセージハンドラ ─────────────────────────
self.onmessage = async (e) => {
  const { type, payload, id } = e.data;

  try {
    switch (type) {
      case 'LOAD_MODEL':
        await handleLoadModel(payload, id);
        break;
      case 'GENERATE':
        await handleGenerate(payload, id);
        break;
      case 'EMBED':
        await handleEmbed(payload, id);
        break;
      case 'UNLOAD':
        await handleUnload(id);
        break;
      default:
        postReply(id, 'ERROR', { message: `Unknown type: ${type}` });
    }
  } catch (err) {
    postReply(id, 'ERROR', { message: err.message || String(err) });
  }
};

// ── モデルロード ──────────────────────────────
async function handleLoadModel({ modelKey }, id) {
  const modelDef = MODELS[modelKey] || MODELS.standard;

  postReply(id, 'PROGRESS', { status: `モデル初期化中: ${modelDef.label}`, progress: 0 });

  // 既存を解放
  if (wllama) {
    try { await wllama.exit(); } catch (_) {}
    wllama = null;
    modelLoaded = false;
  }

  // Wllama インスタンス生成
  wllama = new Wllama({
    'single-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.0/dist/single-thread/wllama.wasm',
    'multi-thread/wllama.wasm':  'https://cdn.jsdelivr.net/npm/@wllama/wllama@2.3.0/dist/multi-thread/wllama.wasm',
  }, {
    n_threads: Math.min(navigator.hardwareConcurrency || 2, 4),
  });

  await wllama.loadModelFromUrl(modelDef.url, {
    n_ctx:     2048,
    n_batch:   512,
    progressCallback: ({ loaded, total }) => {
      const pct = total > 0 ? Math.round((loaded / total) * 100) : 0;
      postReply(id, 'PROGRESS', {
        status:   `モデルダウンロード中... ${pct}%`,
        progress: pct,
      });
    },
  });

  modelLoaded  = true;
  currentModel = modelKey;
  postReply(id, 'SUCCESS', { label: modelDef.label });
}

// ── テキスト生成 ──────────────────────────────
async function handleGenerate({ messages, context }, id) {
  if (!modelLoaded || !wllama) {
    throw new Error('モデルが読み込まれていません');
  }

  // プロンプト構築
  const prompt = buildPrompt(messages, context);

  postReply(id, 'START', {});

  let generated = '';
  await wllama.createCompletion(prompt, {
    nPredict:    800,
    temperature: 0.7,
    topP:        0.9,
    onNewToken: (_token, _p, text) => {
      generated += text;
      postReply(id, 'TOKEN', { token: text });
    },
  });

  postReply(id, 'DONE', { text: generated });
}

// ── Embedding ─────────────────────────────────
async function handleEmbed({ text }, id) {
  if (!embedPipeline) {
    postReply(id, 'EMBED_LOADING', {});
    embedPipeline = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small', {
      progress_callback: (p) => {
        if (p.status === 'progress') {
          postReply(id, 'EMBED_PROGRESS', { progress: Math.round(p.progress || 0) });
        }
      },
    });
  }

  const output = await embedPipeline(`query: ${text}`, {
    pooling:   'mean',
    normalize: true,
  });

  // Float32Array を転送
  const vec = output.data instanceof Float32Array ? output.data : new Float32Array(output.data);
  postReply(id, 'EMBED_RESULT', { embedding: Array.from(vec) });
}

// ── モデル解放 ────────────────────────────────
async function handleUnload(id) {
  if (wllama) {
    await wllama.exit();
    wllama = null;
    modelLoaded = false;
  }
  postReply(id, 'SUCCESS', {});
}

// ── プロンプトビルダー ─────────────────────────
function buildPrompt(messages, context) {
  // 過去資料がある場合はシステムプロンプトに注入
  let sys = SYSTEM_PROMPT;
  if (context && context.length > 0) {
    sys += '\n\n【参考資料（過去のScrapboxページ）】\n';
    context.forEach((doc, i) => {
      sys += `\n--- 資料${i + 1}: ${doc.title} ---\n`;
      sys += doc.body.slice(0, 600) + '\n';
    });
    sys += '\n上記の資料を参考にして回答してください。';
  }

  // Llama-3 / ChatML 共通のシンプルなフォーマット
  let prompt = `<|system|>\n${sys}\n`;
  for (const m of messages) {
    if (m.role === 'user') {
      prompt += `<|user|>\n${m.content}\n`;
    } else if (m.role === 'assistant') {
      prompt += `<|assistant|>\n${m.content}\n`;
    }
  }
  prompt += '<|assistant|>\n';

  return prompt;
}

// ── ユーティリティ ────────────────────────────
function postReply(id, type, payload) {
  self.postMessage({ id, type, payload });
}
