// main.js - Ankel integration script
// WebLLM + IndexedDB based local inference system

const APP_VERSION    = '2025-05-24-r10';
const DB_NAME        = 'ankel_kb';
const DB_VERSION     = 4;  // v4: bodyClean / isSystem 対応
const STORE_PAGES    = 'pages';
const STORE_META     = 'meta';

const WEBLLM_CDN           = 'https://huggingface.co/mlc-ai/';
const WEBLLM_MODULE_URL    = 'https://esm.run/@mlc-ai/web-llm';
const WEBLLM_FALLBACK_URL  = 'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm/+esm';

let webllm = null;
try {
  webllm = await import(WEBLLM_MODULE_URL);
  console.log('[Ankel LLM] webllm loaded from esm.run');
} catch (err) {
  console.warn('[Ankel LLM] esm.run load failed, fallback to jsdelivr', err);
  webllm = await import(WEBLLM_FALLBACK_URL);
  console.log('[Ankel LLM] webllm loaded from jsdelivr');
}

let _db = null;
let engine = null;
let modelLoaded = false;
const MODEL_FALLBACK_CHAIN = [
  'SmolLM2-135M-Instruct-q0f16-MLC',
  'SmolLM2-360M-Instruct-q0f16-MLC',
  'Llama-3.2-1B-Instruct-q4f32_1-MLC',
  'Llama-3.2-3B-Instruct-q4f32_1-MLC',
];
const FALLBACK_MODEL_ID = MODEL_FALLBACK_CHAIN[0];
let currentModelId = FALLBACK_MODEL_ID;
let isGenerating = false;
let generationAbortController = null;
let scrapboxProject = '';
let chatHistory = [];
const MAX_HISTORY = 4;                    // 6→4（Liteモードのトークン節約）
const MAX_CONTEXT_PAGES = 4;
const MAX_CONTEXT_BODY_CHARS_LITE = 600;  // 480→600（中央値514文字をカバー）
const MAX_CONTEXT_BODY_CHARS_FULL = 960;  // 新設（70%のページが全文入る）
const MAX_CONTEXT_CHUNKS = 5;
const CHUNK_TARGET_TOKENS = 320;
const CHUNK_OVERLAP_LINES = 2;
const MAX_PROMPT_CONTEXT_TOKENS = 1400;

let pageChunks = null;
let linkGraph = null;
let contextPageLimit = MAX_CONTEXT_PAGES;

// Static inverted-index (loaded from dist/index.json when available)
let STATIC_INDEX = null;

function tokenizeForIndex(text) {
  text = normalizeText(text || '').toLowerCase();
  const tokens = [];
  // Latin / spacing tokens
  const latin = text.match(/[a-z0-9]+/g);
  if (latin) tokens.push(...latin);
  // Japanese/CJK: generate character bigrams and unigrams
  const cjk = text.replace(/[a-z0-9]+/g, '');
  const chars = Array.from(cjk.replace(/\s+/g, ''));
  for (let i = 0; i < chars.length; i++) {
    const u = chars[i];
    if (u.trim()) tokens.push(u);
    if (i + 1 < chars.length) tokens.push(u + chars[i+1]);
    if (i + 2 < chars.length) tokens.push(u + chars[i+1] + chars[i+2]);
  }
  // filter and dedupe small tokens
  return tokens.map(t => t.trim()).filter(t => t && t.length >= 1);
}

async function loadStaticIndex(url = 'dist/index.json') {
  try {
    const resp = await fetch(url, { cache: 'no-store' });
    if (!resp.ok) throw new Error('not found');
    STATIC_INDEX = await resp.json();
    console.log('[Ankel] loaded static index:', url, 'chunks:', (STATIC_INDEX.chunks||[]).length);
  } catch (e) {
    console.warn('[Ankel] static index not available:', e.message);
    STATIC_INDEX = null;
  }
}

function expandQueryTokens(tokens) {
  const expanded = new Set(tokens);
  for (const t of tokens) {
    for (const [k, syns] of Object.entries(SYNONYM_MAP)) {
      if (k.includes(t) || syns.some(s=>s.includes(t))) syns.forEach(s=>expanded.add(s.toLowerCase()));
    }
  }
  return Array.from(expanded);
}

function bm25Search(query, topK = 10) {
  if (!STATIC_INDEX || !STATIC_INDEX.inverted) return [];
  const k1 = 1.2, b = 0.75;
  const tokens = tokenizeForIndex(query);
  const qTokens = expandQueryTokens(tokens);
  const scores = Object.create(null);
  const inv = STATIC_INDEX.inverted;
  const df = STATIC_INDEX.df || {};
  const N = STATIC_INDEX.N || (STATIC_INDEX.chunks||[]).length;
  const avgdl = STATIC_INDEX.avgdl || 200;

  for (const t of qTokens) {
    const postings = inv[t];
    if (!postings) continue;
    const idf = Math.log(1 + (N - (df[t]||0) + 0.5) / ((df[t]||0) + 0.5));
    for (const p of postings) {
      const dl = STATIC_INDEX.docLens[p.chunkId] || 200;
      const tf = p.tf || 0;
      const denom = tf + k1 * (1 - b + b * (dl / avgdl));
      const score = idf * ((tf * (k1 + 1)) / Math.max(1e-6, denom));
      scores[p.chunkId] = (scores[p.chunkId] || 0) + score;
    }
  }

  const items = Object.keys(scores).map(cid => ({ chunkId: cid, score: scores[cid] }));
  items.sort((a,b)=>b.score - a.score);
  const top = items.slice(0, topK);
  // map chunkId to chunk objects
  const chunks = STATIC_INDEX.chunks || [];
  const chunkMap = Object.fromEntries((chunks||[]).map(c=>[c.chunkId,c]));
  return top.map(t => ({ ...chunkMap[t.chunkId], score: t.score } )).filter(Boolean);
}

// Lite mode / generation defaults
let LITE_MODE = true;
const GENERATION_MAX_TOKENS_DEFAULT = 320;
const GENERATION_MAX_TOKENS_LITE = 160;   // 120→160（短すぎて途切れる問題の対処）
const GENERATION_TEMPERATURE_DEFAULT = 0.3;
const GENERATION_TEMPERATURE_LITE = 0.1;
let generationMaxTokens = GENERATION_MAX_TOKENS_DEFAULT;
let generationTemperature = GENERATION_TEMPERATURE_DEFAULT;

// Lite Mode（SmolLM2-135M）向け: 極限まで短縮して指示遵守率を向上
const SYSTEM_PROMPT_LITE = `あなたは生徒会総務の業務補佐AI「Ankel」です。
提供された参考資料のみを根拠に、簡潔な日本語で回答してください。
資料にない情報は「記載なし」と答えてください。繰り返しを避けてください。`;

// 通常モード（SmolLM2-360M以上）向け
const SYSTEM_PROMPT_FULL = `あなたは生徒会総務の業務補佐システム「Ankel（アンケル）」です。
以下のルールを厳守してください：
- 提供された【参考資料】のみを根拠に回答する。推測で補わない。
- 資料にない情報は「資料には記載がありません」と明示する。
- 回答は簡潔・具体・日本語で。箇条書きを積極的に使う。
- 引用する場合は「資料○」の番号で出典を示す。
- 校閲・資料作成・問題解決を誠実にサポートする。`;

function getSystemPrompt() {
  return LITE_MODE ? SYSTEM_PROMPT_LITE : SYSTEM_PROMPT_FULL;
}

const SYNONYM_MAP = {
  '生徒総会': ['生徒総会', '総会', '全体会議'],
  '予算': ['予算', '会計', '収支', '決算'],
  '引き継ぎ': ['引き継ぎ', '引継', '引継書', '申し送り', '引き継ぎ書', '引継ぎ書'],
  '引き継ぎ書': ['引き継ぎ書', '引継書', '引き継ぎ', '引継ぎ書', '申し送り'],
  '文化祭': ['文化祭', '学校祭', '文化祭実行委員'],
  '体育祭': ['体育祭', '運動会', 'スポーツ大会'],
  '会計': ['会計', '予算', '収支', '決算'],
  '議事録': ['議事録', '議事要旨', '議事メモ'],
  '資料': ['資料', 'ドキュメント', '参考資料'],
  '総務': ['総務', '運営', '事務'],
  '報告': ['報告', 'レポート', '報告書'],
  '企画': ['企画', 'プラン', '計画'],
  'サラマレ': ['サラマレ', 'サラマレ懇談会'],
  'しらかし': ['しらかし', 'しらかし懇談会'],
  '懇談会': ['懇談会', 'サラマレ懇談会', 'しらかし懇談会'],
  '説明会': ['説明会', '学校説明会'],
  '広報': ['広報', '広報委員会', 'はなしたいむず'],
  'ぼくらの時代': ['ぼくらの時代', 'ぼくじだ'],
  '意見郵便': ['意見郵便', '意見箱', '生徒意見'],
  '中央委員会': ['中央委員会', '中央委員'],
};

function openDB() {
  if (_db) {
    try {
      // Ensure the cached connection is still usable before reusing it.
      _db.transaction(STORE_PAGES, 'readonly');
      return Promise.resolve(_db);
    } catch (err) {
      console.warn('[Ankel DB] cached DB unavailable, reopening:', err);
      _db = null;
    }
  }

  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      const oldVersion = e.oldVersion;

      if (oldVersion < 1) {
        const store = db.createObjectStore(STORE_PAGES, { keyPath: 'id', autoIncrement: true });
        store.createIndex('year', 'year', { unique: false });
        store.createIndex('tag', 'tag', { unique: false });
        store.createIndex('subtag', 'subtag', { unique: false });
        store.createIndex('title', 'title', { unique: false });
        db.createObjectStore(STORE_META, { keyPath: 'key' });
      } else {
        if (!db.objectStoreNames.contains(STORE_PAGES)) {
          const store = db.createObjectStore(STORE_PAGES, { keyPath: 'id', autoIncrement: true });
          store.createIndex('year', 'year', { unique: false });
          store.createIndex('tag', 'tag', { unique: false });
          store.createIndex('subtag', 'subtag', { unique: false });
          store.createIndex('title', 'title', { unique: false });
        }
        if (!db.objectStoreNames.contains(STORE_META)) {
          db.createObjectStore(STORE_META, { keyPath: 'key' });
        }
      }

      if (oldVersion < 4) {
        // v4: bodyClean / isSystem 対応（既存データは次回インポートで自動更新）
        console.log('[Ankel DB] v4 スキーマ移行: bodyClean/isSystem 対応');
      }
    };
    req.onsuccess = (e) => {
      _db = e.target.result;
      _db.onversionchange = () => {
        console.warn('[Ankel DB] version change detected; closing DB connection');
        _db.close();
        _db = null;
      };
      resolve(_db);
    };
    req.onerror = (e) => reject(e.target.error);
  });
}

function dbGetAll(storeName) {
  return openDB().then(db => new Promise((resolve, reject) => {
    const req = db.transaction(storeName, 'readonly').objectStore(storeName).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  }));
}

function dbGetByTitle(title) {
  return openDB().then(db => new Promise((resolve, reject) => {
    try {
      const tx = db.transaction(STORE_PAGES, 'readonly');
      const store = tx.objectStore(STORE_PAGES);
      if (store.indexNames && store.indexNames.contains && store.indexNames.contains('title')) {
        const req = store.index('title').get(title);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      } else {
        // fallback: scan all
        const req2 = store.getAll();
        req2.onsuccess = () => resolve((req2.result || []).find(p => p.title === title));
        req2.onerror = () => reject(req2.error);
      }
    } catch (e) { reject(e); }
  }));
}

function dbPut(storeName, record) {
  return openDB().then(db => new Promise((resolve, reject) => {
    const req = db.transaction(storeName, 'readwrite').objectStore(storeName).put(record);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  }));
}

function dbPutMany(storeName, records) {
  return openDB().then(db => new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    const store = tx.objectStore(storeName);
    let i = 0;
    function putNext() {
      if (i >= records.length) return;
      const req = store.put(records[i++]);
      req.onsuccess = putNext;
      req.onerror = () => reject(req.error);
    }
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    putNext();
  }));
}

function dbClear(storeName) {
  return openDB().then(db => new Promise((resolve, reject) => {
    const req = db.transaction(storeName, 'readwrite').objectStore(storeName).clear();
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  }));
}

function countPages() {
  return openDB().then(db => new Promise((resolve, reject) => {
    const req = db.transaction(STORE_PAGES, 'readonly').objectStore(STORE_PAGES).count();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  }));
}

function getAllPagesMeta() {
  return dbGetAll(STORE_META);
}

/**
 * 人物ページ・システムページかどうかを判定する（付録B）
 * 例: "47SoramiFukushima" → true, "settings" → true
 */
function isSystemPage(title) {
  if (/^\d+[A-Za-z]/.test(title)) return true;      // 人物ページ: 47SoramiFukushima 等
  if (/^\d+\s+[A-Za-z]/.test(title)) return true;   // 人物ページ: "47 furuya" 等
  if (['settings', 'hr', '資料管理ページ'].includes(title)) return true;
  return false;
}

/**
 * Scrapbox タイトルを構造化メタデータに分解する（第1章 完全再実装）
 * 実データのパターン:
 *   2024-引き継ぎ書-Microsoft-Teams-設定方法  (5seg, ハイフン含む固有名詞)
 *   2023-広報-外部-大学入学共通テスト          (4seg)
 *   2023-生徒総会-会則改正10-また又             (4seg, 数字混じり)
 *   2020-引き継ぎ書-Word                        (3seg)
 */
function parseTitleMeta(title) {
  const segs = title.split('-');

  // 年度判定（最初のセグメントが20xx）
  const yearMatch = segs[0]?.match(/^(20\d{2})$/);
  if (!yearMatch || segs.length < 2) {
    // 非日付形式: 人物ページ・設定ページ等
    const fallbackYear = title.match(/\b(20\d{2})\b/);
    return {
      year: fallbackYear ? parseInt(fallbackYear[1], 10) : null,
      tag: null,
      subtag: null,
      description: title,
      subdesc: null,
      allSegments: segs,
      isSystem: isSystemPage(title),
    };
  }

  const year = parseInt(yearMatch[1], 10);
  const rawTag = segs[1] || '不明';
  let tag, subtag, description, subdesc;

  if (segs.length === 2) {
    // 2024-予定 のような索引ページ
    tag = rawTag; subtag = null; description = ''; subdesc = null;
  } else if (segs.length === 3) {
    // 最多パターン: 2024-引き継ぎ書-生徒会室
    [, tag, description] = segs;
    subtag = null; subdesc = null;
  } else if (segs.length === 4) {
    // 2023-広報-外部-大学入学共通テスト
    [, tag, subtag, description] = segs;
    subdesc = null;
  } else {
    // 5セグメント以上: 2024-引き継ぎ書-Microsoft-Teams-設定方法
    tag = segs[1];
    subtag = segs[2];
    description = segs[3];
    subdesc = segs.slice(4).join('-') || null;
  }

  return {
    year,
    tag: tag || '不明',
    subtag: subtag || null,
    description: description || title,
    subdesc: subdesc || null,
    allSegments: segs,
    isSystem: false,
  };
}

function normalizeText(text) {
  return String(text || '').replace(/\r\n?/g, '\n').trim();
}

/**
 * Scrapbox本文からLLM向け「意味テキスト」を生成する（第2章）
 * 記法の意味マッピング:
 *   [- text]  → 削除案（除去）
 *   [_ text]  → 採用案（本文として残す）
 *   [/ text]  → 修正案（本文として残す）
 *   [* text]  → 強調（本文として残す）
 *   [*.icon]  → 発言者アイコン（除去）
 *   [PageTitle] → ページリンク（タイトルを保持）
 */
function cleanBodyForLLM(rawBody, options = {}) {
  const {
    removeStrikes = true,
    keepUnderline = true,
    removeIcons = true,
  } = options;

  let text = rawBody;

  // 1. 発言者アイコンを除去: [47SoramiFukushima.icon] 等
  if (removeIcons) {
    text = text.replace(/\[\d+\s*\w+\.icon\]/g, '');
    text = text.replace(/\[[^\]]+\.icon\]/g, '');
  }

  // 2. 削除案を除去: [- 削除テキスト]
  if (removeStrikes) {
    text = text.replace(/\[-\s*([^\]]*)\]/g, '');
  }

  // 3. 採用案を本文として残す: [_ 採用テキスト] → 採用テキスト
  if (keepUnderline) {
    text = text.replace(/\[_\s*([^\]]*)\]/g, '$1');
  }

  // 4. 修正案を本文として残す: [/ 修正テキスト] → 修正テキスト
  text = text.replace(/\[\/\s*([^\]]*)\]/g, '$1');

  // 5. 強調・見出しを本文として残す: [* テキスト] → テキスト
  text = text.replace(/\[\*+\s*([^\]]*)\]/g, '$1');

  // 6. 外部URLはラベルのみ残す、内部ページリンクはタイトルを保持
  text = text.replace(/\[https?:\/\/[^\]\s]+(?:\s([^\]]+))?\]/g, (_, label) => label || '');
  text = text.replace(/\[([^\]]+)\]/g, '$1');

  // 7. 空白・空行の正規化
  text = text.replace(/\t/g, '  ');
  text = text.replace(/\n{3,}/g, '\n\n');
  text = text.replace(/[ \u3000]{2,}/g, ' ');
  text = text.trim();

  return text;
}

/**
 * 文節境界でテキストを切り詰める（第3章）
 * 末尾が文の途中にならないように優先順位付きで境界を探す
 */
function sliceAtBoundary(text, maxLen) {
  if (text.length <= maxLen) return text;
  const sub = text.slice(0, maxLen);

  // 1. 段落区切り（空行）
  const paraBreak = sub.lastIndexOf('\n\n');
  if (paraBreak > maxLen * 0.55) return sub.slice(0, paraBreak).trimEnd();

  // 2. 行末
  const lineBreak = sub.lastIndexOf('\n');
  if (lineBreak > maxLen * 0.55) return sub.slice(0, lineBreak).trimEnd();

  // 3. 句点・疑問符・感嘆符
  const sentenceEnd = Math.max(
    sub.lastIndexOf('。'),
    sub.lastIndexOf('！'),
    sub.lastIndexOf('？'),
    sub.lastIndexOf('…'),
  );
  if (sentenceEnd > maxLen * 0.5) return sub.slice(0, sentenceEnd + 1).trimEnd();

  // 4. 読点・中黒
  const clauseEnd = Math.max(
    sub.lastIndexOf('、'),
    sub.lastIndexOf('・'),
  );
  if (clauseEnd > maxLen * 0.6) return sub.slice(0, clauseEnd + 1).trimEnd();

  // 5. やむなし: そのまま返す
  return sub;
}

function extractLinksFromText(text) {
  const links = new Set();
  const regex = /\[([^\]]+)\]/g;
  let m;
  while ((m = regex.exec(text))) {
    const target = m[1].trim();
    if (target) links.add(target);
  }
  return Array.from(links);
}

function extractHashtags(text) {
  const tags = new Set();
  const regex = /#([A-Za-z0-9_\u3000-\u303F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF-]+)/g;
  let m;
  while ((m = regex.exec(text))) {
    tags.add(m[1].trim());
  }
  return Array.from(tags);
}

function splitIntoBlocks(body) {
  const lines = normalizeText(body).split('\n');
  const blocks = [];
  let buffer = [];
  const flush = () => {
    const block = buffer.join('\n').trim();
    if (block) blocks.push(block);
    buffer = [];
  };

  for (const line of lines) {
    if (!line.trim()) {
      if (buffer.length > 0) {
        buffer.push(line);
      }
      continue;
    }
    const isNewSection = /^\s*[-*+]|^\s*\d+\.|^\s*#/i.test(line) || /^\S/.test(line) && buffer.length >= 5;
    if (buffer.length > 0 && isNewSection) {
      flush();
    }
    buffer.push(line);
  }
  flush();
  return blocks.length > 0 ? blocks : [normalizeText(body)];
}

function buildPageChunks(pages) {
  if (pageChunks) return pageChunks;
  pageChunks = [];
  linkGraph = {};

  pages.forEach(page => {
    const body = normalizeText(page.body || '');
    const pageLinks = extractLinksFromText(body);
    const hashtags = extractHashtags(body);
    const blocks = splitIntoBlocks(body);
    let chunkIndex = 0;

    blocks.forEach(block => {
      const snippet = normalizeText(block).slice(0, 200);
      pageChunks.push({
        pageId: page.id || page.title,
        chunkId: `${page.id || page.title}#${chunkIndex++}`,
        pageTitle: page.title,
        year: page.year,
        tag: page.tag,
        subtag: page.subtag,
        description: page.description,
        subdesc: page.subdesc,
        allSegments: page.allSegments,
        body: block,
        snippet,
        links: pageLinks,
        hashtags,
        score: 0,
      });
    });

    if (!linkGraph[page.title]) linkGraph[page.title] = new Set();
    pageLinks.forEach(link => linkGraph[page.title].add(link));
  });

  return pageChunks;
}

function parseQueryMetadata(query) {
  const normalized = query.trim();
  const yearMatches = Array.from(normalized.matchAll(/\b(20\d{2})\b/g)).map(m => Number(m[1]));
  const tagMatches = [];
  const lowered = normalized.toLowerCase();
  ['予算', '会計', '引き継ぎ', '総務', '文化祭', '体育祭', '議事録', '資料', '報告', '企画'].forEach(term => {
    if (lowered.includes(term.toLowerCase())) tagMatches.push(term);
  });
  return {
    years: yearMatches,
    tags: tagMatches,
  };
}

function queryScoreDropcut(sorted, topK) {
  if (sorted.length <= topK) return sorted.length;
  let cutoff = topK;
  for (let i = 1; i < sorted.length; i++) {
    const drop = sorted[i - 1].score - sorted[i].score;
    if (drop > sorted[0].score * 0.18 && i >= 2) {
      cutoff = i;
      break;
    }
  }
  return Math.min(cutoff, sorted.length);
}

async function importScrapboxJSON(json, onProgress) {
  pageChunks = null;
  linkGraph = null;
  const pages = json.pages || [];
  await dbClear(STORE_PAGES);

  const BATCH = 100;
  let done = 0;
  for (let i = 0; i < pages.length; i += BATCH) {
    const batch = pages.slice(i, i + BATCH);
    const records = batch.map(page => {
      const parsed = parseTitleMeta(page.title || '');
      const bodyRaw = (page.lines || []).map(l => typeof l === 'string' ? l : (l.text || '')).join('\n');
      const bodyClean = cleanBodyForLLM(bodyRaw);  // LLM向け正規化済み本文
      return {
        title: page.title || '',
        year: parsed.year,
        tag: parsed.tag,
        subtag: parsed.subtag,
        description: parsed.description,
        subdesc: parsed.subdesc,
        allSegments: parsed.allSegments,
        isSystem: parsed.isSystem,      // 人物・システムページフラグ
        body: bodyRaw,                  // 生データ（リンクグラフ構築用）
        bodyClean,                      // LLM向け正規化済み本文
        links: extractLinksFromText(bodyRaw),
        hashtags: extractHashtags(bodyRaw),
        updatedAt: page.updated || 0,
      };
    });
    await dbPutMany(STORE_PAGES, records);
    done += records.length;
    if (onProgress) onProgress(done, pages.length);
  }

  await dbPut(STORE_META, { key: 'lastImport', value: new Date().toISOString() });
  return done;
}

function expandQuery(query) {
  const normalized = query.toLowerCase();
  const baseTokens = normalized
    .split(/[\s　、。・,\.\-\/]+/)
    .map(t => t.trim())
    .filter(t => t.length >= 2);

  const expanded = new Set(baseTokens);
  for (const [key, synonyms] of Object.entries(SYNONYM_MAP)) {
    if (normalized.includes(key) || synonyms.some(s => normalized.includes(s))) {
      synonyms.forEach(s => { if (s.length >= 2) expanded.add(s.toLowerCase()); });
    }
  }
  return Array.from(expanded);
}

async function searchKeyword(query, topK = 10) {
  const allPages = await dbGetAll(STORE_PAGES);
  const chunks = buildPageChunks(allPages);
  const queryTokens = expandQuery(query);
  if (queryTokens.length === 0) queryTokens.push(query.toLowerCase().trim());
  const queryMeta = parseQueryMetadata(query);

  const scored = chunks.map(chunk => {
    // isSystem（人物ページ等）は検索対象外
    if (chunk.isSystem) return { ...chunk, score: 0 };

    const titleL = (chunk.pageTitle || '').toLowerCase();
    const tagL = (chunk.tag || '').toLowerCase();
    const subtagL = (chunk.subtag || '').toLowerCase();
    const descL = (chunk.description || '').toLowerCase();
    const subdescL = (chunk.subdesc || '').toLowerCase();
    const bodyL = (chunk.bodyClean || chunk.body || '').toLowerCase();
    const pageTokens = new Set([...chunk.allSegments.map(s => s.toLowerCase()), ...chunk.hashtags.map(t => t.toLowerCase())]);
    let score = 0;

    for (const token of queryTokens) {
      if (!token) continue;
      if (titleL === token) score += 28;
      if (titleL.includes(token)) score += 14;
      if (tagL === token) score += 12;
      if (subtagL === token) score += 10;
      if (descL.includes(token)) score += 8;
      if (subdescL.includes(token)) score += 6;
      if (pageTokens.has(token)) score += 8;
      const bodyCount = bodyL.split(token).length - 1;
      if (bodyCount > 0) score += bodyCount * 2;
    }

    // 年度一致: 乗数ブースト（加算から乗算へ変更、第4章）
    // 引き継ぎ書(1458件)のような大タグに埋もれないよう年度一致を強調
    let yearMultiplier = 1.0;
    queryMeta.years.forEach(year => {
      if (chunk.year === year) {
        yearMultiplier = Math.max(yearMultiplier, 1.6);  // 年度完全一致: 1.6倍
      } else if (Math.abs((chunk.year || 0) - year) === 1) {
        yearMultiplier = Math.max(yearMultiplier, 1.15); // 前後1年: 1.15倍
      }
    });
    score *= yearMultiplier;

    queryMeta.tags.forEach(tag => {
      if (tagL.includes(tag) || subtagL.includes(tag) || descL.includes(tag) || bodyL.includes(tag)) {
        score += 10;
      }
    });

    if (chunk.links && chunk.links.length > 0) {
      const linkBonus = chunk.links.filter(link => queryTokens.some(t => link.toLowerCase().includes(t))).length;
      score += Math.min(linkBonus, 3) * 3;
    }

    if (chunk.hashtags && chunk.hashtags.some(tag => queryTokens.includes(tag.toLowerCase()))) {
      score += 5;
    }

    if (chunk.body.length > 1200) score *= 0.92;
    if (chunk.body.length < 120) score *= 1.05;

    return { ...chunk, score };
  });

  const candidates = scored.filter(item => item.score > 0);
  candidates.sort((a, b) => b.score - a.score);

  const cutoff = queryScoreDropcut(candidates, topK);
  const filtered = candidates.slice(0, cutoff);

  const uniqueByPage = [];
  const seenPages = new Set();
  for (const item of filtered) {
    if (!seenPages.has(item.pageTitle) && uniqueByPage.length < contextPageLimit) {
      uniqueByPage.push(item);
      seenPages.add(item.pageTitle);
    }
  }

  return uniqueByPage;
}

async function collectPagesForQuery(query, topK = 5) {
  // get top chunk-level candidates
  const chunks = await searchKeyword(query, topK);
  const pages = [];
  for (const c of chunks) {
    try {
      const page = await dbGetByTitle(c.pageTitle);
      if (page) {
        // attach chunk-level hints
        page._score = c.score || 0;
        page._snippet = c.snippet || (page.body || '').slice(0, 200);
        pages.push(page);
      }
    } catch (e) {
      console.warn('[Ankel Search] collectPagesForQuery dbGetByTitle failed:', e);
    }
  }
  return pages;
}

function buildContextBlockFromPages(pages) {
  if (!pages || pages.length === 0) return '';
  const maxChars = LITE_MODE ? MAX_CONTEXT_BODY_CHARS_LITE : MAX_CONTEXT_BODY_CHARS_FULL;
  let block = '\n\n【参考資料（抽出された Scrapbox ページ）】\n';
  pages.slice(0, contextPageLimit).forEach((p, i) => {
    const title = p.title || '不明ページ';
    const yearTag = p.year ? `#${p.year}` : '';
    const tagTag = p.tag ? `#${p.tag}` : '';
    const subtagTag = p.subtag ? `#${p.subtag}` : '';
    // bodyClean を優先、なければ生bodyをフォールバック（後方互換）
    const rawText = normalizeText(p.bodyClean || p.body || '');
    const pageText = sliceAtBoundary(rawText, maxChars);
    block += `\n--- 資料${i + 1}: ${title} ${yearTag} ${tagTag} ${subtagTag} (score:${Math.round(p._score || 0)}) ---\n`;
    block += `${pageText}\n`;
    if (p.links && p.links.length > 0) block += `リンク: ${p.links.join(' / ')}\n`;
    if (p.hashtags && p.hashtags.length > 0) block += `ハッシュタグ: ${p.hashtags.map(t => `#${t}`).join(' ')}\n`;
  });
  block += '\n上記の参考資料は Scrapbox の構造を尊重して参照してください。必要ならば引用元のページタイトルを明示してください。';
  return block;
}

function checkWebGPU() {
  const warningBanner = document.getElementById('webgpu-warning');
  if (!navigator.gpu) {
    warningBanner.style.display = 'block';
    return false;
  }
  return navigator.gpu.requestAdapter()
    .then(adapter => {
      if (!adapter) throw new Error('No adapter');
      return true;
    })
    .catch(() => {
      warningBanner.style.display = 'block';
      return false;
    });
}

function getPreferredCacheBackends() {
  // Edge判定: EdgeはCache APIが最も安定している
  const isEdge = navigator.userAgent.includes('Edg/');
  if (isEdge) {
    return ['cache', 'indexeddb'];
  }
  return ['cache', 'indexeddb'];
}

function makeAppConfig(cacheBackend) {
  return {
    ...webllm.prebuiltAppConfig,
    cacheBackend,
  };
}

function isMemoryError(err) {
  if (!err || !err.message) return false;
  return /out of memory|memory|oom|Failed to allocate|Allocation failed/i.test(err.message);
}

function isWASMExitError(err) {
  if (!err) return false;
  const text = `${err.name || ''} ${err.message || ''}`;
  return /ExitStatus|Program terminated with exit\(1\)/i.test(text);
}

function isDeviceLostError(err) {
  if (!err) return false;
  const text = `${err.name || ''} ${err.message || ''}`;
  return /A valid external Instance reference no longer exists|Device was lost|GPUDeviceLostInfo|OperationError/i.test(text);
}

function isModelFallbackError(err) {
  return isMemoryError(err) || isWASMExitError(err) || isDeviceLostError(err);
}

function isIndexedDBClosingError(err) {
  if (!err || (!err.message && !err.name)) return false;
  const text = `${err.name || ''} ${err.message || ''}`;
  return /InvalidStateError|The database connection is closing|transaction[^\n]*on 'IDBDatabase'/i.test(text);
}

async function loadModel(modelId, { allowModelFallback = true } = {}) {
  currentModelId = modelId;
  if (modelSelect) modelSelect.value = modelId;
  console.log('[Ankel LLM] モデル読み込み開始:', modelId);
  setStatus('loading', 'モデル読み込み中...');
  modelOverlay.classList.add('visible');
  overlayText.textContent = 'モデルを準備しています...';
  overlayBar.style.width = '0%';
  sendBtn.disabled = true;

  const backends = getPreferredCacheBackends();
  let lastError = null;

  try {
    for (const cacheBackend of backends) {
      try {
        if (engine) {
          await engine.unload().catch(() => {});
          engine = null;
        }
        const appConfig = makeAppConfig(cacheBackend);
        console.log('[Ankel LLM] WebLLM engine init with cacheBackend:', cacheBackend);
        engine = new webllm.MLCEngine({ appConfig });
        engine.setInitProgressCallback(report => {
          overlayText.textContent = report.text;
          overlayBar.style.width = Math.round(report.progress * 100) + '%';
          console.log('[Ankel LLM] モデルロード:', Math.round(report.progress * 100) + '%', report.text);
        });
        await engine.reload(modelId);
        modelLoaded = true;
        console.log('[Ankel LLM] モデルロード完了:', modelId, 'cacheBackend:', cacheBackend);
        setStatus('ready', '準備完了');
        return;
      } catch (err) {
        lastError = err;
        console.warn('[Ankel LLM] モデルロード失敗:', cacheBackend, err);
        if ((cacheBackend === 'cache' || cacheBackend === 'opfs') && /Failed to execute 'add' on 'Cache'|Cache\.add\(\) encountered a network error/.test(err.message)) {
          console.warn('[Ankel LLM] cache/opfs backend failed, retrying with next backend');
          continue;
        }
        if (cacheBackend === 'indexeddb' && (isIndexedDBClosingError(err) || /IDBObjectStore|DataError|Evaluating the object store\'s key path did not yield a value/.test(err.message))) {
          console.warn('[Ankel LLM] indexeddb backend failed, retrying with next backend');
          continue;
        }
        if (cacheBackend === 'opfs' && err.name === 'NotSupportedError') {
          console.warn('[Ankel LLM] opfs backend unsupported, retrying with next backend');
          continue;
        }
        throw err;
      }
    }
    throw lastError;
  } catch (err) {
    if (allowModelFallback && isModelFallbackError(err)) {
      const currentIndex = MODEL_FALLBACK_CHAIN.indexOf(modelId);
      const nextFallback = currentIndex === -1
        ? MODEL_FALLBACK_CHAIN[0]
        : currentIndex < MODEL_FALLBACK_CHAIN.length - 1
          ? MODEL_FALLBACK_CHAIN[currentIndex + 1]
          : null;
      if (nextFallback && nextFallback !== modelId) {
        console.warn('[Ankel LLM] モデルロード失敗を検出したため別モデルへ切り替えます:', nextFallback, err);
        showToast('読み込みに失敗したため、より軽いモデルに切り替えます');
        await loadModel(nextFallback, { allowModelFallback: false });
        return;
      }
    }
    console.error('[Ankel LLM] モデルロードエラー:', err);
    setStatus('error', 'モデルロード失敗');
    showToast('モデルロードに失敗しました');
    throw err;
  } finally {
    modelOverlay.classList.remove('visible');
    sendBtn.disabled = false;
  }
}

function buildContextBlock(chunks) {
  if (!chunks || chunks.length === 0) return '';
  const maxChars = LITE_MODE ? MAX_CONTEXT_BODY_CHARS_LITE : MAX_CONTEXT_BODY_CHARS_FULL;
  let block = '\n\n【参考資料（Scrapbox 過去資料）】\n';

  chunks.slice(0, contextPageLimit).forEach((chunk, index) => {
    const title = chunk.pageTitle || '不明ページ';
    const yearTag = chunk.year ? `#${chunk.year}` : '';
    const tagTag = chunk.tag ? `#${chunk.tag}` : '';
    const subtagTag = chunk.subtag ? `#${chunk.subtag}` : '';
    const rawText = normalizeText(chunk.bodyClean || chunk.body || '');
    const chunkText = sliceAtBoundary(rawText, maxChars);

    block += `\n--- 資料${index + 1}: ${title} ${yearTag} ${tagTag} ${subtagTag} (chunk:${chunk.chunkId}, score:${Math.round(chunk.score)}) ---\n`;
    block += `${chunkText}\n`;
    if (chunk.links && chunk.links.length > 0) {
      block += `リンク: ${chunk.links.join(' / ')}\n`;
    }
    if (chunk.hashtags && chunk.hashtags.length > 0) {
      block += `ハッシュタグ: ${chunk.hashtags.map(t => `#${t}`).join(' ')}\n`;
    }
  });

  block += '\n上記の参考資料は Scrapbox の構造（タイトル/タグ/リンク/箇条書き）を尊重して参照してください。必要に応じて引用元タイトルとチャンクIDを明示してください。';
  return block;
}

function addToHistory(role, content) {
  chatHistory.push({ role, content });
  if (chatHistory.length > MAX_HISTORY) {
    chatHistory = chatHistory.slice(-MAX_HISTORY);
  }
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isGenerating) return;
  userInput.value = '';
  userInput.style.height = 'auto';
  isGenerating = true;
  sendBtn.disabled = true;
  stopBtn.style.display = 'inline-flex';
  setStatus('loading', '推論中...');

  appendMessage('user', text);
  addToHistory('user', text);

  const contextPages = await retrieveContext(text);
  const assistantBubble = appendMessage('assistant', '', contextPages);
  const bubbleBody = assistantBubble.querySelector('.msg-text');

  generationAbortController = new AbortController();
  const abortSignal = generationAbortController.signal;

  try {
    if (!modelLoaded || !engine) throw new Error('モデルが読み込まれていません');
    if (!contextPages || contextPages.length === 0) {
      bubbleBody.textContent = '参照ページが見つかりません。別のキーワードで再検索してください。';
      addToHistory('assistant', bubbleBody.textContent);
      return;
    }
    const contextBlock = buildContextBlockFromPages(contextPages);
    const stream = await engine.chat.completions.create({
      messages: [
        { role: 'system', content: getSystemPrompt() + contextBlock },
        ...chatHistory,
        { role: 'user', content: text },
      ],
      stream: true,
      temperature: generationTemperature,
      top_p: 0.9,
      max_tokens: generationMaxTokens,
      frequency_penalty: 0.3,
      presence_penalty: 0.1,
    });

    let fullText = '';
    for await (const chunk of stream) {
      if (abortSignal.aborted) break;
      const delta = chunk.choices?.[0]?.delta?.content ?? '';
      fullText += delta;
      bubbleBody.textContent = fullText;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    if (abortSignal.aborted && !fullText) {
      bubbleBody.textContent = '生成が停止されました。';
    }

    addToHistory('assistant', fullText || bubbleBody.textContent);
  } catch (err) {
    if (abortSignal.aborted) {
      console.log('[Ankel LLM] 生成が中断されました');
    } else {
      console.error('[Ankel LLM] 生成エラー:', err);
      if (isDeviceLostError(err)) {
        if (engine) {
          await engine.unload().catch(() => {});
          engine = null;
          modelLoaded = false;
        }
        const currentIndex = MODEL_FALLBACK_CHAIN.indexOf(currentModelId);
        const nextFallback = currentIndex >= 0 && currentIndex < MODEL_FALLBACK_CHAIN.length - 1
          ? MODEL_FALLBACK_CHAIN[currentIndex + 1]
          : null;
        if (nextFallback) {
          bubbleBody.textContent = 'GPUデバイスが切断されたため、より軽いモデルに切り替えています...';
          console.warn('[Ankel LLM] GPU device lost during generation, switching to smaller model:', nextFallback);
          showToast('GPUデバイスが切断されたため、軽量モデルへ切り替えます');
          await loadModel(nextFallback, { allowModelFallback: false });
          bubbleBody.textContent += '\n再送信してください。';
        } else {
          bubbleBody.textContent = 'エラー: GPUデバイスが切断されました。ページを再読み込みしてください。';
        }
      } else {
        if (!bubbleBody.textContent) bubbleBody.textContent = 'エラー: ' + err.message;
      }
    }
  } finally {
    generationAbortController = null;
    isGenerating = false;
    sendBtn.disabled = false;
    stopBtn.style.display = 'none';
    setStatus('ready', '準備完了');
  }
}

function stopGeneration() {
  if (generationAbortController) {
    generationAbortController.abort();
    console.log('[Ankel LLM] 生成を中断しました');
  }
}

async function retrieveContext(query) {
  console.log('[Ankel Search] retrieveContext:', query.slice(0, 100));
  const pages = await collectPagesForQuery(query, MAX_CONTEXT_CHUNKS);
  console.log('[Ankel Search] 取得ページ:', pages.map(p => `${p.title}(${Math.round(p._score||0)})`));
  return pages;
}

function makeScrapboxUrl(title) {
  if (!scrapboxProject) return null;
  return `https://scrapbox.io/${encodeURIComponent(scrapboxProject)}/${encodeURIComponent(title)}`;
}

let haloCount = 0;
function makeHaloSVG() {
  const uid = ++haloCount;
  return `
<svg width="32" height="32" viewBox="0 0 180 180" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <clipPath id="rc-msg-${uid}">
      <rect width="180" height="180" rx="37.8" ry="37.8"/>
    </clipPath>
  </defs>
  <rect width="180" height="180" rx="37.8" ry="37.8" fill="#0d1117"/>
  <g transform="translate(90,90) scale(0.84375) translate(-80,-80)" clip-path="url(#rc-msg-${uid})">
    <path d="M 80,12 L 108,52 L 148,80 L 108,108 L 80,148 L 52,108 L 12,80 L 52,52 Z"
          fill="none" stroke="#76c0ea" stroke-width="20"
          stroke-linejoin="miter" stroke-miterlimit="10"/>
  </g>
</svg>
`;
}

function appendMessage(role, text, sources = []) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;
  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.innerHTML = role === 'assistant' ? makeHaloSVG() : 'YOU';
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  const bodyEl = document.createElement('span');
  bodyEl.className = 'msg-text';
  bodyEl.textContent = text;
  bubble.appendChild(bodyEl);

  if (role === 'assistant' && !text) {
    const typing = document.createElement('div');
    typing.className = 'typing-indicator';
    typing.innerHTML = '<span></span><span></span><span></span>';
    bubble.appendChild(typing);
    const obs = new MutationObserver(() => {
      if (bodyEl.textContent) { typing.remove(); obs.disconnect(); }
    });
    obs.observe(bodyEl, { childList: true, characterData: true, subtree: true });
  }

  if (sources.length > 0) {
    const srcWrap = document.createElement('div');
    srcWrap.className = 'msg-sources';
    srcWrap.innerHTML = '<div class="src-label">参照資料</div>';
    sources.slice(0, 5).forEach(src => {
      const title = typeof src === 'string' ? src : src.title;
      const url = makeScrapboxUrl(title);
      const row = document.createElement('div');
      row.className = 'src-row';
      if (url) {
        const a = document.createElement('a');
        a.className = 'src-link';
        a.href = url;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.textContent = title;
        row.appendChild(a);
      } else {
        const span = document.createElement('span');
        span.className = 'src-chip';
        span.textContent = title;
        row.appendChild(span);
      }
      srcWrap.appendChild(row);
    });
    bubble.appendChild(srcWrap);
  }

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function setStatus(state, text) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}

let toastTimer;
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), 2800);
}

function escHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const importScreen       = document.getElementById('import-screen');
const appEl              = document.getElementById('app');
const dropZone           = document.getElementById('drop-zone');
const fileInput          = document.getElementById('json-file-input');
const importStatus       = document.getElementById('import-status');
const importProgressWrap = document.getElementById('import-progress-wrap');
const importProgressBar  = document.getElementById('import-progress-bar');
const importProgressPct  = document.getElementById('import-progress-pct');
const messagesEl         = document.getElementById('messages');
const userInput          = document.getElementById('user-input');
const sendBtn            = document.getElementById('send-btn');
const stopBtn            = document.getElementById('stop-btn');
const modelSelect        = document.getElementById('model-select');
const statusDot          = document.getElementById('status-dot');
const statusText         = document.getElementById('status-text');
const modelOverlay       = document.getElementById('model-overlay');
const overlayText        = document.getElementById('overlay-text');
const overlayBar         = document.getElementById('overlay-bar');
const toastEl            = document.getElementById('toast');
const updateBtn          = document.getElementById('update-btn');
const updateModal        = document.getElementById('update-modal');
const updateDropZone     = document.getElementById('update-drop-zone');
const updateFileInput    = document.getElementById('update-file-input');
const updateStatus       = document.getElementById('update-status');
const updateProgressWrap = document.getElementById('update-progress-wrap');
const updateProgressBar  = document.getElementById('update-progress-bar');
const updateProgressPct  = document.getElementById('update-progress-pct');
const updateModalClose   = document.getElementById('update-modal-close');
const liteModeCheckbox   = document.getElementById('lite-mode-checkbox');

async function initialize() {
  console.log(`[Ankel] ${APP_VERSION} 起動`);
  const webgpuOk = await checkWebGPU();
  // attempt to load static index for fast retrieval
  await loadStaticIndex('dist/index.json');
  await openDB();
  const count = await countPages();
  console.log('[Ankel DB] IndexedDB ページ数:', count);

  const meta = await getAllPagesMeta();
  const projectMeta = meta.find(r => r.key === 'projectName');
  if (projectMeta) scrapboxProject = projectMeta.value;

  if (count > 0) {
    console.log('[Ankel] 既存データあり → チャット画面へ');
    if (!webgpuOk) {
      setStatus('error', 'WebGPUに対応していません');
      showToast('WebGPU非対応のためモデルは読み込めません');
    }
    await launchApp(webgpuOk);
  } else {
    console.log('[Ankel] データなし → インポート画面を表示');
    importScreen.style.display = 'flex';
    setStatus('error', 'データがありません。JSONをインポートしてください。');
  }
}

function setImportProgress(pct, label) {
  importProgressWrap.classList.add('show');
  importProgressBar.style.width = pct + '%';
  importProgressPct.textContent = label ?? pct + '%';
}

function hideImportProgress() {
  importProgressWrap.classList.remove('show');
  importProgressBar.style.width = '0%';
  importProgressPct.textContent = '0%';
}

function setImportStatus(msg, cls) {
  importStatus.textContent = msg;
  importStatus.className = `import-status ${cls}`;
}

function setUpdateProgress(pct, label) {
  updateProgressWrap.classList.add('show');
  updateProgressBar.style.width = pct + '%';
  updateProgressPct.textContent = label ?? pct + '%';
}

function hideUpdateProgress() {
  updateProgressWrap.classList.remove('show');
  updateProgressBar.style.width = '0%';
  updateProgressPct.textContent = '0%';
}

function setUpdateStatus(msg, cls) {
  updateStatus.textContent = msg;
  updateStatus.className = `import-status ${cls}`;
}

async function launchApp(webgpuOk) {
  importScreen.style.display = 'none';
  appEl.classList.add('visible');
  modelSelect.value = currentModelId;
  if (webgpuOk) {
    await loadModel(currentModelId);
  }
  setStatus(webgpuOk ? 'ready' : 'error', webgpuOk ? '準備完了' : 'WebGPU非対応');
}

function openFileDialog(fileInput) {
  fileInput.click();
}

function setupDropZone(zone, input, mode) {
  zone.addEventListener('click', () => openFileDialog(input));
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) handleJSONFile(e.dataTransfer.files[0], mode);
  });
  input.addEventListener('change', () => {
    if (input.files[0]) handleJSONFile(input.files[0], mode);
  });
}

setupDropZone(dropZone, fileInput, 'import');
setupDropZone(updateDropZone, updateFileInput, 'update');

updateBtn.addEventListener('click', () => {
  updateModal.classList.add('visible');
  setUpdateStatus('', '');
  hideUpdateProgress();
});

updateModalClose.addEventListener('click', () => updateModal.classList.remove('visible'));
updateModal.addEventListener('click', e => {
  if (e.target === updateModal) updateModal.classList.remove('visible');
});

async function handleJSONFile(file, mode) {
  const isUpdate = mode === 'update';
  const setStatus_ = isUpdate ? setUpdateStatus : setImportStatus;
  const setProgress_ = isUpdate ? setUpdateProgress : setImportProgress;
  const hideProgress_ = isUpdate ? hideUpdateProgress : hideImportProgress;

  if (!file.name.endsWith('.json')) {
    setStatus_('JSONファイルを選択してください', 'error');
    console.warn('[Ankel] ファイル拒否:', file.name);
    return;
  }

  console.log('[Ankel] ファイル受付:', file.name, `${(file.size / 1024).toFixed(2)} KB`);
  setStatus_('JSONを解析中...', '');
  setProgress_(0, '0%');

  try {
    const raw = await file.text();
    const json = JSON.parse(raw);
    if (!Array.isArray(json.pages)) {
      setStatus_('Scrapbox JSONの形式が正しくありません', 'error');
      hideProgress_();
      return;
    }

    if (json.name) {
      scrapboxProject = json.name;
      await dbPut(STORE_META, { key: 'projectName', value: json.name });
      console.log('[Ankel DB] Scrapboxプロジェクト名保存:', json.name);
    }

    const done = await importScrapboxJSON(json, (doneCount, totalCount) => {
      const pct = Math.round((doneCount / totalCount) * 100);
      setStatus_(`${doneCount} / ${totalCount} ページを保存中...`, '');
      setProgress_(pct, pct + '%');
    });

    console.log('[Ankel DB] DB書き込み完了:', done, 'ページ');
    setProgress_(100, '100%');
    setStatus_(`✓ ${done}ページを読み込みました`, 'success');

    if (isUpdate) {
      setTimeout(() => {
        updateModal.classList.remove('visible');
        showToast(`${done}ページを更新しました`);
      }, 1200);
    } else {
      const webgpuOk = await checkWebGPU();
      setTimeout(() => launchApp(webgpuOk), 1000);
    }
  } catch (err) {
    setStatus_(`エラー: ${err.message}`, 'error');
    console.error('[Ankel] handleJSONFile エラー:', err);
    hideProgress_();
  }
}

sendBtn.addEventListener('click', sendMessage);
stopBtn.addEventListener('click', stopGeneration);
userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
userInput.addEventListener('input', () => {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 140) + 'px';
});

modelSelect.addEventListener('change', async () => {
  if (isGenerating) {
    showToast('生成中はモデルを切り替えできません');
    modelSelect.value = currentModelId;
    return;
  }
  if (currentModelId !== modelSelect.value) {
    await loadModel(modelSelect.value);
  }
});

function applyLiteModeSettings() {
  LITE_MODE = !!(liteModeCheckbox && liteModeCheckbox.checked);
  generationMaxTokens = LITE_MODE ? GENERATION_MAX_TOKENS_LITE : GENERATION_MAX_TOKENS_DEFAULT;
  generationTemperature = LITE_MODE ? GENERATION_TEMPERATURE_LITE : GENERATION_TEMPERATURE_DEFAULT;
  contextPageLimit = LITE_MODE ? Math.max(1, Math.min(2, MAX_CONTEXT_PAGES)) : MAX_CONTEXT_PAGES;
  if (LITE_MODE && modelSelect) {
    // enforce smallest model selection in lite mode
    modelSelect.value = MODEL_FALLBACK_CHAIN[0];
    currentModelId = MODEL_FALLBACK_CHAIN[0];
  }
}

if (liteModeCheckbox) {
  liteModeCheckbox.addEventListener('change', async () => {
    applyLiteModeSettings();
    showToast(`Lite Mode ${liteModeCheckbox.checked ? '有効' : '無効'}`);
    if (!isGenerating && modelSelect && currentModelId !== modelSelect.value) {
      try { await loadModel(modelSelect.value); } catch(e) { console.warn(e); }
    }
  });
}

applyLiteModeSettings();

initialize().catch(err => {
  console.error('[Ankel] 初期化エラー:', err);
  setStatus('error', '初期化に失敗しました');
});
