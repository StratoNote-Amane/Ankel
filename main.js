/**
 * main.js - Ankel 統合スクリプト
 *
 * 変更点:
 *  - サイドバー廃止
 *  - データ更新ボタン・モーダル追加
 *  - アバターをヘイローSVGに変更
 *  - 情報源をページ全文タイトル表示 + Scrapboxリンク付き
 *  - parseTitleMeta を多段タグ構造に対応
 *    (yyyy-tag-subtag-desc-subdesc, 各サブは省略可)
 *  - searchKeyword を多段スコアリング＋類似語展開に強化
 */

// ════════════════════════════════════════════
//  STORAGE
// ════════════════════════════════════════════

const DB_NAME     = 'ankel_kb';
const DB_VERSION  = 3;
const STORE_PAGES = 'pages';
const STORE_META  = 'meta';

let _db = null;

function openDB() {
  if (_db) return Promise.resolve(_db);
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const d = e.target.result;
      if (!d.objectStoreNames.contains(STORE_PAGES)) {
        const store = d.createObjectStore(STORE_PAGES, { keyPath: 'id', autoIncrement: true });
        store.createIndex('year',   'year',   { unique: false });
        store.createIndex('tag',    'tag',    { unique: false });
        store.createIndex('subtag', 'subtag', { unique: false });
        store.createIndex('title',  'title',  { unique: false });
      } else {
        const store = e.target.transaction.objectStore(STORE_PAGES);
        ['year','tag','subtag','title'].forEach(idx => {
          if (!store.indexNames.contains(idx))
            store.createIndex(idx, idx, { unique: false });
        });
      }
      if (!d.objectStoreNames.contains(STORE_META))
        d.createObjectStore(STORE_META, { keyPath: 'key' });
    };
    req.onsuccess = (e) => { _db = e.target.result; resolve(_db); };
    req.onerror   = (e) => reject(e.target.error);
  });
}

function dbGetAll(storeName) {
  return openDB().then(d => new Promise((resolve, reject) => {
    const req = d.transaction(storeName, 'readonly').objectStore(storeName).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  }));
}

function dbPut(storeName, record) {
  return openDB().then(d => new Promise((resolve, reject) => {
    const req = d.transaction(storeName, 'readwrite').objectStore(storeName).put(record);
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  }));
}

function dbPutMany(storeName, records) {
  return openDB().then(d => new Promise((resolve, reject) => {
    const tx    = d.transaction(storeName, 'readwrite');
    const store = tx.objectStore(storeName);
    let i = 0;
    function putNext() {
      if (i >= records.length) return;
      const req = store.put(records[i++]);
      req.onsuccess = putNext;
      req.onerror   = () => reject(req.error);
    }
    tx.oncomplete = () => resolve();
    tx.onerror    = () => reject(tx.error);
    putNext();
  }));
}

function dbClear(storeName) {
  return openDB().then(d => new Promise((resolve, reject) => {
    const req = d.transaction(storeName, 'readwrite').objectStore(storeName).clear();
    req.onsuccess = () => resolve();
    req.onerror   = () => reject(req.error);
  }));
}

function countPages() {
  return openDB().then(d => new Promise((resolve, reject) => {
    const req = d.transaction(STORE_PAGES, 'readonly').objectStore(STORE_PAGES).count();
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  }));
}

// ════════════════════════════════════════════
//  タイトルパーサー（多段構造対応）
//
//  対応フォーマット（- 区切り、先頭4桁が年度）:
//    yyyy-tag-desc
//    yyyy-tag-subtag-desc
//    yyyy-tag-desc-subdesc
//    yyyy-tag-subtag-desc-subdesc
//    上記以外（年度なし等）はベストエフォートで解析
// ════════════════════════════════════════════

function parseTitleMeta(title) {
  // セグメント分割
  const segs = title.split('-');

  // 先頭が4桁年度かチェック
  const yearMatch = segs[0]?.match(/^(20\d{2})$/);
  if (!yearMatch || segs.length < 3) {
    // 年度なし or セグメント不足 → フォールバック
    const y = title.match(/\b(20\d{2})\b/);
    return {
      year:        y ? parseInt(y[1]) : null,
      tag:         segs[1] || '不明',
      subtag:      null,
      description: segs.slice(2).join('-') || title,
      subdesc:     null,
      allSegments: segs,
    };
  }

  const year = parseInt(yearMatch[1]);
  const rest  = segs.slice(1); // 年度以降

  // rest = [tag, ...残り]
  // 残りは 1〜3個。Scrapboxの実態に合わせてヒューリスティックに解釈:
  //   rest.length === 1 : [tag]            → desc は空
  //   rest.length === 2 : [tag, desc]
  //   rest.length === 3 : [tag, subtag, desc]  or [tag, desc, subdesc]
  //   rest.length === 4 : [tag, subtag, desc, subdesc]
  //   rest.length >= 5  : [tag, subtag, desc, subdesc, ...残りをsubdescに結合]

  let tag, subtag, description, subdesc;

  if (rest.length === 1) {
    tag = rest[0]; subtag = null; description = ''; subdesc = null;
  } else if (rest.length === 2) {
    [tag, description] = rest; subtag = null; subdesc = null;
  } else if (rest.length === 3) {
    // 2番目が業務サブタグかどうか判定するのは困難なため、
    // [tag, desc, subdesc] と解釈（サブタグはdescに含める）
    [tag, description, subdesc] = rest; subtag = null;
  } else {
    // 4セグメント以上: [tag, subtag, desc, subdesc, ...]
    tag         = rest[0];
    subtag      = rest[1];
    description = rest[2];
    subdesc     = rest.slice(3).join('-') || null;
  }

  return {
    year,
    tag:         tag         || '不明',
    subtag:      subtag      || null,
    description: description || title,
    subdesc:     subdesc     || null,
    allSegments: segs,
  };
}

// ════════════════════════════════════════════
//  インポート
// ════════════════════════════════════════════

async function importScrapboxJSON(json, onProgress) {
  const pages = json.pages || [];
  await dbClear(STORE_PAGES);

  const BATCH = 100;
  let done = 0;

  for (let i = 0; i < pages.length; i += BATCH) {
    const batch   = pages.slice(i, i + BATCH);
    const records = batch.map(page => {
      const parsed   = parseTitleMeta(page.title || '');
      const bodyText = (page.lines || [])
        .map(l => typeof l === 'string' ? l : (l.text || ''))
        .join('\n');
      return {
        title:       page.title || '',
        year:        parsed.year,
        tag:         parsed.tag,
        subtag:      parsed.subtag,
        description: parsed.description,
        subdesc:     parsed.subdesc,
        allSegments: parsed.allSegments,
        body:        bodyText,
        embedding:   null,
        updatedAt:   page.updated || 0,
      };
    });

    await dbPutMany(STORE_PAGES, records);
    done += batch.length;
    onProgress && onProgress(done, pages.length);
  }

  await dbPut(STORE_META, { key: 'lastImport', value: new Date().toISOString() });
  await dbPut(STORE_META, { key: 'embeddingDone', value: false });
  return done;
}

// ════════════════════════════════════════════
//  バックグラウンド Embedding
// ════════════════════════════════════════════

async function embedAllPagesInBackground(onProgress) {
  const allRows   = await dbGetAll(STORE_PAGES);
  const needEmbed = allRows.filter(p => !p.embedding);
  if (needEmbed.length === 0) return;

  for (let i = 0; i < needEmbed.length; i++) {
    const p = needEmbed[i];
    try {
      // 文書側: passage: + タイトル全体 + 本文先頭
      const embedText = `passage: ${p.title} ${p.body.slice(0, 400)}`;
      const vec = await sendWorker('EMBED', { text: embedText });
      await dbPut(STORE_PAGES, { ...p, embedding: Array.from(vec) });
    } catch (_) {}
    onProgress && onProgress(i + 1, needEmbed.length);
  }

  await dbPut(STORE_META, { key: 'embeddingDone', value: true });
}

// ════════════════════════════════════════════
//  検索
// ════════════════════════════════════════════

function cosineSimilarity(a, b) {
  if (a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
  }
  return (na && nb) ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

async function searchSimilar(queryVec, topK = 7) {
  const all   = await dbGetAll(STORE_PAGES);
  const cands = all.filter(p => p.embedding);
  const qv    = Array.from(queryVec);
  return cands
    .map(p => ({ ...p, score: cosineSimilarity(qv, p.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

/**
 * キーワード検索（多段スコアリング）
 *
 * スコア内訳:
 *   タイトル完全一致フレーズ: ×10
 *   タグ一致: ×5
 *   サブタグ一致: ×3
 *   description一致: ×4
 *   本文出現回数: ×1（出現ごと）
 *   年度一致（数字4桁が含まれる場合）: ×2
 *
 * さらに、クエリを形態素的に分割（スペース・読点・句点区切り）して
 * 各トークンでスコアを合算することで「似たような事象」を拾う。
 */
async function searchKeyword(query, topK = 7) {
  const all = await dbGetAll(STORE_PAGES);

  // トークン分割: スペース・句読点・助詞的な短語で分割
  const tokens = query
    .toLowerCase()
    .split(/[\s　、。・,.\-\/]+/)
    .map(t => t.trim())
    .filter(t => t.length >= 2); // 1文字トークンは除外

  // トークンが空ならクエリ全体をそのまま使う
  if (tokens.length === 0) tokens.push(query.toLowerCase().trim());

  // 年度トークンを抽出（4桁数字）
  const yearTokens = tokens.filter(t => /^20\d{2}$/.test(t)).map(Number);

  const scored = all.map(p => {
    const titleL = (p.title  || '').toLowerCase();
    const tagL   = (p.tag    || '').toLowerCase();
    const subtagL= (p.subtag || '').toLowerCase();
    const descL  = (p.description || '').toLowerCase();
    const bodyL  = (p.body   || '').toLowerCase();

    let score = 0;

    for (const token of tokens) {
      // タイトル内フレーズ出現
      const titleCount = titleL.split(token).length - 1;
      score += titleCount * 10;

      // タグ一致
      if (tagL.includes(token))    score += 5;
      if (subtagL.includes(token)) score += 3;
      if (descL.includes(token))   score += 4;

      // 本文出現回数
      const bodyCount = bodyL.split(token).length - 1;
      score += bodyCount;
    }

    // 年度ボーナス
    if (yearTokens.length > 0 && yearTokens.includes(p.year)) {
      score += 2 * yearTokens.length;
    }

    return { ...p, score };
  });

  return scored
    .filter(p => p.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// ════════════════════════════════════════════
//  WORKER ブリッジ
// ════════════════════════════════════════════

// ESM Worker（ai-worker.js は type:module）
const worker      = new Worker('./ai-worker.js', { type: 'module' });
let   msgCounter  = 0;
const pendingMsgs = new Map();
let   onTokenCb   = null;

worker.onmessage = (e) => {
  const { id, type, payload } = e.data;
  if (type === 'TOKEN') { onTokenCb && onTokenCb(payload.token); return; }
  const h = pendingMsgs.get(id);
  if (!h) return;
  if (type === 'ERROR') {
    pendingMsgs.delete(id); h.reject(new Error(payload.message)); return;
  }
  if (['PROGRESS','EMBED_PROGRESS','EMBED_LOADING'].includes(type)) {
    h.onProgress && h.onProgress(payload); return;
  }
  if (type === 'START')        { h.onStart && h.onStart(); return; }
  if (type === 'EMBED_RESULT') { pendingMsgs.delete(id); h.resolve(payload.embedding); return; }
  if (type === 'DONE' || type === 'SUCCESS') { pendingMsgs.delete(id); h.resolve(payload); return; }
};

function sendWorker(type, payload, onProgress, onStart) {
  return new Promise((resolve, reject) => {
    const id = ++msgCounter;
    pendingMsgs.set(id, { resolve, reject, onProgress, onStart });
    worker.postMessage({ type, payload, id });
  });
}

// ════════════════════════════════════════════
//  DOM 参照
// ════════════════════════════════════════════

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

// ════════════════════════════════════════════
//  状態
// ════════════════════════════════════════════

let chatHistory     = [];
let isGenerating    = false;
let currentBubble   = null;
let currentModelKey = 'standard';
let embeddingReady  = false;

// Scrapbox のプロジェクト名（情報源リンク生成に使用）
// Scrapbox JSON の "name" フィールドから取得する
let scrapboxProject = '';

// ════════════════════════════════════════════
//  初期化
// ════════════════════════════════════════════

(async () => {
  console.log('[Ankel] 初期化開始');
  await openDB();
  const count = await countPages();
  console.log(`[Ankel] IndexedDB: ${count}ページ存在`);
  if (count > 0) {
    // プロジェクト名を meta から復元
    const meta = await dbGetAll(STORE_META);
    const pj   = meta.find(r => r.key === 'projectName');
    if (pj) scrapboxProject = pj.value;
    console.log('[Ankel] 既存データあり → チャット画面へ');
    await launchApp();
  } else {
    console.log('[Ankel] データなし → インポート画面を表示');
    importScreen.style.display = 'flex';
  }
})();

// ════════════════════════════════════════════
//  インポート（初回）
// ════════════════════════════════════════════

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) handleJSONFile(e.dataTransfer.files[0], 'import');
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleJSONFile(fileInput.files[0], 'import');
});

// ════════════════════════════════════════════
//  データ更新モーダル
// ════════════════════════════════════════════

updateBtn.addEventListener('click', () => {
  updateModal.classList.add('visible');
  setUpdateStatus('', '');
  hideUpdateProgress();
});
updateModalClose.addEventListener('click', () => updateModal.classList.remove('visible'));
updateModal.addEventListener('click', (e) => { if (e.target === updateModal) updateModal.classList.remove('visible'); });

updateDropZone.addEventListener('click', () => updateFileInput.click());
updateDropZone.addEventListener('dragover',  (e) => { e.preventDefault(); updateDropZone.classList.add('dragover'); });
updateDropZone.addEventListener('dragleave', () => updateDropZone.classList.remove('dragover'));
updateDropZone.addEventListener('drop', (e) => {
  e.preventDefault(); updateDropZone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) handleJSONFile(e.dataTransfer.files[0], 'update');
});
updateFileInput.addEventListener('change', () => {
  if (updateFileInput.files[0]) handleJSONFile(updateFileInput.files[0], 'update');
});

// ════════════════════════════════════════════
//  JSON ファイル処理（初回 / 更新 共通）
// ════════════════════════════════════════════

async function handleJSONFile(file, mode) {
  const isUpdate = mode === 'update';

  const setStatus_ = isUpdate ? setUpdateStatus      : setImportStatus;
  const setProgress_= isUpdate ? setUpdateProgress   : setImportProgress;
  const hideProgress_= isUpdate ? hideUpdateProgress : hideImportProgress;

  if (!file.name.endsWith('.json')) {
    setStatus_('JSONファイルを選択してください', 'error');
    console.warn('[Ankel] ファイル拒否:', file.name);
    return;
  }

  console.log('[Ankel] ファイル受付:', file.name, `(${(file.size / 1024).toFixed(1)} KB)`);
  setStatus_('JSONを解析中...', '');
  setProgress_(0, '0%');

  try {
    const raw  = await file.text();
    const json = JSON.parse(raw);
    console.log('[Ankel] JSON.parse完了。pages数:', json.pages?.length ?? 'なし');

    if (!Array.isArray(json.pages)) {
      setStatus_('Scrapbox JSONの形式が正しくありません', 'error');
      hideProgress_();
      return;
    }

    // プロジェクト名を保存（リンク生成に使う）
    if (json.name) {
      scrapboxProject = json.name;
      await dbPut(STORE_META, { key: 'projectName', value: json.name });
      console.log('[Ankel] Scrapboxプロジェクト名:', json.name);
    }

    const done = await importScrapboxJSON(json, (d, t) => {
      const pct = Math.round((d / t) * 100);
      setStatus_(`${d} / ${t} ページを保存中...`, '');
      setProgress_(pct, `${pct}%`);
    });

    console.log(`[Ankel] DB書き込み完了: ${done}ページ`);
    setProgress_(100, '100%');
    setStatus_(`✓ ${done}ページを読み込みました`, 'success');

    if (isUpdate) {
      embeddingReady = false;
      setTimeout(async () => {
        updateModal.classList.remove('visible');
        showToast(`${done}ページを更新しました`);
        startBackgroundEmbedding();
      }, 1200);
    } else {
      setTimeout(() => launchApp(), 1000);
    }

  } catch (err) {
    setStatus_(`エラー: ${err.message}`, 'error');
    console.error('[Ankel] handleJSONFile エラー:', err);
    hideProgress_();
  }
}

// ════════════════════════════════════════════
//  進捗バーヘルパー（インポート画面用）
// ════════════════════════════════════════════

function setImportProgress(pct, label) {
  importProgressWrap.classList.add('show');
  importProgressBar.style.width  = `${pct}%`;
  importProgressPct.textContent  = label ?? `${pct}%`;
}
function hideImportProgress() {
  importProgressWrap.classList.remove('show');
  importProgressBar.style.width = '0%';
  importProgressPct.textContent = '0%';
}
function setImportStatus(msg, cls) {
  importStatus.textContent = msg;
  importStatus.className   = `import-status ${cls}`;
}

// 更新モーダル用
function setUpdateProgress(pct, label) {
  updateProgressWrap.classList.add('show');
  updateProgressBar.style.width  = `${pct}%`;
  updateProgressPct.textContent  = label ?? `${pct}%`;
}
function hideUpdateProgress() {
  updateProgressWrap.classList.remove('show');
  updateProgressBar.style.width = '0%';
  updateProgressPct.textContent = '0%';
}
function setUpdateStatus(msg, cls) {
  updateStatus.textContent = msg;
  updateStatus.className   = `import-status ${cls}`;
}

// ════════════════════════════════════════════
//  アプリ起動
// ════════════════════════════════════════════

async function launchApp() {
  console.log('[Ankel] launchApp: チャット画面を起動');
  importScreen.style.display = 'none';
  appEl.classList.add('visible');
  await loadModel(currentModelKey);
  startBackgroundEmbedding();
}

// ════════════════════════════════════════════
//  バックグラウンド Embedding
// ════════════════════════════════════════════

async function startBackgroundEmbedding() {
  const metaRows = await dbGetAll(STORE_META);
  const doneMeta = metaRows.find(r => r.key === 'embeddingDone');
  console.log('[Ankel] startBackgroundEmbedding: embeddingDone =', doneMeta?.value ?? '(未設定)');

  if (doneMeta && doneMeta.value === true) {
    embeddingReady = true;
    console.log('[Ankel] Embedding済み。スキップ');
    return;
  }

  setStatus('loading', '検索インデックス構築中...');
  let lastPct = 0;

  try {
    await embedAllPagesInBackground((done, total) => {
      const pct = Math.round((done / total) * 100);
      if (pct !== lastPct) {
        lastPct = pct;
        setStatus('loading', `インデックス構築中 ${pct}%`);
        if (pct % 10 === 0) console.log(`[Ankel] Embedding進捗: ${done}/${total} (${pct}%)`);
      }
    });
    embeddingReady = true;
    console.log('[Ankel] Embedding完了');
    setStatus('ready', '準備完了');
    showToast('検索インデックスの構築が完了しました');
  } catch (err) {
    console.warn('[Ankel] バックグラウンドEmbedding失敗:', err);
    setStatus('ready', '準備完了（キーワード検索のみ）');
  }
}

// ════════════════════════════════════════════
//  モデルロード
// ════════════════════════════════════════════

async function loadModel(key) {
  currentModelKey = key;
  console.log('[Ankel] loadModel 開始:', key);
  setStatus('loading', 'モデル読み込み中...');
  modelOverlay.classList.add('visible');
  sendBtn.disabled = true;

  try {
    await sendWorker('LOAD_MODEL', { modelKey: key }, ({ status, progress }) => {
      overlayText.textContent = status;
      overlayBar.style.width  = `${progress}%`;
      if (progress % 10 === 0) console.log(`[Ankel] モデルロード: ${progress}% — ${status}`);
    });
    console.log('[Ankel] モデルロード完了:', key);
    setStatus('ready', '準備完了');
    showToast('モデル読み込み完了');
  } catch (err) {
    setStatus('error', 'ロードエラー');
    showToast(`エラー: ${err.message}`);
    console.error('[Ankel] モデルロードエラー:', err);
  } finally {
    modelOverlay.classList.remove('visible');
    sendBtn.disabled = false;
  }
}

modelSelect.addEventListener('change', () => { if (!isGenerating) loadModel(modelSelect.value); });

// ════════════════════════════════════════════
//  チャット
// ════════════════════════════════════════════

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
userInput.addEventListener('input', () => {
  userInput.style.height = 'auto';
  userInput.style.height = `${Math.min(userInput.scrollHeight, 140)}px`;
});

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isGenerating) return;
  userInput.value = '';
  userInput.style.height = 'auto';
  isGenerating = true;
  sendBtn.disabled = true;
  setStatus('loading', '推論中...');

  appendMessage('user', text);
  chatHistory.push({ role: 'user', content: text });

  const context = await retrieveContext(text);
  currentBubble = appendMessage('assistant', '', context);
  const bubbleBody = currentBubble.querySelector('.msg-text');

  onTokenCb = (token) => {
    bubbleBody.textContent += token;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  };

  try {
    await sendWorker('GENERATE', { messages: chatHistory, context });
  } catch (err) {
    bubbleBody.textContent = `エラー: ${err.message}`;
    console.error('[Ankel] 生成エラー:', err);
  }

  onTokenCb = null;
  chatHistory.push({ role: 'assistant', content: bubbleBody.textContent });
  isGenerating = false;
  currentBubble = null;
  sendBtn.disabled = false;
  setStatus('ready', '準備完了');
}

// ════════════════════════════════════════════
//  RAG
// ════════════════════════════════════════════

async function retrieveContext(query) {
  console.log('[Ankel] retrieveContext:', query.slice(0, 50), '| embeddingReady:', embeddingReady);

  if (embeddingReady) {
    try {
      const vec = await sendWorker('EMBED', { text: `query: ${query}` });
      const res = await searchSimilar(vec, 7);
      console.log('[Ankel] 意味検索結果:', res.map(r => `${r.title}(${r.score?.toFixed(3)})`));
      if (res.length > 0) return res;
    } catch (e) {
      console.warn('[Ankel] Embedding検索失敗、キーワード検索にフォールバック:', e);
    }
  }

  const kwRes = await searchKeyword(query, 7);
  console.log('[Ankel] キーワード検索結果:', kwRes.map(r => r.title));
  return kwRes;
}

// ════════════════════════════════════════════
//  Scrapbox リンク生成
// ════════════════════════════════════════════

function makeScrapboxUrl(title) {
  if (!scrapboxProject) return null;
  return `https://scrapbox.io/${encodeURIComponent(scrapboxProject)}/${encodeURIComponent(title)}`;
}

// ════════════════════════════════════════════
//  メッセージ描画
// ════════════════════════════════════════════

// アバター用ヘイローSVG（指定デザイン統一版）
// clipPath ID の重複を避けるためカウンタで連番管理
let haloCount = 0;

function makeHaloSVG() {
  const uid = ++haloCount;
  return `<svg width="32" height="32" viewBox="0 0 180 180" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <clipPath id="rc-msg${uid}">
        <rect width="180" height="180" rx="37.8" ry="37.8"/>
      </clipPath>
    </defs>
    <rect width="180" height="180" rx="37.8" ry="37.8" fill="#0d1117"/>
    <g transform="translate(90,90) scale(0.84375) translate(-80,-80)" clip-path="url(#rc-msg${uid})">
      <path d="M 80,12 L 108,52 L 148,80 L 108,108 L 80,148 L 52,108 L 12,80 L 52,52 Z"
            fill="none" stroke="#76c0ea" stroke-width="20"
            stroke-linejoin="miter" stroke-miterlimit="10"/>
    </g>
  </svg>`;
}

function appendMessage(role, text, sources = []) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  if (role === 'assistant') {
    avatar.innerHTML = makeHaloSVG();
  } else {
    avatar.textContent = 'YOU';
  }

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

  // 情報源: ページタイトル全文 + Scrapboxリンク（あれば）
  if (sources.length > 0) {
    const srcWrap = document.createElement('div');
    srcWrap.className = 'msg-sources';
    srcWrap.innerHTML = '<div class="src-label">参照資料</div>';

    sources.slice(0, 5).forEach(src => {
      const title = typeof src === 'string' ? src : src.title;
      const url   = makeScrapboxUrl(title);

      const row = document.createElement('div');
      row.className = 'src-row';

      if (url) {
        const a = document.createElement('a');
        a.className  = 'src-link';
        a.href       = url;
        a.target     = '_blank';
        a.rel        = 'noopener noreferrer';
        a.textContent = title;  // タイトル全文
        row.appendChild(a);
      } else {
        const span = document.createElement('span');
        span.className  = 'src-chip';
        span.textContent = title;  // タイトル全文
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

// ════════════════════════════════════════════
//  ユーティリティ
// ════════════════════════════════════════════

function setStatus(state, text) {
  statusDot.className    = `status-dot ${state}`;
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
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
