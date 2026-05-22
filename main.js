/**
 * main.js - Ankel 統合スクリプト
 * storage.js の内容をインライン統合済み。
 * ES Module import を使わず、<script src> で直接読み込める。
 */

// ════════════════════════════════════════════
//  STORAGE（旧 storage.js をインライン統合）
// ════════════════════════════════════════════

const DB_NAME     = 'ankel_kb';
const DB_VERSION  = 2;
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
        store.createIndex('year',  'year',  { unique: false });
        store.createIndex('tag',   'tag',   { unique: false });
        store.createIndex('title', 'title', { unique: false });
      }
      if (!d.objectStoreNames.contains(STORE_META)) {
        d.createObjectStore(STORE_META, { keyPath: 'key' });
      }
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

function getAllPagesMeta() {
  return dbGetAll(STORE_PAGES).then(rows =>
    rows.map(p => ({ id: p.id, title: p.title, year: p.year, tag: p.tag, description: p.description }))
  );
}

function getAllTags() {
  return dbGetAll(STORE_PAGES).then(rows =>
    [...new Set(rows.map(p => p.tag).filter(Boolean))].sort()
  );
}

function parseTitleMeta(title) {
  const m = title.match(/^(\d{4})-([^-]+)-(.+)$/);
  if (m) return { year: parseInt(m[1]), tag: m[2], description: m[3] };
  const y = title.match(/\b(20\d{2})\b/);
  return { year: y ? parseInt(y[1]) : null, tag: '不明', description: title };
}

async function importScrapboxJSON(json, embedFn, onProgress) {
  const pages = json.pages || [];
  await dbClear(STORE_PAGES);
  let done = 0;
  const BATCH = 10;
  for (let i = 0; i < pages.length; i += BATCH) {
    const batch = pages.slice(i, i + BATCH);
    await Promise.all(batch.map(async (page) => {
      const parsed   = parseTitleMeta(page.title || '');
      const bodyText = (page.lines || []).map(l => typeof l === 'string' ? l : (l.text || '')).join('\n');
      let embedding  = null;
      try { embedding = await embedFn(`${page.title} ${bodyText.slice(0, 400)}`); } catch (_) {}
      await dbPut(STORE_PAGES, {
        title: page.title || '', year: parsed.year, tag: parsed.tag,
        description: parsed.description, body: bodyText,
        embedding: embedding ? Array.from(embedding) : null,
        updatedAt: page.updated || 0,
      });
    }));
    done += batch.length;
    onProgress && onProgress(done, pages.length);
  }
  await dbPut(STORE_META, { key: 'lastImport', value: new Date().toISOString() });
  return done;
}

function cosineSimilarity(a, b) {
  if (a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return (na && nb) ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

async function searchSimilar(queryVec, topK = 5, filters = {}) {
  const all  = await dbGetAll(STORE_PAGES);
  let cands  = all.filter(p => p.embedding);
  if (filters.year) cands = cands.filter(p => p.year === filters.year);
  if (filters.tag && filters.tag !== 'all') cands = cands.filter(p => p.tag === filters.tag);
  const qv = Array.from(queryVec);
  return cands
    .map(p => ({ ...p, score: cosineSimilarity(qv, p.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

async function searchKeyword(query, topK = 5) {
  const all = await dbGetAll(STORE_PAGES);
  const q   = query.toLowerCase();
  return all
    .map(p => ({ ...p, score: (`${p.title} ${p.body}`).toLowerCase().split(q).length - 1 }))
    .filter(p => p.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// ════════════════════════════════════════════
//  WORKER ブリッジ
// ════════════════════════════════════════════

const worker      = new Worker('./ai-worker.js');
let   msgCounter  = 0;
const pendingMsgs = new Map();
let   onTokenCb   = null;

worker.onmessage = (e) => {
  const { id, type, payload } = e.data;
  if (type === 'TOKEN') { onTokenCb && onTokenCb(payload.token); return; }
  const h = pendingMsgs.get(id);
  if (!h) return;
  if (type === 'ERROR')        { pendingMsgs.delete(id); h.reject(new Error(payload.message)); return; }
  if (type === 'PROGRESS' || type === 'EMBED_PROGRESS' || type === 'EMBED_LOADING') { h.onProgress && h.onProgress(payload); return; }
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
//  DOM
// ════════════════════════════════════════════

const importScreen = document.getElementById('import-screen');
const appEl        = document.getElementById('app');
const dropZone     = document.getElementById('drop-zone');
const fileInput    = document.getElementById('json-file-input');
const importStatus = document.getElementById('import-status');
const messagesEl   = document.getElementById('messages');
const userInput    = document.getElementById('user-input');
const sendBtn      = document.getElementById('send-btn');
const modelSelect  = document.getElementById('model-select');
const statusDot    = document.getElementById('status-dot');
const statusText   = document.getElementById('status-text');
const kbStats      = document.getElementById('kb-stats');
const recentPages  = document.getElementById('recent-pages');
const tagFilter    = document.getElementById('tag-filter');
const modelOverlay = document.getElementById('model-overlay');
const overlayText  = document.getElementById('overlay-text');
const overlayBar   = document.getElementById('overlay-bar');
const toastEl      = document.getElementById('toast');

// ════════════════════════════════════════════
//  状態
// ════════════════════════════════════════════

let chatHistory     = [];
let isGenerating    = false;
let currentBubble   = null;
let currentModelKey = 'standard';

// ════════════════════════════════════════════
//  初期化
// ════════════════════════════════════════════

(async () => {
  await openDB();
  const count = await countPages();
  if (count > 0) {
    await launchApp();
  } else {
    importScreen.style.display = 'flex';
  }
})();

// ════════════════════════════════════════════
//  インポート
// ════════════════════════════════════════════

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleJSONFile(file);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleJSONFile(fileInput.files[0]); });

async function handleJSONFile(file) {
  if (!file.name.endsWith('.json')) { setImportStatus('JSONファイルを選択してください', 'error'); return; }
  setImportStatus('読み込み中...', '');
  try {
    const json = JSON.parse(await file.text());
    if (!Array.isArray(json.pages)) { setImportStatus('Scrapbox JSONの形式が正しくありません', 'error'); return; }
    const total = json.pages.length;
    const embedFn = async (text) => sendWorker('EMBED', { text });
    const done = await importScrapboxJSON(json, embedFn, (d, t) => setImportStatus(`${d} / ${t} ページを処理中...`, ''));
    setImportStatus(`✓ ${done}ページを読み込みました`, 'success');
    setTimeout(() => launchApp(), 1200);
  } catch (err) {
    setImportStatus(`エラー: ${err.message}`, 'error');
  }
}

function setImportStatus(msg, cls) {
  importStatus.textContent = msg;
  importStatus.className   = `import-status ${cls}`;
}

// ════════════════════════════════════════════
//  アプリ起動
// ════════════════════════════════════════════

async function launchApp() {
  importScreen.style.display = 'none';
  appEl.classList.add('visible');
  await refreshSidebar();
  await loadModel(currentModelKey);
}

// ════════════════════════════════════════════
//  モデルロード
// ════════════════════════════════════════════

async function loadModel(key) {
  currentModelKey = key;
  setStatus('loading', 'モデル読み込み中...');
  modelOverlay.classList.add('visible');
  sendBtn.disabled = true;
  try {
    await sendWorker('LOAD_MODEL', { modelKey: key }, ({ status, progress }) => {
      overlayText.textContent = status;
      overlayBar.style.width  = `${progress}%`;
    });
    setStatus('ready', '準備完了');
    showToast('モデル読み込み完了');
  } catch (err) {
    setStatus('error', 'ロードエラー');
    showToast(`エラー: ${err.message}`);
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
  userInput.value = ''; userInput.style.height = 'auto';
  isGenerating = true; sendBtn.disabled = true;
  setStatus('loading', '推論中...');

  appendMessage('user', text);
  chatHistory.push({ role: 'user', content: text });

  const context = await retrieveContext(text);
  currentBubble = appendMessage('assistant', '', context.map(c => c.title));
  const bubbleBody = currentBubble.querySelector('.msg-text');

  onTokenCb = (token) => {
    bubbleBody.textContent += token;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  };

  try {
    await sendWorker('GENERATE', { messages: chatHistory, context });
  } catch (err) {
    bubbleBody.textContent = `エラー: ${err.message}`;
  }

  onTokenCb = null;
  chatHistory.push({ role: 'assistant', content: bubbleBody.textContent });
  isGenerating = false; currentBubble = null; sendBtn.disabled = false;
  setStatus('ready', '準備完了');
}

// ════════════════════════════════════════════
//  RAG
// ════════════════════════════════════════════

async function retrieveContext(query) {
  try {
    const vec = await sendWorker('EMBED', { text: `query: ${query}` });
    const res = await searchSimilar(vec, 5);
    if (res.length > 0) return res;
  } catch (_) {}
  return searchKeyword(query, 5);
}

// ════════════════════════════════════════════
//  メッセージ描画
// ════════════════════════════════════════════

function appendMessage(role, text, sources = []) {
  const wrap   = document.createElement('div');
  wrap.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = role === 'user' ? 'YOU' : 'ANK';

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
    sources.slice(0, 3).forEach(title => {
      const chip = document.createElement('span');
      chip.className = 'src-chip';
      chip.textContent = title.length > 30 ? title.slice(0, 30) + '…' : title;
      srcWrap.appendChild(chip);
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
//  サイドバー
// ════════════════════════════════════════════

async function refreshSidebar() {
  const count = await countPages();
  kbStats.innerHTML = `<span>ページ数:</span> ${count}`;
  const tags = await getAllTags();
  tagFilter.innerHTML = '<option value="all">すべて</option>';
  tags.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t; opt.textContent = t;
    tagFilter.appendChild(opt);
  });
  await refreshPageList();
}

async function refreshPageList() {
  const pages    = await getAllPagesMeta();
  recentPages.innerHTML = '';
  const tag      = tagFilter.value;
  const filtered = tag === 'all' ? pages : pages.filter(p => p.tag === tag);
  filtered
    .sort((a, b) => (b.year || 0) - (a.year || 0))
    .slice(0, 50)
    .forEach(p => {
      const item = document.createElement('div');
      item.className = 'page-item';
      item.innerHTML = `<div class="pi-tag">${p.year || '--'} · ${escHtml(p.tag || '?')}</div>
                        <div class="pi-title">${escHtml(p.description || p.title)}</div>`;
      item.addEventListener('click', () => {
        userInput.value = `「${p.title}」について教えてください`;
        userInput.focus();
      });
      recentPages.appendChild(item);
    });
}

tagFilter.addEventListener('change', refreshPageList);

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
