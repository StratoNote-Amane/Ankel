/**
 * coi-serviceworker.js - Ankel カスタム版
 *
 * GitHub Pages で SharedArrayBuffer を有効にするため
 * Cross-Origin-Embedder-Policy と Cross-Origin-Opener-Policy を付与する。
 *
 * 問題と対策:
 *  - 外部CDN (jsdelivr, HuggingFace) は CORP ヘッダーを返さないため
 *    COEP: require-corp だとブロックされる。
 *  - 対策: COEP を "credentialless" に変更する。
 *    credentialless は CORP ヘッダー不要で外部リソースを読める。
 *    SharedArrayBuffer も有効になる (Chrome 91+, Firefox 119+)。
 *  - opaque response (status=0) は Response コンストラクタに渡さずパススルー。
 */

self.addEventListener('install',  () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  event.respondWith(handleFetch(event.request));
});

async function handleFetch(request) {
  // only-if-cached は same-origin 以外では使えない
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') {
    return fetch(request);
  }

  let response;
  try {
    response = await fetch(request);
  } catch (_) {
    return Response.error();
  }

  // opaque response (status=0) や リダイレクト はそのまま返す
  // Response コンストラクタに status=0 を渡すと RangeError になる
  if (response.status === 0 || response.type === 'opaque' || response.type === 'opaqueredirect') {
    return response;
  }

  // すでに COEP が付いていればそのまま返す
  if (response.headers.get('cross-origin-embedder-policy')) {
    return response;
  }

  // ヘッダーを書き換えたレスポンスを返す
  // COEP: credentialless → CDN リソースに CORP ヘッダーが不要になる
  const newHeaders = new Headers(response.headers);
  newHeaders.set('Cross-Origin-Embedder-Policy', 'credentialless');
  newHeaders.set('Cross-Origin-Opener-Policy',   'same-origin');

  return new Response(response.body, {
    status:     response.status,
    statusText: response.statusText,
    headers:    newHeaders,
  });
}
