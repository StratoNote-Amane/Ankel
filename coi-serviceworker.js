/**
 * coi-serviceworker.js (Ankel カスタム版)
 *
 * 目的: GitHub Pages で SharedArrayBuffer を有効にするために
 *       Cross-Origin-Embedder-Policy: require-corp を付与する。
 *
 * 修正点（オリジナル v0.1.7 からの変更）:
 *   - status=0 の opaque response (CDNなど cross-origin な不透明レスポンス) を
 *     そのままパススルーする。
 *     → Response コンストラクタに status=0 を渡すと RangeError になるため。
 *   - 外部オリジン (CDN / HuggingFace / Google Fonts) へのリクエストは
 *     ヘッダー書き換えを行わずそのまま返す。
 *     COEP require-corp はページ自身のヘッダーで宣言するだけで十分であり、
 *     サブリソースのヘッダーを書き換える必要はない。
 *   - 同一オリジン (GitHub Pages 上の自分のファイル) のみヘッダーを付与する。
 */

self.addEventListener('install',  () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  event.respondWith(handleFetch(event.request));
});

async function handleFetch(request) {
  // only-if-cached は same-origin でないと機能しない → そのまま返す
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') {
    return fetch(request);
  }

  // 外部オリジン (CDN, HuggingFace, Google Fonts 等) はパススルー
  // ヘッダーを書き換えると opaque response (status=0) で RangeError になる
  const isSameOrigin = request.url.startsWith(self.location.origin);
  if (!isSameOrigin) {
    return fetch(request);
  }

  // 同一オリジンのリクエストのみ COEP / COOP ヘッダーを付与
  let response;
  try {
    response = await fetch(request);
  } catch (_) {
    return Response.error();
  }

  // すでに COEP が付いていればそのまま返す
  if (response.headers.get('cross-origin-embedder-policy')) {
    return response;
  }

  // ヘッダーを追加したレスポンスを再構築
  const newHeaders = new Headers(response.headers);
  newHeaders.set('Cross-Origin-Embedder-Policy', 'require-corp');
  newHeaders.set('Cross-Origin-Opener-Policy',   'same-origin');

  return new Response(response.body, {
    status:     response.status,
    statusText: response.statusText,
    headers:    newHeaders,
  });
}
