/* coi-serviceworker v0.1.7 - https://github.com/gzuidhof/coi-serviceworker */
/* License: MIT */

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) =>
  event.waitUntil(self.clients.claim())
);

async function handleFetch(event) {
  if (event.request.cache === "only-if-cached" && event.request.mode !== "same-origin") {
    return;
  }

  const response = await fetch(event.request).catch(() => null);
  if (!response) return;

  if (
    response.status === 0 ||
    !response.headers.get("cross-origin-embedder-policy")
  ) {
    const newHeaders = new Headers(response.headers);
    newHeaders.set("Cross-Origin-Embedder-Policy", "require-corp");
    newHeaders.set("Cross-Origin-Opener-Policy", "same-origin");

    const moddedResponse = new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: newHeaders,
    });
    return moddedResponse;
  }
  return response;
}

self.addEventListener("fetch", (event) => {
  event.respondWith(handleFetch(event));
});
