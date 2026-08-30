// Minimal app-shell service worker.
// Strategy: network-first for everything that changes during development
// (index.html, JS, CSS, model manifest, directory listing) so edits appear
// immediately; cache-first for heavy/static assets (.glb models, icon, root
// manifest) with lazy caching. Pinned CDN libraries (three.js, mediapipe,
// transformers, kokoro, three-vrm, web-llm) are cached cache-first so the
// app keeps working through CDN outages and offline. Offline still works via
// cache fallbacks.
// Bump CACHE_NAME on any shell change to invalidate old caches automatically.
const CACHE_NAME = 'aiface-shell-v40';
const CDN_CACHE = 'aiface-cdn-v1';

// Exact pinned CDN resources (substring match against href). These are cached
// cache-first after their first successful load.
const CDN_PINS = [
  'unpkg.com/three@0.183.0',
  'cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14',
  'cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1',
  'cdn.jsdelivr.net/npm/kokoro-js@1.2.1',
  'cdn.jsdelivr.net/npm/@pixiv/three-vrm@3.1.2',
  'cdn.jsdelivr.net/gh/mrdoob/three.js@master/examples/models/gltf/facecap.glb',
  'esm.run/@mlc-ai/web-llm',
  'cdn.jsdelivr.net/npm/onnxruntime-web/dist',
  'cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist',
  'cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd/dist'
];

function isCdnPinned(url) {
  return CDN_PINS.some((pin) => url.href.includes(pin));
}
const SHELL_FILES = [
  './',
  './index.html',
  './css/styles.css',
  './js/app.js',
  './js/pure.js',
  './js/presets.js',
  './js/mirror.js',
  './js/vision.js',
  './js/visionLogic.js',
  './js/localSpeech.js',
  './js/chatStore.js',
  './js/masterBus.js',
  './js/progress.js',
  './i18n/en.json',
  './i18n/ar.json',
  './i18n/fr.json',
  './i18n/de.json',
  './i18n/es.json',
  './i18n/ja.json',
  './manifest.json',
  './assets/icon.svg',
  './assets/icon-192.png',
  './assets/icon-512.png',
  './models/manifest.json',
  './models/yolo26n_int8.onnx'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_NAME && k !== CDN_CACHE).map((k) => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

function isMutable(url) {
  const p = url.pathname;
  const last = p.split('/').pop() || '';
  const modelsManifest = p.endsWith('models/manifest.json');
  const isListing = /\/models\/?$/.test(p);
  const isI18n = /\/i18n\/[^/]+\.json$/.test(p);
  const isCode = last.endsWith('.js') || last.endsWith('.css') || last === 'index.html' || p === '/';
  return modelsManifest || isListing || isI18n || isCode;
}

function isStaticCacheable(url) {
  const p = url.pathname;
  return p.endsWith('.glb') || p.endsWith('icon.svg') || p.endsWith('.png') || p.endsWith('manifest.json');
}

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  const sameOrigin = url.origin === self.location.origin;

  // ?nosw dev mode: always hit the network so the kill-switch page is fresh,
  // even while an older cache-first service worker is still controlling the tab.
  if (url.search.includes('nosw')) {
    event.respondWith(fetch(event.request, { cache: 'no-store' }));
    return;
  }

  // Pinned CDN libraries: cache-first so the app survives CDN outages and works
  // offline after the first successful load. Background refresh updates the copy.
  if (!sameOrigin && isCdnPinned(url)) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        const network = fetch(event.request)
          .then((res) => {
            if (res && res.ok) {
              const clone = res.clone();
              caches.open(CDN_CACHE).then((cache) => cache.put(event.request, clone));
            }
            return res;
          })
          .catch(() => cached);
        if (cached) {
          network.catch(() => {});
          return cached;
        }
        return network;
      })
    );
    return;
  }

  if (!sameOrigin) return; // Other cross-origin (LLM APIs, fonts, etc.) go untouched.

  if (isMutable(url)) {
    // Network-first: always get the latest code/listings when online,
    // fall back to the cached copy (then the app shell) when offline.
    event.respondWith(
      fetch(event.request, { cache: 'no-store' })
        .then((res) => {
          if (res && res.ok) {
            const clone = res.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return res;
        })
        .catch(() => caches.match(event.request).then((c) => c || caches.match('./index.html')))
    );
    return;
  }

  if (isStaticCacheable(url)) {
    // Cache-first with lazy runtime caching (large/static assets).
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((res) => {
          if (res && res.ok) {
            const clone = res.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return res;
        });
      })
    );
    return;
  }

  // Everything else (LLM provider APIs, WebLLM model shards, fonts, three.js, etc.)
  // goes to the network as-is; the service worker stays out of the way.
});
