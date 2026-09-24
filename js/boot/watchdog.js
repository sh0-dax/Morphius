// ============================================================
// Boot: module-failure banner, boot watchdog, service-worker registration.
//
// Must run AFTER the module script (js/app.js) so the watchdog can tell that
// the module graph never evaluated. Extracted from index.html so the page
// ships without inline <script> blocks (CSP: script-src has no
// 'unsafe-inline').
// ============================================================
(function () {
  // Detect if modules failed (file:// protocol) and show a helpful message
  window.addEventListener('DOMContentLoaded', function () {
    setTimeout(function () {
      var loadMsg = document.getElementById('loadMsg');
      if (loadMsg && loadMsg.style.display !== 'none' && loadMsg.textContent.indexOf('server') === -1) {
        if (location.protocol === 'file:') {
          loadMsg.textContent = 'ES modules require a web server. Run: npx serve . --port 3000';
          loadMsg.style.display = 'flex';
          loadMsg.style.color = '#ff4757';
        }
      }
    }, 3000);
  });

  // Watchdog: if app.js's module graph never evaluates (a pinned CDN library
  // is unreachable), show the error banner instead of hanging on "Initializing".
  window.__APP_STATIC_READY = 0;
  window.addEventListener('DOMContentLoaded', function () {
    setTimeout(function () {
      var banner = document.getElementById('loadErrorBanner');
      var msgEl = document.getElementById('loadErrorMsg');
      if (!window.__APP_STATIC_READY && banner && msgEl) {
        msgEl.textContent = 'App module failed to load (a CDN/library is unreachable). Check your connection, then Retry, or load the URL with ?nosw to bypass the cache.';
        banner.hidden = false;
      }
    }, 15000);
  });

  if ('serviceWorker' in navigator && !window.__NOSW) {
    window.addEventListener('load', function () {
      navigator.serviceWorker.register('./sw.js', { updateViaCache: 'none' })
        .then(function (reg) {
          if (reg && typeof reg.update === 'function') return reg.update();
        })
        .catch(function (e) { console.warn('SW registration skipped:', e.message); });
      var refreshing = false;
      navigator.serviceWorker.addEventListener('controllerchange', function () {
        if (refreshing) return;
        refreshing = true;
        window.location.reload();
      });
    });
  }
})();
