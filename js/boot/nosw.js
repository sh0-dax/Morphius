// ============================================================
// Boot: DEV/self-heal kill switch.
//
// Loading the app with ?nosw unregisters any service worker and clears the
// app-shell caches so a stale SW can never pin old code. Must run BEFORE the
// module script (js/app.js) because that module reads window.__NOSW.
//
// Extracted from index.html so the page ships without inline <script> blocks
// (CSP: script-src has no 'unsafe-inline').
// ============================================================
window.__NOSW = new URLSearchParams(location.search).has('nosw');
if (window.__NOSW) {
  (function () {
    try {
      navigator.serviceWorker.getRegistrations().then(function (regs) {
        return Promise.all(regs.map(function (r) { return r.unregister(); }));
      }).catch(function () {});
      caches.keys().then(function (keys) {
        return Promise.all(keys.filter(function (k) { return k.indexOf('aiface-shell') === 0; }).map(function (k) { return caches.delete(k); }));
      }).catch(function () {});
    } catch (e) {}
    document.addEventListener('DOMContentLoaded', function () {
      var chip = document.createElement('div');
      chip.className = 'nosw-chip';
      chip.title = 'Service worker bypassed — PWA/offline disabled. Remove ?nosw from the URL to re-enable.';
      chip.textContent = 'NO-SW // DEV';
      document.body.appendChild(chip);
    });
  })();
  console.warn('[nosw] service worker bypassed — PWA/offline disabled.');
}
