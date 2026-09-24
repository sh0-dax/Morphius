// ============================================================
// Boot: apply the saved theme before first paint (prevents flash).
//
// Extracted verbatim from index.html so the page ships without inline
// <script> blocks — the CSP in index.html keeps script-src free of
// 'unsafe-inline'. Keep this file dependency-free and synchronous.
// ============================================================
(function () {
  try {
    var s = JSON.parse(localStorage.getItem('aiface_llm_settings'));
    if (s && s.theme && s.theme !== 'blueprint') document.documentElement.className = 'theme-' + s.theme;
  } catch (e) { /* storage blocked or malformed JSON: keep the default theme */ }
})();
