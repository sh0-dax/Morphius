// ============================================================
// Shared model-download progress overlay (HUD).
// Import { modelProgress } from './progress.js'.
// modelProgress(pct, label): pct 0..100, null hides the bar.
// ============================================================

export function modelProgress(pct, label) {
  const overlay = document.getElementById('progressOverlay');
  const bar = document.getElementById('progressFill');
  const text = document.getElementById('progressText');
  if (!overlay || !bar || !text) return;

  if (pct == null) {
    overlay.classList.remove('show');
    return;
  }
  if (pct === -1) {
    overlay.classList.add('show');
    bar.classList.add('indeterminate');
    bar.style.width = '55%';
    text.textContent = (label ? label : 'Loading') + '\u2026';
    return;
  }
  const capped = Math.max(0, Math.min(100, Math.round(pct)));
  overlay.classList.add('show');
  bar.classList.remove('indeterminate');
  bar.style.width = capped + '%';
  text.textContent = (label ? label + ' ' : '') + capped + '%';
}