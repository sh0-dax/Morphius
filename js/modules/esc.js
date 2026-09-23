/**
 * js/modules/esc.js
 *
 * HTML-escape leaf used by every innerHTML site in the app. Pure and DOM-free
 * so it can be unit-tested in Node. Extracted from js/app.js (P2 slice 1).
 *
 * @module modules/esc
 */

/**
 * Escape a value for safe interpolation into HTML.
 * @param {*} s
 * @returns {string}
 */
export function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
