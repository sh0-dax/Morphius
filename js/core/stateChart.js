/**
 * core/stateChart.js
 *
 * Pure, DOM-free description of the avatar "state chart": the canonical states,
 * the body-CSS class each maps to, and small validation/label helpers used by
 * the app's setState/updateStateBody. Kept framework-free so it's unit-testable.
 *
 * @module core/stateChart
 */

/** Canonical drive states, in the order the UI presents them. */
export const STATES = ['idle', 'listening', 'thinking', 'responding', 'alert'];

/** DOM <body> class applied for a given state. */
export const STATE_BODY_CLASS = {
  responding: 'state-responding',
  listening: 'state-listening',
};

/**
 * The <body> class for the given state ('' for states with no dedicated class).
 * @param {string} state
 * @returns {string}
 */
export function stateBodyClass(state) {
  return state === 'responding' ? 'state-responding'
    : (state === 'listening' ? 'state-listening' : '');
}

/**
 * @param {string} state
 * @returns {boolean} true if `state` is a canonical drive state.
 */
export function isValidState(state) {
  return STATES.indexOf(state) !== -1;
}
