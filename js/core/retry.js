/**
 * core/retry.js
 *
 * Abort-aware fetch with exponential backoff for transient failures.
 * Extracted from pure.js into its own module so the streaming error/retry
 * policy has a single, testable home.
 *
 * @module core/retry
 */

/**
 * Wait `ms` milliseconds, or reject with an AbortError if `signal` fires first
 * (so a retry backoff still honours an abort without waiting out the delay).
 * @param {number} ms
 * @param {AbortSignal} [signal]
 * @returns {Promise<void>}
 */
export function abortableDelay(ms, signal) {
  if (!signal) return new Promise((r) => setTimeout(r, ms));
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(createAbortError());
    /** @type {ReturnType<typeof setTimeout>} */
    let t;
    t = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    /** @param {Event} ev */
    function onAbort(ev) {
      clearTimeout(t);
      reject(createAbortError());
    }
    signal.addEventListener('abort', onAbort, { once: true });
  });
}

/** @returns {DOMException} */
function createAbortError() {
  return new DOMException('aborted', 'AbortError');
}

/**
 * fetch() with exponential backoff on transient failures (HTTP 429 and 5xx).
 * Returns the final Response so callers can inspect/throw on it normally.
 *
 * @param {string} url
 * @param {RequestInit} [options]
 * @param {{ retries?: number, backoffMs?: number, signal?: AbortSignal | null }} [cfg]
 * @returns {Promise<Response>}
 */
export async function fetchWithRetry(url, options, { retries = 2, backoffMs = 1000, signal } = {}) {
  const sig = signal || (options && options.signal) || null;
  let attempt = 0;
  for (;;) {
    let res;
    try {
      res = await fetch(url, options);
    } catch (e) {
      if (sig && sig.aborted) throw e;
      if (attempt >= retries) throw e;
      attempt += 1;
      await abortableDelay(backoffMs * 2 ** (attempt - 1), sig);
      continue;
    }
    if (res.ok) return res;
    const retryable = res.status === 429 || res.status >= 500;
    if (!retryable || attempt >= retries || (sig && sig.aborted)) return res;
    attempt += 1;
    await abortableDelay(backoffMs * 2 ** (attempt - 1), sig);
  }
}
