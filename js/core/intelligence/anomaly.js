/**
 * core/intelligence/anomaly.js
 *
 * Lightweight per-session anomaly detection using a bounded rolling window per
 * metric (NOT cumulative Welford): only the last `windowSize` observations of
 * a metric contribute to its mean/stdev, so "anomalous" always means "far from
 * the recent behaviour", which is what a per-session monitor wants. Pure +
 * DOM-free + unit-testable.
 *
 * @module core/intelligence/anomaly
 */

const EPS = 1e-9;

/**
 * Create an anomaly tracker with one bounded value ring per metric.
 * @param {object} [opts]
 * @param {number} [opts.windowSize] max observations kept per metric (default 50)
 * @param {number} [opts.minSamples] observations required before scoring (default 3)
 * @returns {{ observe: Function, mean: Function, stdev: Function, score: Function, summary: Function, reset: Function }}
 */
export function createAnomalyTracker({ windowSize = 50, minSamples = 3 } = {}) {
  const rings = new Map();

  function values(metric) {
    if (!rings.has(metric)) rings.set(metric, []);
    return rings.get(metric);
  }

  /**
   * Record one observation for a metric, keeping only the rolling window.
   * @param {string} metric
   * @param {number} value
   */
  function observe(metric, value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return;
    const ring = values(metric);
    ring.push(n);
    if (ring.length > windowSize) ring.shift();
    return ring.length;
  }

  /** Rolling-window mean for a metric (0 with no observations). */
  function mean(metric) {
    const ring = values(metric);
    if (!ring.length) return 0;
    return ring.reduce((a, b) => a + b, 0) / ring.length;
  }

  /** Rolling-window sample stdev for a metric (0 with < 2 observations). */
  function stdev(metric) {
    const ring = values(metric);
    if (ring.length < 2) return 0;
    const m = mean(metric);
    const variance = ring.reduce((a, b) => a + (b - m) * (b - m), 0) / (ring.length - 1);
    return Math.sqrt(variance);
  }

  /**
   * Anomaly score = |value - mean| / stdev (z-like). Returns 0 when there are
   * too few observations or the window is degenerate (constant values).
   * @param {string} metric
   * @param {number} value
   * @returns {number}
   */
  function score(metric, value) {
    const ring = values(metric);
    if (ring.length < minSamples) return 0;
    const m = mean(metric);
    const sd = stdev(metric);
    if (sd < EPS) return 0;
    return Number(Math.abs(Number(value) - m).toFixed(3)) / sd;
  }

  /** Read-only summary of every tracked metric. */
  function summary() {
    const out = {};
    for (const [metric, ring] of rings) {
      out[metric] = { count: ring.length, mean: Number(mean(metric).toFixed(3)), stdev: Number(stdev(metric).toFixed(3)) };
    }
    return out;
  }

  function reset() {
    rings.clear();
  }

  return { observe, mean, stdev, score, summary, reset };
}