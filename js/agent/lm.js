// ============================================================
// M7 — jitter/variation LM (bigram over template surface form only).
// This is NOT a language model in the LLM sense: it never invents
// factual content. The pipeline is always:
//   FACT -> template -> surface variation
// So augment() only re-uses transition words that already appear in
// the corpus templates, and pickVariant() selects a whole existing
// template (plus a deterministic rotation so the same intent does
// not repeat its last variant immediately).
// Everything is deterministic (seeded by hashString) for tests.
// ============================================================

import { hashString } from './nlp.js';

export function buildBigram(texts) {
  const next = Object.create(null); // word -> Map<word,count>
  const start = Object.create(null); // count of sentence starts
  for (const text of texts || []) {
    const words = String(text).trim().split(/\s+/).filter(Boolean);
    if (!words.length) continue;
    start[words[0]] = (start[words[0]] || 0) + 1;
    for (let i = 0; i < words.length - 1; i++) {
      const a = words[i];
      const b = words[i + 1];
      if (!next[a]) next[a] = Object.create(null);
      next[a][b] = (next[a][b] || 0) + 1;
    }
  }
  return { next, start };
}

// Deterministically pick an index into `list` for a given seed word and a
// rotation key (the intent's previous pick, shifted, to avoid repeats).
export function pickVariantIndex(list, seed, rotationKey) {
  const len = Array.isArray(list) ? list.length : 0;
  if (len === 0) return -1;
  const h = hashString(String(seed || ''));
  return ((h >>> 0) + (Number(rotationKey) || 0)) % len;
}

// Surface connector augmentation: if the chosen template has no terminal
// punctuation and a neutral connector word (from the corpus bigram) exists,
// append it deterministically. Never inserts facts. Returns the template as-is
// in the common case.
export function augmentVariant(template, seed, bigram) {
  const t = String(template || '');
  if (!t || /[.!?…]/.test(t.trim().slice(-1))) return t;
  const words = t.trim().split(/\s+/).filter(Boolean);
  if (!words.length || !bigram || !bigram.next[words[words.length - 1]]) return t;
  const h = hashString(String(seed || ''));
  const candidates = Object.keys(bigram.next[words[words.length - 1]]);
  const pick = candidates[h % candidates.length];
  const suffix = String(pick || '').trim().replace(/^[A-Za-z]/u, (c) => c.toLowerCase());
  if (!suffix) return t;
  return t + ' ' + suffix;
}