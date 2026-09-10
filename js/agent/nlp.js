// ============================================================
// M7 — Agent NLP kernels (from scratch, zero dependencies).
// Tokenizer keeps BOTH raw and normalized forms. Features always
// come from the normalized stream; raw is kept for future
// multilingual models (M9+) and never used for scores.
// Shared stopwords live in data/agent/stopwords.json so the
// Python trainer and the browser read the exact same data.
// ============================================================

const AR_MAP = { '\u0623': '\u0627', '\u0625': '\u0627', '\u0622': '\u0627', '\u0649': '\u064A', '\u0629': '\u0647', '\u0626': '\u064A', '\u0624': '\u0648' };
// Arabic combining marks + tatweel + harakat (removed in the normalized stream only).
const AR_MARKS_RE = /[\u0610-\u061A\u0620\u0640\u064B-\u065F\u0670\u06D6-\u06ED\u08D4-\u08E1\uFB1D]/g;

const LATIN_LETTER = /[\p{L}\p{N}]/u;

export function hasArabicScript(text) {
  return /[\u0600-\u06FF]/.test(String(text || ''));
}

// Deterministic hash (FNV-1a 32-bit) over a string -> uint32. Used for
// variant seeding so tests stay stable across runs and platforms.
export function hashString(str) {
  let h = 0x811c9dc5;
  const s = String(str || '');
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) | 0;
    h = h >>> 0;
  }
  return h >>> 0;
}

// Normalize one Arabic char using the AR_MAP; identity otherwise.
function mapArChar(ch) {
  return AR_MAP[ch] || ch;
}

// Normalized token stream for a chunk of text.
//   Latin : lowercase, punctuation stripped, split on anything that is not a
//           letter/digit.
//   Arabic: lowercase, harakat/tatweel removed, أإآ->ا, ى->ي, ة->ه, ئ->ي,
//           ؤ->و, punctuation stripped, split on non-letter/digit.
export function normalizeArabic(text) {
  return String(text || '')
    .replace(AR_MARKS_RE, '')
    .split('')
    .map(mapArChar)
    .join('');
}

export function tokenize(text) {
  const raw = String(text || '');
  const normalized = hasArabicScript(raw) ? normalizeArabic(raw) : raw.toLowerCase();
  const tokens = [];
  let cur = '';
  for (const ch of normalized) {
    if (LATIN_LETTER.test(ch)) {
      cur += ch;
    } else if (cur) {
      tokens.push(cur);
      cur = '';
    }
  }
  if (cur) tokens.push(cur);
  return tokens;
}

// Raw (un-normalized) whitespace words — kept for reference/tests, not scored.
export function tokenizeRaw(text) {
  return String(text || '').split(/\s+/).filter((w) => w.length > 0);
}

// Remove stopwords for `lang`; if that would empty the result, fall back to
// the full token list (short utterances like "hi" must survive).
export function filterStopwords(tokens, lang, stopwords) {
  const list = (stopwords && stopwords[lang]) || [];
  const set = list instanceof Set ? list : new Set(list);
  const kept = tokens.filter((t) => !set.has(t));
  return kept.length > 0 ? kept : tokens;
}

export function featuresForText(text, lang, stopwords) {
  const tokens = tokenize(text);
  return filterStopwords(tokens, lang, stopwords);
}

// Softmax over an array of numbers (bu n * shock 1e300 for overflow safety).
export function softmax(scores) {
  if (!Array.isArray(scores) || scores.length === 0) return [];
  const max = Math.max(...scores);
  let sum = 0;
  const exps = scores.map((s) => {
    const e = Math.exp(s - max);
    sum += e;
    return e;
  });
  return exps.map((e) => e / sum);
}

// Relative model score helpers. These are NOT calibrated probabilities —
// they only order the classes and give a conservative certainty hint.
export function marginOf(probs) {
  if (!Array.isArray(probs) || probs.length < 2) return 0;
  const sorted = [...probs].sort((a, b) => b - a);
  return sorted[0] - sorted[1];
}