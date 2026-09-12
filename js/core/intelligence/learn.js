/**
 * core/intelligence/learn.js
 *
 * Feedback-driven intent learning ("M4"). A deterministic, purely local learner
 * that grows the classifier's vocabulary inside the CURRENT session from:
 *
 *   1. Corrections — "no, I meant X" / "not X, I mean Y" / "لا أقصد X". The
 *      corrected phrase is parsed, classified against the current knowledge, and
 *      its meaningful tokens are weighted toward the taught intent.
 *   2. Repetition  — a novel token seen in >= MIN_REPEATS distinct turns that
 *      consistently accompanies the same intent earns a small learned weight.
 *
 * Learned weights live in an in-memory "extra lexicon" that classifyIntent()
 * merges at CALL time. While intelligence is off or the learner is reset the
 * extra lexicon is empty, so classification is byte-for-byte identical to the
 * static M1/M2 seed (parity guarantee). No network, no model files, no Python.
 *
 * @module core/intelligence/learn
 */

import { extractFeatures, STOPWORDS_EN, STOPWORDS_AR } from './features.js';
import { INTENT_KEYS, INTENT_LEXICON } from './intent.js';

/** Cap per-token learned weight so teaching can't dominate the seed lexicon. */
const MAX_TOKEN_WEIGHT = 3;
/** Fractional learned weight granted when a repeated token is auto-learned. */
const REPETITION_WEIGHT = 0.5;
/** Distinct-turn appearances required before a token is auto-learned. */
const MIN_REPEATS = 3;
/** Recent turns kept for post-correction reclassification. */
const RECENT_TURNS = 8;

/**
 * Correction utterance templates (highest signal first): a rejection is the
 * strongest indicator of teaching intent, a bare "I mean …" the weakest.
 */
const CORRECTION_PATTERNS = [
  // "not X, I mean Y" / "not X, I said Y" (en)
  /\bnot\s+[^.!?;,]+?\s*,\s*(?:no\b\s*)?I\s*(?:mean|meant|said)\s+([^.!?;,]+)/i,
  // "no, I mean Y" / "no I meant Y" (en)
  /\bno\b[^.!?;,]*?(?:,?\s*I\s*)?(?:mean|meant|said)\s+([^.!?;,]+)/i,
  // "I mean Y" / "I said Y" — clarification, only taught when novel or rejected
  /\bI\s*(?:mean|meant|said)\s+([^.!?;,]+)/i,
  // "ليس X بل Y" / "مش X أقصد Y" (ar; no \b — ASCII-only in non-u regexes)
  /(?:ليس|مش)\s+[^.؟;]+?\s*(?:بل|أقصد|اقصد|يعني)\s+([^.؟;]+)/,
  // "لا أقصد Y" (ar)
  /(?:لا|لأ)\s*(?:أقصد|اقصد|يعني|أعني)\s+([^.؟;]+)/,
  // "أقصد Y" / "قلت Y" (ar)
  /(?:أقصد|اقصد|يعني|أعني|قلت)\s+([^.؟;]+)/,
];

const REJECTION_MARKER = /(?:^|[\s,;:.!?؟…'"«»()\[\]{}—–-])(?:no|not|لا|ليس|ليست|لن|مش|مو|بل)(?=[\s,;:.!?؟…'"«»()\[\]{}—–-]|$)/iu;

/** Is this token already meaningful to the static classifier? */
function isSeedToken(token) {
  for (const key of INTENT_KEYS) {
    if (INTENT_LEXICON[key].has(token)) return true;
  }
  return STOPWORDS_EN.has(token) || STOPWORDS_AR.has(token);
}

/**
 * Create the feedback-driven intent learner.
 * @returns {{
 *   parseCorrection: Function, teach: Function, observe: Function,
 *   getExtraLexicon: Function, getLearnedTokens: Function, getStats: Function,
 *   getRecent: Function, reset: Function
 * }}
 */
export function createIntentLearner() {
  /** @type {Record<string, Record<string, number>>} intent -> token -> weight */
  const extra = {};
  /** @type {Record<string, {count: number, intents: Record<string, number>}>} */
  const seen = {};
  /** @type {Array<{text: string, features: object, intent: string}>} */
  const recent = [];
  let utterances = 0;
  let correctionsDetected = 0;
  let correctionsTaught = 0;
  let repetitionsLearned = 0;

  /**
   * Detect a correction ("no, I meant …") and extract the corrected phrase.
   * @param {string} text raw user utterance
   * @returns {{ hadCorrection: boolean, hadRejection: boolean, phrase: string | null }}
   */
  function parseCorrection(text) {
    const raw = String(text == null ? '' : text).trim();
    if (!raw) return { hadCorrection: false, hadRejection: false, phrase: null };
    for (const re of CORRECTION_PATTERNS) {
      const m = re.exec(raw);
      if (m && m[1]) {
        const phrase = m[1].trim();
        if (extractFeatures(phrase).tokenCount > 0) {
          correctionsDetected += 1;
          return { hadCorrection: true, hadRejection: REJECTION_MARKER.test(raw), phrase };
        }
      }
    }
    return { hadCorrection: false, hadRejection: false, phrase: null };
  }

  /**
   * Weight the meaningful tokens of a phrase toward one intent.
   * @param {string} intent a key of INTENT_KEYS (never 'none')
   * @param {string} phrase the corrected phrase
   * @returns {{intent: string, tokens: string[], weight: number} | null}
   */
  function teach(intent, phrase) {
    if (typeof intent !== 'string' || intent === 'none' || !INTENT_KEYS.includes(intent)) return null;
    const tokens = extractFeatures(String(phrase == null ? '' : phrase)).tokens;
    if (!tokens.length) return null;
    if (!extra[intent]) extra[intent] = {};
    const written = [];
    for (const t of tokens) {
      if (isSeedToken(t)) continue;
      const next = Math.min(MAX_TOKEN_WEIGHT, (extra[intent][t] || 0) + 1);
      extra[intent][t] = next;
      written.push(t);
    }
    if (!written.length) return null;
    correctionsTaught += 1;
    return { intent, tokens: written, weight: 1 };
  }

  /**
   * Observe one user turn: keeps the recent-turn ring and auto-learns tokens
   * that repeatedly accompany the same non-'none' intent.
   * @param {string} text raw user utterance
   * @param {object} features pre-computed features (may be the empty feature object)
   * @param {string} intent the intent this turn was classified as
   * @returns {boolean} true when a repetition was learned just now
   */
  function observe(text, features, intent) {
    const f = features || extractFeatures(text);
    utterances += 1;
    recent.push({ text: String(text == null ? '' : text), features: f, intent: String(intent || 'none') });
    if (recent.length > RECENT_TURNS) recent.shift();

    let learned = false;
    for (const t of f.tokens) {
      if (isSeedToken(t)) continue;
      if (!seen[t]) seen[t] = { count: 0, intents: {} };
      const rec = seen[t];
      rec.count += 1;
      if (intent && intent !== 'none') rec.intents[intent] = (rec.intents[intent] || 0) + 1;

      // Majority intent among the appearances so far (deterministic tiebreak by
      // highest count, then declaration order of INTENT_KEYS).
      let majority = 'none';
      let bestCount = 0;
      for (const key of INTENT_KEYS) {
        const n = rec.intents[key] || 0;
        if (n > bestCount) { bestCount = n; majority = key; }
      }
      if (rec.count >= MIN_REPEATS && majority !== 'none' && bestCount >= 2) {
        const target = extra[majority] || (extra[majority] = {});
        const next = Math.min(MAX_TOKEN_WEIGHT, (target[t] || 0) + REPETITION_WEIGHT);
        if (next !== target[t]) {
          target[t] = next;
          repetitionsLearned += 1;
          learned = true;
        }
      }
    }
    return learned;
  }

  /** @returns {Record<string, Record<string, number>>} the merged extra lexicon */
  function getExtraLexicon() {
    return extra;
  }

  /** @returns {Array<{intent: string, token: string, weight: number}>} */
  function getLearnedTokens() {
    const out = [];
    for (const intent of Object.keys(extra)) {
      for (const token of Object.keys(extra[intent])) {
        out.push({ intent, token, weight: extra[intent][token] });
      }
    }
    return out;
  }

  /**
   * @returns {{utterances: number, correctionsDetected: number, correctionsTaught: number, repetitionsLearned: number, learnedTokens: number}}
   */
  function getStats() {
    return {
      utterances,
      correctionsDetected,
      correctionsTaught,
      repetitionsLearned,
      learnedTokens: getLearnedTokens().length,
    };
  }

  /** @returns {Array<{text: string, features: object, intent: string}>} recent turns (newest last) */
  function getRecent() {
    return recent.slice();
  }

  function reset() {
    for (const k of Object.keys(extra)) delete extra[k];
    for (const k of Object.keys(seen)) delete seen[k];
    recent.length = 0;
    utterances = 0;
    correctionsDetected = 0;
    correctionsTaught = 0;
    repetitionsLearned = 0;
  }

  return { parseCorrection, teach, observe, getExtraLexicon, getLearnedTokens, getStats, getRecent, reset };
}