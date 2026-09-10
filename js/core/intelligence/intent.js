/**
 * core/intelligence/intent.js
 *
 * Deterministic, offline, nearest-centroid-style intent classifier for short
 * chat utterances. No model files, no network, no Python — just a small
 * bilingual (en + ar) seed lexicon scored as a bag of words. Weak or absent
 * signals map to 'none' with a low confidence so downstream behavior stays
 * neutral. Pure + DOM-free + unit-testable.
 *
 * @module core/intelligence/intent
 */

import { extractFeatures } from './features.js';

/** Minimum confidence below which a classification is treated as weak. */
export const MIN_CONFIDENCE = 0.5;

/** Fraction of a maximal single-word match needed before confidence saturates. */
const CONF_RAMP = 3;

/**
 * Intent classes that Morphius knows how to react to. 'none' is always
 * returned when the best match is empty or too weak to matter.
 * @typedef {'greeting'|'farewell'|'help'|'confusion'|'positive'|'negative'|'question'|'none'} IntentKey
 */

/** Bilingual seed lexicon: intent -> set of discriminative tokens. */
export const INTENT_LEXICON = {
  greeting: new Set([
    'hi', 'hello', 'hey', 'howdy', 'salam', 'marhaba', 'ya', 'mr', 'مرحبا',
    'اهلا', 'سلام', 'السلام', 'صباح', 'مساء', 'الخير', 'نورت',
  ]),
  farewell: new Set([
    'bye', 'goodbye', 'farewell', 'tata', 'later', 'see', 'الي', 'اللقاء',
    'وداعا', 'باي', 'معالسلامة', 'تصبح', 'على', 'خير',
  ]),
  help: new Set([
    'help', 'assist', 'support', 'stuck', 'solve', 'fix', 'ساعد', 'ساعدني',
    'ساعدوني', 'تساعدني', 'مساعدة', 'مساعدتي', 'عون', 'مشكلة',
  ]),
  confusion: new Set([
    'confused', 'confusing', 'unclear', 'dunno', 'huh', 'lost', 'bewildered',
    'افهم', 'أفهم', 'فهم', 'فاهم', 'مافهمت', 'مش', 'محتار', 'محتارة', 'حيرة',
  ]),
  positive: new Set([
    'great', 'awesome', 'amazing', 'perfect', 'excellent', 'wonderful',
    'fantastic', 'love', 'nice', 'good', 'cool', 'thanks', 'thank', 'yay',
    'حلو', 'رائع', 'رائعة', 'ممتاز', 'جيد', 'جميل', 'جميلة', 'عظيم', 'شكرا',
    'شكراً', 'تمام', 'حبيت', 'يعجبني', 'بوركت',
  ]),
  negative: new Set([
    'bad', 'terrible', 'awful', 'wrong', 'hate', 'angry', 'upset',
    'disappointed', 'unfortunately', 'سيء', 'سيئ', 'سيئة', 'زعلت', 'زعلان',
    'انزعجت', 'غلط', 'فاشل', 'حزين', 'خاطئ', 'مستاء', 'مستاءة',
  ]),
  question: new Set([
    'what', 'when', 'where', 'why', 'how', 'which', 'any', 'هل', 'ماذا',
    'لماذا', 'كيف', 'أين', 'اين', 'متى', 'ما', 'ليش', 'إيش', 'أي',
  ]),
};

export const INTENT_KEYS = Object.keys(INTENT_LEXICON);

/**
 * Score a token bag against one intent lexicon (exact token equality).
 * @param {string[]} tokens
 * @param {Set<string>} lexicon
 * @returns {number}
 */
export function scoreTokens(tokens, lexicon) {
  let n = 0;
  for (const t of tokens) {
    if (lexicon.has(t)) n += 1;
  }
  return n;
}

/**
 * Classify a short utterance into an intent.
 *
 * Ties are resolved by the declaration order of INTENT_LEXICON (deterministic).
 * Confidence combines (a) the best intent's dominance over the runner-up and
 * (b) a linear ramp on the raw match count, so a single coincidental token
 * yields a weak confidence.
 *
 * @param {string} text
 * @param {object} [features] optional pre-computed features (avoids re-extraction)
 * @returns {{ intent: IntentKey, confidence: number }}
 */
export function classifyIntent(text, features) {
  const f = features || extractFeatures(text);
  const tokens = f ? f.tokens : [];
  const scores = {};
  for (const key of INTENT_KEYS) scores[key] = scoreTokens(tokens, INTENT_LEXICON[key]);

  let bestKey = 'none';
  let best = 0;
  let second = 0;
  for (const key of INTENT_KEYS) {
    const s = scores[key];
    if (s > best) {
      second = best;
      best = s;
      bestKey = key;
    } else if (s > second) {
      second = s;
    }
  }

  if (best === 0) return { intent: 'none', confidence: 0 };

  const relative = best / (best + second);
  const ramp = Math.min(1, best / CONF_RAMP);
  const confidence = Number((relative * ramp).toFixed(3));
  return { intent: /** @type {IntentKey} */ (bestKey), confidence };
}