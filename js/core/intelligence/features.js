/**
 * core/intelligence/features.js
 *
 * Lexical feature extraction for short chat utterances. Pure and DOM-free so
 * it can be unit-tested in Node. Produces the flat feature object consumed by
 * the intent classifier (js/core/intelligence/intent.js) and by the user-model
 * follow-up detection. Primed bilingually (en + ar) to match the existing
 * FEELING_LEXICON style used elsewhere in the app.
 *
 * @module core/intelligence/features
 */

export const STOPWORDS_EN = new Set([
  'a', 'an', 'the', 'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'she',
  'it', 'we', 'us', 'they', 'them', 'to', 'and', 'or', 'of', 'in', 'on', 'at',
  'for', 'with', 'from', 'by', 'about', 'into', 'through', 'during', 'is', 'are',
  'am', 'be', 'been', 'being', 'was', 'were', 'do', 'does', 'did', 'will', 'would',
  'this', 'that', 'these', 'those', 'its', 'there', 'here', 'then', 'than', 'so',
  'just', 'very', 'too', 'also', 'if', 'but', 'because', 'as', 'up', 'down',
]);

export const STOPWORDS_AR = new Set([
  'ال', 'و', 'في', 'على', 'عن', 'هذا', 'هذه', 'ذلك', 'تلك', 'هي', 'هو', 'انا',
  'أنا', 'أنت', 'انتم', 'نحن', 'مع', 'من', 'الى', 'إلى', 'عند', 'بين',
]);

export const QUESTION_WORDS = new Set([
  'who', 'what', 'where', 'when', 'why', 'how', 'which', 'any',
  'هل', 'ماذا', 'لماذا', 'كيف', 'أين', 'اين', 'متى', 'ما', 'ليش', 'إيش', 'أي',
]);

export const NEGATION_WORDS = new Set([
  'not', 'no', 'never', 'dont', 'cannot', 'cant', 'wont',
  'لم', 'لا', 'ليس', 'ليست', 'لن', 'أبدا', 'مش', 'مو',
]);

const PUNCT_RE = /[!?؟]+/g;
const SENTENCE_END_RE = /[.!?؟]+/g;
const TOKEN_RE = /[\p{L}\p{N}]+/gu;
/** Normalize English contractions BEFORE tokenization so apostrophe forms
 *  survive as single tokens (don't → dont, can't → cant, won't → wont,
 *  it's → its). Curly/straight apostrophes between letters are dropped;
 *  everything else is left for the tokenizer to split. */
export function normalizeContractions(text) {
  return String(text ?? '')
    .replace(/[‘’‚‛`´ʹʼˊ＇]/g, "'")
    .replace(/([A-Za-z])'([A-Za-z])/g, '$1$2');
}

/**
 * Extract a flat lexical feature vector from a text string.
 *
 * @param {string} text raw user utterance
 * @returns {{
 *   tokens: string[], tokenCount: number, firstWord: string, rawLength: number,
 *   hasQuestionMark: boolean, questionWordCount: number, negationCount: number,
 *   punctuationEnergy: number, sentenceCount: number, avgTokenLength: number,
 *   isQuestion: boolean
 * }}
 */
export function extractFeatures(text) {
  const raw = String(text == null ? '' : text).trim();
  const normalized = normalizeContractions(raw);
  const rawTokens = normalized ? [...normalized.toLowerCase().matchAll(TOKEN_RE)].map((m) => m[0]) : [];

  const fullyQuestionMarked = (raw.match(/[?؟]$/) !== null);
  const questionMarks = (raw.match(PUNCT_RE) || []).join('').length;

  const tokens = rawTokens.filter((t) => !STOPWORDS_EN.has(t) && !STOPWORDS_AR.has(t));

  let questionWordCount = 0;
  let negationCount = 0;
  for (const t of rawTokens) {
    if (QUESTION_WORDS.has(t)) questionWordCount += 1;
    if (NEGATION_WORDS.has(t)) negationCount += 1;
  }

  const sentenceCount = (raw.match(SENTENCE_END_RE) || []).length;
  const avgTokenLength = rawTokens.length
    ? Math.round((rawTokens.reduce((s, t) => s + t.length, 0) / rawTokens.length) * 100) / 100
    : 0;

  return {
    tokens,
    tokenCount: tokens.length,
    firstWord: tokens[0] || '',
    rawLength: raw.length,
    hasQuestionMark: fullyQuestionMarked || questionMarks > 0,
    questionWordCount,
    negationCount,
    punctuationEnergy: questionMarks,
    sentenceCount,
    avgTokenLength,
    isQuestion: fullyQuestionMarked || questionWordCount > 0,
  };
}