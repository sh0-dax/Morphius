// ============================================================
// M8 — Agent memory (long-term fact store) + deterministic extractors.
//
// The ONLY place the agent talks to history:
//   - session memory: injected `sessionList()` (browser passes
//     chatStore.listSessions; tests pass a fake) -> summary for 'memory_recall'.
//   - fact memory:    stored as {key -> {text, lang, ts}} rows in a long-term
//     store (modelStore.MEMORY_STORE / any storage exposing saveMemoryItem,
//     listMemoryItems, deleteMemoryItem).
//
// Fact recall is strictly EXACT -> FUZZY -> CONFIRM-ON-AMBIGUOUS -> HONEST
// UNKNOWN. Deterministic only. No invented facts (principle: no generation).
//
// extractStoreKV / extractQueryKey / matchFact are pure and exported so the
// runtime AND the test-suite share the same extraction spec.
// ============================================================

import { tokenize, hasArabicScript } from './nlp.js';

// ---- normalization (mirrors nlp.js tokenize: marks stripped + AR_MAP) ----
const AR_MAP = { 'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ى': 'ي', 'ة': 'ه', 'ئ': 'ي', 'ؤ': 'و' };
function normKey(text) {
  if (hasArabicScript(String(text))) {
    const stripped = String(text)
      .replace(/[\u0610-\u061A\u0620\u064B-\u065F\u0670\u06D6-\u06ED\u08D4-\u08E1]/g, '')
      .split('')
      .map((ch) => AR_MAP[ch] || ch)
      .join('');
    return tokenize(stripped).join(' ') || tokenize(String(text)).join(' ') || '';
  }
  return tokenize(text).join(' ');
}

// ---- phrase-prefix matching on normalized strings ----
function matchLead(norm, leads) {
  let best = null;
  for (const raw of leads) {
    const ln = normKey(raw);
    if (!ln) continue;
    if (norm === ln || norm.startsWith(ln + ' ')) {
      if (!best || ln.length > best.norm.length) best = { norm, rest: norm.slice(ln.length).trim() };
    }
  }
  return best;
}

const STORE_LEADS = {
  en: ['remember that', 'remember', 'note that', 'note', 'keep in mind that', 'take note that', 'dont forget that', 'store', 'save that'],
  fr: ['souviens-toi que', 'souviens-toi', 'retiens que', 'retiens', 'mémorise que', 'mémorise', 'note que', 'note', 'garde en mémoire que', 'garde en mémoire'],
  ar: ['تذكر أن', 'تذكر بان', 'تذكر', 'احفظ أن', 'احفظ بان', 'احفظ', 'حفظ', 'لاحظ أن', 'لاحظ', 'سجل'],
};

const QUERY_LEADS = {
  en: ['what did i tell you to remember', 'what did i ask you to remember', 'do you remember what', 'what do you remember about', 'what do you know about', 'what facts do you know about me', 'what should i remember about', 'what did i say about', 'what s', 'what is', 'what was', 'what are', 'tell me', 'wats'],
  fr: ['tu te souviens de quoi', 'tu te souviens de', 'est-ce que tu te souviens de', 'est ce que tu te souviens de', "de quoi tu te souviens", 'qu est ce que tu sais', 'qu est ce que', 'c est quoi', "c'est quoi", 'quelle est', 'quel est', 'quelles sont', 'quels sont', 'rappelle-moi', 'rappelle moi', 'dis-moi', 'dis moi'],
  ar: ['ما الذي تعرف', 'ماذا تعرف', 'ماذا حفظت', 'ماذا تذكرت', 'ما هو', 'ما هي', 'ما', 'ماذا', 'اذكر ما', 'اذكر', 'تذكرني', 'وش'],
};

const GENERIC_KEYS = {
  en: ['me', 'about me', 'about us', ''],
  fr: ['moi', 'sur moi', 'de moi', 'a propos de moi', '', 'sur', 'de'],
  ar: ['عني', 'عنك', ''],
};

const SEPARATORS = {
  en: [['is'], ['are'], ['=']],
  fr: [['est'], ['sont'], ['=']],
  ar: [['هو'], ['هي'], [':'], ['=']],
};

const POSSESSIVE_EN = ['my'];
const POSSESSIVE_FR = ['ma', 'mon', 'mes'];
const QUERY_TRAIL = ['is', 'are', 'est', 'sont', 'هو', 'هي'];
const AR_NAME_KEY = 'اسمي';

// ---- store parsing: "remember that my favorite color is teal" ----
// Keys are canonical (normalized tokens) so recall matches robustly; VALUES
// are taken from the ORIGINAL user wording ("سارة" stays سارة, not ساره).
export function extractStoreKV(text, lang) {
  const raw = String(text || '').trim();
  const norm = normKey(raw);
  if (!norm) return null;
  const rawToks = raw.split(/\s+/).filter(Boolean);
  const normToks = norm.split(' ').filter(Boolean);
  const lead = matchLead(norm, (STORE_LEADS[lang] || STORE_LEADS.en));
  const hadLead = !!lead;
  let toks = normToks;
  let rawSlice = rawToks;
  if (lead) {
    if (!lead.rest) return null; // whole utterance was a lead
    const restLen = lead.rest.split(' ').filter(Boolean).length;
    const offset = normToks.length - restLen;
    toks = normToks.slice(offset);
    rawSlice = rawToks.length === normToks.length ? rawToks.slice(offset) : toks;
  }
  if (!toks.length) return null;

  // Arabic copula-less name fact: "اسمي سارة" (also valid lead-less)
  if (lang === 'ar' && toks[0] === AR_NAME_KEY && toks.length > 1) {
    return { key: AR_NAME_KEY, value: rawSlice.slice(1).join(' ') };
  }
  // Lead-less stores are only recognized when the utterance opens with a
  // possessive — otherwise a QUERY like "ما هو اسمي" would parse as a store.
  const possPattern = lang === 'en' ? POSSESSIVE_EN : (lang === 'fr' ? POSSESSIVE_FR : []);
  if (!hadLead && (lang === 'ar' || !possPattern.includes(toks[0]))) return null;

  const seps = SEPARATORS[lang] || SEPARATORS.en;
  let splitAt = -1;
  for (let i = 0; i < toks.length - 1 && splitAt < 0; i++) {
    for (const sep of seps) {
      let ok = true;
      for (let j = 0; j < sep.length; j++) ok = ok && toks[i + j] === sep[j];
      if (ok) { splitAt = i; break; }
    }
  }
  if (splitAt < 0) return null;

  let keyToks = toks.slice(0, splitAt);
  const valueToks = rawSlice.slice(splitAt + (SEPARATORS[lang] || SEPARATORS.en)[0].length);
  if (!keyToks.length || !valueToks.length) return null;

  if (lang === 'en' && POSSESSIVE_EN.includes(keyToks[0])) keyToks = keyToks.slice(1);
  if (lang === 'fr' && POSSESSIVE_FR.includes(keyToks[0])) keyToks = keyToks.slice(1);

  const key = keyToks.join(' ');
  if (!key) return null;
  return { key, value: valueToks.join(' ') };
}

// ---- query parsing: "what is my favorite color" ----
export function extractQueryKey(text, lang) {
  const norm = normKey(text);
  if (!norm) return { key: '', generic: false };
  const lead = matchLead(norm, QUERY_LEADS[lang] || QUERY_LEADS.en);
  let toks = (lead && lead.rest ? lead.rest : norm).split(' ').filter(Boolean);

  const remPhrase = toks.join(' ');
  if ((GENERIC_KEYS[lang] || []).includes(remPhrase)) return { key: '', generic: true };

  if (lang === 'en' && POSSESSIVE_EN.includes(toks[0])) toks = toks.slice(1);
  if (lang === 'fr' && POSSESSIVE_FR.includes(toks[0])) toks = toks.slice(1);

  while (toks.length && QUERY_TRAIL.includes(toks[toks.length - 1])) toks = toks.slice(0, -1);

  return { key: toks.join(' '), generic: false };
}

// ---- fact retrieval: exact -> fuzzy -> ambiguous -> unknown ----
// Fuzzy similarity = Dice coefficient (2*inter / (|q|+|k|)): a lone extra
// token does not break recall ("color" matches "favorite color"), but two
// genuinely different 2-token keys stay gated behind ambiguity/confirmation.
const FUZZY_MIN_SCORE = 0.6;
const FUZZY_AMBIGUOUS_GAP = 0.2;
export function matchFact(items, queryKey, lang) {
  const qKey = String(queryKey || '').trim();
  if (!qKey) return { found: false, reason: 'no-key' };
  const qNorm = normKey(qKey);
  const q = tokenize(qKey);

  const exact = (Array.isArray(items) ? items : []).find((it) => it && it.key && normKey(it.key) === qNorm);
  if (exact) return { found: true, key: exact.key, value: exact.value, item: exact, via: 'exact' };

  const scored = [];
  for (const it of Array.isArray(items) ? items : []) {
    if (!it || !it.key || typeof it.key !== 'string') continue;
    const k = tokenize(it.key);
    if (!k.length) continue;
    let inter = 0;
    for (const t of k) if (q.includes(t)) inter++;
    const denom = q.length + k.length;
    if (denom > 0) scored.push({ it, score: (2 * inter) / denom });
  }
  scored.sort((a, b) => b.score - a.score);
  const top = scored.filter((s) => s.score >= FUZZY_MIN_SCORE);
  if (!top.length) return { found: false };
  if (top.length > 1 && top[0].score - top[1].score < FUZZY_AMBIGUOUS_GAP) {
    return { found: false, ambiguous: true, candidates: top.slice(0, 3).map((s) => s.it.key) };
  }
  return { found: true, key: top[0].it.key, value: top[0].it.value, item: top[0].it, via: 'fuzzy' };
}

// ---- runtime facade ----
export function createMemory({ sessionList, storage, eventStore, log }) {
  async function sessionSummary() {
    const list = typeof sessionList === 'function' ? await sessionList() : [];
    const safe = (Array.isArray(list) ? list : []).filter(Boolean);
    const latest = safe.length ? safe[0] : null;
    return { count: safe.length, title: (latest && latest.title) || '', latest };
  }

  async function factList() {
    if (!storage || typeof storage.listMemoryItems !== 'function') return [];
    let rows = [];
    try { rows = await storage.listMemoryItems(); } catch (e) { rows = []; }
    if (!Array.isArray(rows)) return [];
    return rows.filter(Boolean).map((r) => ({
      key: r.key || '',
      value: (r.value && typeof r.value.text === 'string' ? r.value.text : r.value || ''),
      lang: (r.value && r.value.lang) || '',
      ts: (r.ts || 0),
    }));
  }

  async function factRecall(text, lang) {
    const items = await factList();
    const parsed = extractQueryKey(text || '', lang);
    if (parsed.generic || !parsed.key) {
      const keys = items.map((i) => i.key).filter(Boolean);
      return { found: false, list: true, keys };
    }
    const m = matchFact(items, parsed.key, lang);
    return { ...m, lang };
  }

  /**
   * @param {{ kind?: string; text?: string; language?: string } | string} query
   * @returns {Promise<{ count?: number; title?: string; latest?: any; found?: boolean; key?: string; value?: string; list?: boolean; keys?: string[]; ambiguous?: boolean; candidates?: string[]; hint?: boolean }>}
   */
  async function recall(query) {
    if (query && typeof query === 'object' && (query.kind === 'fact' || query.kind === 'facts')) {
      return factRecall(query.text, query.language);
    }
    return sessionSummary();
  }

  async function store(event) {
    if (!event) return false;
    if (event.kind === 'fact') {
      const key = String(event.key || '').trim();
      if (!key) return false;
      if (storage && typeof storage.saveMemoryItem === 'function') {
        try {
          await storage.saveMemoryItem(key, { text: String(event.value == null ? '' : event.value), lang: String(event.lang || ''), ts: Date.now() });
          return true;
        } catch (e) {
          if (log) log('agent: memory store failed (' + e.message + ')', 'warn');
          return false;
        }
      }
    }
    if (typeof eventStore === 'function') { await eventStore(event); return true; }
    return false;
  }

  async function listAll() { return factList(); }

  async function forget(key) {
    if (storage && typeof storage.deleteMemoryItem === 'function') {
      try { await storage.deleteMemoryItem(key); return true; } catch (e) { return false; }
    }
    return false;
  }

  return { recall, store, list: listAll, forget };
}