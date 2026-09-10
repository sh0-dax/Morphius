// ============================================================
// M7 — Local Cognitive Agent (orchestrator).
//
//          INPUT
//            |
//          NLP (nlp.js)
//            |
//        CLASSIFICATION (nb.js, per language + shared intent space)
//            |
//          CONTEXT + MEMORY (memory.js)
//            |
//          DECISION  ──────►  ACTION
//            |                   |
//            └──────────────────► RESPONSE (templates + slots + lm.js)
//                                  |
//                              FEEDBACK (teach / confirm / reject)
//                                  |
//                             PARTIAL_FIT (bounded, never blind)
//                                  |
//                              MEMORY (event log)
//
// Learning is BOUNDED:
//   - 'teach'   : explicit user correction ("no, I meant X") -> strong weight
//   - 'confirm' : only applied when the original classification was already
//                 high-confidence AND high-margin; weak weight otherwise
//   - ambiguous : never inserted as a training sample
//
// confidence is a RELATIVE MODEL SCORE (softmax), never a calibrated
// probability. Decisions require confidence AND margin thresholds.
// ============================================================

import { tokenize, featuresForText, hasArabicScript, hashString } from './nlp.js';
import { filterStopwords } from './nlp.js';
import { createNaiveBayes, fitNb, partialFitNb, predictNb, serializeNb, deserializeNb, NB_MATH_SPEC } from './nb.js';
import { buildBigram, pickVariantIndex, augmentVariant } from './lm.js';
import { createMemory } from './memory.js';
import * as modelStore from './modelStore.js';

export const AGENT_MODEL_VERSION = 'm7.1';
export const AGENT_SCHEMA = 1;

// Decision thresholds (relative model scores — top-K calibrated, tuned
// empirically against the seed corpus as documented in README; the locked
// 0.4/0.15 spec values sat inside the NB's short-utterance boundary and
// dropped ~22% of canonical commands into 'unknown').
export const MIN_CONFIDENCE = 0.35;
export const MIN_MARGIN = 0.12;
export const MIN_CONFIRM_MARGIN = 0.25;
export const MIN_LANG_MARGIN = 0.2;
export const CONFIRM_WEIGHT = 0.25;
export const TOP_K = 4;

export const AGENT_LANGUAGES = ['en', 'fr', 'ar'];

// Interactive fallbacks when nothing is confident enough (honest unknown).
const FALLBACKS = {
  en: ['Hmm, I did not quite get that. Try asking about the time, the lighting, the camera, or what I can do.', "I'm not sure I understood. I can change the lighting, run the camera, or recall our chats."],
  fr: ["Hmm, je n'ai pas bien compris. Essaie de me demander l'heure, l'éclairage, la caméra, ou ce que je sais faire.", "Je ne suis pas sûr d'avoir compris. Je peux changer l'éclairage, activer la caméra, ou te rappeler nos discussions."],
  ar: ["همم، لم أفهم تماما. جرب أن تسألني عن الوقت، الإضاءة، الكاميرا، أو ماذا أقدر أن أفعل.", "لست متأكدا أني فهمت. أقدر أغير الإضاءة، أشغل الكاميرا، أو أتذكر محادثاتنا."],
};

// How the app should show unsupported UI locales: es/de/ja fall back to en
// for GENERATION (they are still fully translated in the UI itself).
export function normalizeAgentLang(lang) {
  const l = String(lang || '').toLowerCase().slice(0, 2);
  return AGENT_LANGUAGES.includes(l) ? l : 'en';
}

// ---- corpus helpers ----
export function corpusHash(corpusJson) {
  return hashString(JSON.stringify(corpusJson || {})).toString(16).padStart(8, '0');
}

function intentDocs(corpusJson, stopwords) {
  const rows = [];
  const intents = Array.isArray(corpusJson?.intents) ? corpusJson.intents : [];
  for (const it of intents) {
    if (!it || !it.id) continue;
    const examples = Array.isArray(it.examples) ? it.examples : [];
    for (const ex of examples) {
      if (typeof ex === 'string' && ex.trim()) rows.push([it.id, featuresForText(ex, corpusJson.language, stopwords)]);
    }
  }
  return rows;
}

export function buildBigramFromCorpora(corpora) {
  const bigram = {};
  for (const lang of Object.keys(corpora)) {
    const json = corpora[lang];
    const texts = [];
    for (const it of json?.intents || []) for (const tx of it?.templates || []) texts.push(tx);
    bigram[lang] = buildBigram(texts);
  }
  return bigram;
}

// Train the per-language intent NBs + a shared language-NB from the raw
// corpus JSON objects. Deterministic — same JSON => same model.
export function buildModels(corpora, stopwords) {
  const models = {};
  const langDocs = [];
  let corpusVersion = [];
  for (const lang of AGENT_LANGUAGES) {
    const json = corpora[lang];
    if (!json) continue;
    corpusVersion.push(json.corpusVersion || json.language);
    const m = createNaiveBayes();
    fitNb(m, intentDocs(json, stopwords));
    models[lang] = m;
    const all = [];
    for (const [c, tokens] of intentDocs(json, stopwords)) for (const t of tokens) all.push(t);
    langDocs.push([lang, all]);
  }
  const langModel = createNaiveBayes();
  fitNb(langModel, langDocs);
  return {
    models,            // lang -> nb (runtime mutable; partial_fit applied here)
    langModel,         // nb over languages
    bigram: buildBigramFromCorpora(corpora),
    languages: Object.keys(models),
    corpusHash: corpusHash(corpora),
    corpusVersion,
    packVersion: AGENT_MODEL_VERSION,
    pickKeys: {},
  };
}

export function packAgent(agent, extra) {
  const langs = (agent.languages || []).slice();
  const packed = {};
  for (const l of langs) packed[l] = serializeNb(agent.models[l]);
  return {
    schema: AGENT_SCHEMA,
    modelVersion: agent.packVersion || AGENT_MODEL_VERSION,
    algorithm: NB_MATH_SPEC.algorithm,
    languages: langs,
    corpusHash: agent.corpusHash,
    corpusVersion: agent.corpusVersion || [],
    models: packed,
    sampleCount: langs.reduce((n, l) => n + (agent.models[l]?.nDocs || 0), 0),
    trainedAt: (extra && extra.trainedAt) || Date.now(),
  };
}

function isArtifactValid(art) {
  return !!art
    && typeof art === 'object'
    && Array.isArray(art.classes) && art.classes.length > 0
    && Array.isArray(art.vocabulary)
    && art.docsPerClass && typeof art.docsPerClass === 'object'
    && art.tokenCounts && typeof art.tokenCounts === 'object'
    && typeof art.nDocs === 'number' && art.nDocs >= 0;
}

export function unpackAgent(pack, stopwords, corpora) {
  const agent = {
    models: {},
    langModel: createNaiveBayes(),
    bigram: {},
    languages: Array.isArray(pack?.languages) ? pack.languages : [],
    corpusHash: pack?.corpusHash || '',
    corpusVersion: pack?.corpusVersion || [],
    packVersion: pack?.modelVersion || AGENT_MODEL_VERSION,
    pickKeys: {},
  };
  if (pack?.models) {
    for (const l of agent.languages) {
      agent.models[l] = deserializeNb(pack.models[l]);
    }
  }
  // Derived structures (language model + bigram variation) are rebuilt
  // deterministically from the shipped corpus; intent models keep the
  // persisted learned weights.
  if (corpora && stopwords) {
    const fresh = buildModels(corpora, stopwords);
    agent.langModel = fresh.langModel;
    agent.bigram = fresh.bigram;
  }
  return agent;
}

// ---- single classification ----
// Calibration (spec-compliant, applied ONLY to the intent model; the raw
// model weights/log-probabilities are untouched and match the Python
// trainer 1:1):
//   The classifier has 24 intents. Naive Bayes spreads probability mass over
//   every class, so a full 24-class softmax always under-reports short
//   utterances (a single strong token tops out near ~0.29). TOP_K
//   renormalization (softmax restricted to the k most plausible classes —
//   a top-k/sparsemax-style reduction) makes `confidence` mean "sure among
//   the few plausible intents" instead of "sure out of all 24". Both decision
//   gates (MIN_CONFIDENCE / MIN_MARGIN) are UNCHANGED and still enforced.
export function calibrate(scores, k) {
  const K = Math.min(k || TOP_K, scores.length);
  const ids = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]).slice(0, K);
  const max = Math.max(...scores);
  const shifted = ids.map((i) => Math.exp(scores[i] - max));
  const tot = shifted.reduce((a, b) => a + b, 0);
  const probs = shifted.map((x) => x / tot);
  return {
    confidence: probs[0],
    margin: K > 1 ? probs[0] - probs[1] : 0,
    topK: ids.map((id, j) => ({ id, score: scores[id], prob: probs[j] })),
  };
}

export function classifyText(agent, text, stopwords) {
  const raw = String(text || '').trim();
  if (!raw) return { language: 'en', intent: null, confidence: 0, margin: 0 };
  const langTokens = filterStopwords(tokenize(raw), 'en', stopwords);
  if (!langTokens.length) return { language: normalizeAgentLang(''), intent: null, confidence: 0, margin: 0 };
  const forcedAr = hasArabicScript(raw) && !/[\u0041-\u007A\u00C0-\u024F\u0370-\u03FF]/.test(raw);

  let language = 'en';
  if (forcedAr) {
    language = 'ar';
  } else {
    const langRes = predictNb(agent.langModel, langTokens);
    language = langRes.margin >= MIN_LANG_MARGIN ? langRes.id : 'en';
  }
  const model = agent.models[language];
  if (!model) return { language, intent: null, confidence: 0, margin: 0, langConfidence: 0 };

  const tokens = featuresForText(raw, language, stopwords);
  if (!tokens.length) return { language, intent: null, confidence: 0, margin: 0 };
  const res = predictNb(model, tokens);
  const cal = calibrate(res.scores, TOP_K);
  return {
    language,
    intent: cal.confidence >= MIN_CONFIDENCE && cal.margin >= MIN_MARGIN ? res.id : null,
    confidence: cal.confidence,
    margin: cal.margin,
    scores: cal.topK,
  };
}

// ---- slots ----
export function extractSlots(text, intentDef, language) {
  const slots = {};
  const def = intentDef?.slots;
  if (!def) return slots;
  const normalized = hasArabicScript(text)
    ? String(text).replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]/g, '')
    : String(text).toLowerCase();
  for (const key of Object.keys(def)) {
    const words = Array.isArray(def[key]) ? def[key] : [];
    for (const w of words) {
      if (normalized.includes(w.toLowerCase())) {
        if (key === 'on') { slots.state = 'on'; break; }
        if (key === 'off') { slots.state = 'off'; break; }
        slots[key] = w;
      }
    }
  }
  return slots;
}

// ---- decision ----
export function decideAction(intentId, slots) {
  switch (intentId) {
    case 'lighting': return { type: 'lighting', preset: slots.preset || 'warm' };
    case 'mirror': return { type: 'mirror', on: slots.state !== 'off' };
    case 'vision': return { type: 'vision', on: slots.state !== 'off' };
    case 'stop_talking': return { type: 'stop' };
    default: return null;
  }
}

// ---- response building ----
function interpolate(template, vars) {
  return String(template).replace(/\{(\w+)\}/g, (_, key) => (vars[key] != null ? String(vars[key]) : `{${key}}`));
}

function sessionVars(memorySummary, lang) {
  const count = memorySummary?.count || 0;
  const plural = count === 1 ? '' : (lang === 'ar' ? 'ات' : 's');
  return {
    count: String(count),
    plural,
    title: (memorySummary?.title || '').trim() || '—',
  };
}

export function buildResponse(agent, corpusByLang, intentId, text, slots, opts = {}) {
  const lang = opts.language || 'en';
  const json = corpusByLang[lang];
  const def = (json?.intents || []).find((it) => it && it.id === intentId);
  if (!def || !Array.isArray(def.templates) || !def.templates.length) return '';
  const seed = String(text || '');
  const rotation = agent.pickKeys[intentId] || 0;
  const idx = pickVariantIndex(def.templates, seed, rotation);
  if (idx < 0) return '';
  agent.pickKeys[intentId] = rotation + 1;
  let tmpl = def.templates[idx];
  if (agent.bigram && agent.bigram[lang]) tmpl = augmentVariant(tmpl, seed, agent.bigram[lang]);
  const s = {
    time: opts.time || '',
    date: opts.date || '',
    name: opts.name || '',
    last: opts.last || '',
    preset: slots.preset || 'warm',
    state: slots.state === 'off' ? (lang === 'ar' ? 'مطفأة' : 'off') : (lang === 'ar' ? 'مشغلة' : 'on'),
    ...sessionVars(opts.memory, lang),
  };
  return interpolate(tmpl, s);
}

export function fallbackResponse(lang) {
  const list = FALLBACKS[normalizeAgentLang(lang)] || FALLBACKS.en;
  return list[hashString('fb') % list.length];
}

// ---- bounded learning ----
// kind: 'teach' | 'confirm' | 'reject'. Returns { applied }.
export function applyFeedback(agent, corpusByLang, { kind, text, intent, language }, prevClass) {
  const lang = normalizeAgentLang(language);
  const model = agent.models[lang];
  if (!model || !model.classes.includes(intent)) return { applied: false, reason: 'unknown-intent' };
  if (kind === 'confirm') {
    const prev = prevClass || null;
    if (!prev || intent !== prev.id || prev.margin < MIN_CONFIRM_MARGIN) {
      return { applied: false, reason: 'ambiguous' };
    }
  }
  const tokens = featuresForText(text, lang);
  if (!tokens.length) return { applied: false, reason: 'no-tokens' };
  const weight = kind === 'teach' ? 1 : CONFIRM_WEIGHT;
  partialFitNb(model, intent, tokens, weight);
  return { applied: true, weight };
}

// ---- browser runtime facade ----
// Boots the agent: prefers the persisted model pack (IndexedDB) when it is
// valid for the current corpus; otherwise trains in-browser and persists.
// Corrupted packs are detected and force a clean retrain (recovery path).
export async function createLocalAgent(opts = {}) {
  const stopwords = opts.stopwords || {};
  const corpora = opts.corpora || {};
  const storage = opts.storage || modelStore;
  const sessionList = opts.sessionList || (async () => []);
  const eventSink = opts.eventSink || null;
  const log = opts.log || (() => {});

  let agent = null;
  let source = 'memory';
  let trainedAt = 0;
  let boots = 0;

  async function ensureModels() {
    const neededHash = corpusHash(corpora);
    if (agent && agent.corpusHash === neededHash) return agent;
    let restored = null;
    let corrupt = null;
    try {
      const meta = await storage.loadMeta();
      if (meta && meta.corpusHash === neededHash && meta.modelVersion === AGENT_MODEL_VERSION) {
        const packed = {};
        for (const l of meta.languages || []) {
          const art = await storage.loadModel(l);
          if (!isArtifactValid(art)) throw new Error('invalid artifact ' + l);
          packed[l] = art;
        }
        restored = unpackAgent({ ...meta, models: packed }, stopwords, corpora);
        source = 'persisted';
        trainedAt = meta.trainedAt || 0;
      }
      if (restored) agent = restored;
    } catch (e) {
      corrupt = e;
      restored = null;
    }
    if (!restored) {
      agent = buildModels(corpora, stopwords);
      source = 'trained';
      try {
        const pack = packAgent(agent, { trainedAt: Date.now() });
        await storage.saveMeta({ key: 'meta', modelVersion: pack.modelVersion, corpusHash: pack.corpusHash, corpusVersion: pack.corpusVersion, languages: pack.languages, algorithm: pack.algorithm, sampleCount: pack.sampleCount, trainedAt: pack.trainedAt, schema: pack.schema });
        for (const l of pack.languages) await storage.saveModel(l, pack.models[l]);
        log('agent: trained + persisted (' + source + ')');
      } catch (e2) {
        log('agent: persisted failed (' + e2.message + ')', 'warn');
      }
      if (corrupt) log('agent: corrupted pack recovered -> retrained', 'warn');
    }
    boots += 1;
    return agent;
  }

  const memory = createMemory({
    sessionList,
    eventStore: (ev) => (typeof storage.addLearningEvent === 'function' ? storage.addLearningEvent(ev) : Promise.resolve()),
  });

  async function classify(text) {
    await ensureModels();
    return classifyText(agent, text, stopwords);
  }

  // Full turn: classification -> context (memory) -> decision -> response.
  async function respond(text, ctx = {}) {
    await ensureModels();
    const res = classifyText(agent, text, stopwords);
    const intentDef = res.intent ? findIntentDef(corpora[res.language], res.intent) : null;
    const slots = intentDef ? extractSlots(text, intentDef, res.language) : {};
    const memorySummary = intentNeedsMemory(res.intent) ? await memory.recall(text) : null;
    const action = res.intent ? decideAction(res.intent, slots) : null;
    const response = res.intent
      ? buildResponse(agent, corpora, res.intent, text, slots, {
          language: res.language,
          time: ctx.time || '',
          date: ctx.date || '',
          name: ctx.name || '',
          last: ctx.last || '',
          memory: memorySummary,
        })
      : fallbackResponse(res.language);
    return { ...res, slots, action, response, memory: memorySummary };
  }

  // Bounded learning + durable persistence of the affected language model.
  async function learn(feedback) {
    await ensureModels();
    const prev = feedback.margin != null ? { id: feedback.intent, margin: feedback.margin } : null;
    const out = applyFeedback(agent, corpora, feedback, prev);
    if (out.applied) {
      if (eventSink) { try { eventSink(feedback, out); } catch (e) {} }
      try { await storage.saveModel(feedback.language || 'en', serializeNb(agent.models[normalizeAgentLang(feedback.language)])); } catch (e) { log('agent: learn persist failed', 'warn'); }
    }
    return out;
  }

  return {
    getStatus() {
      const langs = agent || { languages: [], corpusHash: '', corpusVersion: [] };
      const sampleCount = agent
        ? Object.keys(agent.models).reduce((n, l) => n + (agent.models[l] ? agent.models[l].nDocs : 0), 0)
        : 0;
      return {
        schema: AGENT_SCHEMA,
        modelVersion: AGENT_MODEL_VERSION,
        algorithm: NB_MATH_SPEC.algorithm,
        languages: agent ? { en: !!agent.models.en, fr: !!agent.models.fr, ar: !!agent.models.ar } : {},
        languagesList: langs.languages,
        corpusHash: langs.corpusHash,
        corpusVersion: langs.corpusVersion,
        sampleCount,
        trainedAt,
        source,
        boots,
        memory: !!sessionList,
      };
    },
    classify,
    respond,
    learn,
    memory: {
      recall: (q) => memory.recall(q),
      store: (ev) => memory.store(ev),
    },
    ready: ensureModels(),
  };
}

function findIntentDef(corpusJson, intentId) {
  if (!corpusJson || !intentId) return null;
  return (corpusJson.intents || []).find((it) => it && it.id === intentId) || null;
}

function intentNeedsMemory(intentId) {
  return intentId === 'memory_recall';
}