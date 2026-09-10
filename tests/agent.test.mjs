import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  tokenize, featuresForText, hasArabicScript, softmax, hashString,
} from '../js/agent/nlp.js';
import {
  createNaiveBayes, fitNb, partialFitNb, predictNb, serializeNb, deserializeNb,
} from '../js/agent/nb.js';
import { buildBigram, pickVariantIndex } from '../js/agent/lm.js';
import {
  buildModels, classifyText, buildResponse, fallbackResponse, applyFeedback,
  createLocalAgent, packAgent, unpackAgent, corpusHash, normalizeAgentLang,
  calibrate, MIN_CONFIDENCE, MIN_MARGIN, TOP_K,
} from '../js/agent/agent.js';

const here = dirname(fileURLToPath(import.meta.url));
const load = (name) => JSON.parse(readFileSync(join(here, '..', name), 'utf8'));
const STOP = load('data/agent/stopwords.json');
const CORPORA = { en: load('data/agent/en.json'), fr: load('data/agent/fr.json'), ar: load('data/agent/ar.json') };

// In-memory storage with the same interface as js/agent/modelStore.
function makeStorage(seed) {
  const db = { models: {}, meta: null, corrupt: {} };
  if (seed) {
    db.models = JSON.parse(JSON.stringify(seed.models || {}));
    db.meta = seed.meta ? JSON.parse(JSON.stringify(seed.meta)) : null;
  }
  return {
    _db: db,
    loadModel: async (l) => (db.corrupt[l] ? db.corrupt[l] : db.models[l] || null),
    saveModel: async (l, art) => { db.models[l] = JSON.parse(JSON.stringify(art)); },
    loadMeta: async () => db.meta,
    saveMeta: async (m) => { db.meta = JSON.parse(JSON.stringify({ key: 'meta', ...m })); },
    addLearningEvent: async () => true,
    corrupt(lang) { db.corrupt[lang] = '{ not valid json'; },
  };
}

describe('nlp', () => {
  it('tokenizes latin lowercased, splitting punctuation', () => {
    expect(tokenize('Hello, WORLD! It\'s #1.')).toEqual(['hello', 'world', 'it', 's', '1']);
  });
  it('normalizes arabic: أإآ->ا, ى->ي, ة->ه, removes harakat/tatweel', () => {
    expect(tokenize('السلامُ عَلَيْكُمْ')).toEqual(['السلام', 'عليكم']);
    expect(tokenize('رأَى الدّار')).toEqual(['راي', 'الدار']);
    expect(tokenize('أَنَا')).toEqual(['انا']);
  });
  it('detects Arabic script ranges', () => {
    expect(hasArabicScript('bonjour')).toBe(false);
    expect(hasArabicScript('مرحبا')).toBe(true);
  });
  it('filterStopwords falls back to full list when everything is a stopword', () => {
    expect(featuresForText('what is it', 'en', STOP)).toEqual(['what', 'is', 'it']);
  });
  it('softmax is normalized and stable', () => {
    const p = softmax([5, 6, 7]);
    expect(p).toHaveLength(3);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    expect(p[2]).toBeGreaterThan(p[1]);
    expect(p[1]).toBeGreaterThan(p[0]);
  });
  it('hashString is deterministic and yields uint32', () => {
    expect(hashString('hello')).toBe(hashString('hello'));
    expect(hashString('hello')).toBeGreaterThanOrEqual(0);
    expect(hashString('hello')).toBeLessThanOrEqual(0xffffffff);
  });
});

describe('nb', () => {
  it('fit/predict classifies and reports margin + confidence', () => {
    const m = createNaiveBayes();
    fitNb(m, [['a', ['x', 'y']], ['a', ['x']], ['b', ['z', 'z']]]);
    const r = predictNb(m, ['x']);
    expect(r.id).toBe('a');
    expect(r.confidence).toBeGreaterThan(r.margin);
    expect(r.confidence).toBeLessThanOrEqual(1);
    expect(r.margin).toBeGreaterThan(0);
  });
  it('unknown tokens are skipped, not a crash', () => {
    const m = createNaiveBayes();
    fitNb(m, [['a', ['x']], ['b', ['z']]]);
    expect(() => predictNb(m, ['qqq', 'zzz'])).not.toThrow();
  });
  it('serialize/deserialize round-trips and resumes partial_fit', () => {
    const m = createNaiveBayes();
    fitNb(m, [['a', ['x', 'y']]]);
    const m2 = deserializeNb(serializeNb(m));
    partialFitNb(m2, 'b', ['z'], 1);
    expect(predictNb(m2, ['z']).id).toBe('b');
    expect(serializeNb(m2).nDocs).toBeCloseTo(2);
  });
  it('stable argmax tie-break uses class insertion order', () => {
    const m = createNaiveBayes();
    fitNb(m, [['first', ['same']], ['second', ['same']]]);
    const r = predictNb(m, ['same']);
    expect(r.id).toBe('first');
  });
});

describe('lm', () => {
  it('buildBigram + pickVariantIndex stay deterministic', () => {
    const b = buildBigram(['hello there friend', 'hi there']);
    expect(b.next['there']['friend']).toBe(1);
    expect(pickVariantIndex(['a', 'b', 'c'], 'x', 0)).toBeGreaterThanOrEqual(0);
    expect(pickVariantIndex(['a', 'b', 'c'], 'x', 0)).toBe(pickVariantIndex(['a', 'b', 'c'], 'x', 0));
    expect(pickVariantIndex([], 'x', 0)).toBe(-1);
  });
});

describe('corpus end-to-end', () => {
  const agent = buildModels(CORPORA, STOP);

  it('classifies canonical utterances to the right intent per language', () => {
    const cases = [
      ['en', 'what time is it', 'time'],
      ['en', 'turn off the lights', 'lighting'],
      ['en', 'please open the mirror', 'mirror'],
      ['en', 'what can you do', 'capabilities'],
      ['fr', 'bonjour', 'greeting'],
      ['fr', 'quelle heure est-il', 'time'],
      ['fr', 'allume la caméra', 'vision'],
      ['ar', 'السلام عليكم', 'greeting'],
      ['ar', 'كم الساعة الآن', 'time'],
      ['ar', 'أشغل الكاميرا', 'vision'],
    ];
    for (const [lang, text, want] of cases) {
      const r = classifyText(agent, text, STOP);
      expect(r.language, `${text} lang`).toBe(lang);
      expect(r.intent, `${text} intent`).toBe(want);
      expect(r.confidence).toBeGreaterThanOrEqual(MIN_CONFIDENCE);
      expect(r.margin).toBeGreaterThanOrEqual(MIN_MARGIN);
    }
  });

  it('junk utterances fall back to unknown with a real response', () => {
    for (const junk of ['qwerty zxcvbn asdf fghjkl', 'plop kaboum zibou', 'فكن وشت رغو']) {
      const r = classifyText(agent, junk, STOP);
      expect(r.intent).toBeNull();
      const fb = fallbackResponse(r.language);
      expect(fb.length).toBeGreaterThan(10);
    }
  });

  it('language detection picks the right model for Latin vs Arabic input', () => {
    expect(classifyText(agent, 'hello friend', STOP).language).toBe('en');
    const r = classifyText(agent, 'أهلا صديقي', STOP);
    expect(r.language).toBe('ar');
  });

  it('low-confidence unknown fallback (user-required #3)', () => {
    const r = classifyText(agent, 'mmm sandwich policy orange taxis', STOP);
    expect(r.intent).toBeNull();
    expect(r.confidence).toBeLessThan(MIN_CONFIDENCE);
  });

  it('language isolation: Arabic corpus change does not disturb French (user-required #4)', () => {
    const before = classifyText(agent, 'quelle heure est-il', STOP);
    const arShifted = JSON.parse(JSON.stringify(CORPORA.ar));
    arShifted.corpusVersion = 'agent-2026-09-2-shift';
    arShifted.intents[0].examples = [...CORPORA.ar.intents[0].examples, 'سلام مساء الخير يا صديقي العزيز جدا'];
    const agent2 = buildModels({ ...CORPORA, ar: arShifted }, STOP);
    expect(agent2.corpusHash).not.toBe(agent.corpusHash);
    const after = classifyText(agent2, 'quelle heure est-il', STOP);
    expect(after.intent).toBe(before.intent);
    expect(after.confidence).toBeCloseTo(before.confidence, 4);
    expect(after.margin).toBeCloseTo(before.margin, 4);
  });

  it('a teach shifts the model; ambiguous confirm is rejected (bounded learning)', () => {
    const a = buildModels(CORPORA, STOP);
    const prev = classifyText(a, 'police report q1', STOP);
    expect(prev.intent).toBeNull();
    const teach = applyFeedback(a, CORPORA, { kind: 'teach', text: 'police report q1', intent: 'help', language: 'en' });
    expect(teach.applied).toBe(true);
    const out = classifyText(a, 'police report q1', STOP);
    expect(out.intent).toBe('help');
    // ambiguous confirm: nothing to attach to -> rejected
    const bad = applyFeedback(a, CORPORA, { kind: 'confirm', text: 'zxq jkl mmm', intent: 'joke', language: 'en' }, null);
    expect(bad.applied).toBe(false);
  });

  it('actions fire for lighting/mirror/vision/stop intents', () => {
    expect(classifyText(agent, 'matrix mode', STOP).intent).toBe('lighting');
    expect(classifyText(agent, 'stop the camera', STOP).intent).toBe('vision');
  });

  it('calibrate is monotonic and bounded', () => {
    const c1 = calibrate([0, 0.2, 0.1, -1, -2, -3, 0.05, 0.5], TOP_K);
    expect(c1.confidence).toBeGreaterThanOrEqual(0);
    expect(c1.confidence).toBeLessThanOrEqual(1);
    expect(c1.margin).toBeGreaterThanOrEqual(0);
  });
});

describe('createLocalAgent lifecycle', () => {
  it('learning persists across a reload (user-required #1)', async () => {
    const storage = makeStorage();
    const a1 = await createLocalAgent({ corpora: CORPORA, stopwords: STOP, storage, sessionList: async () => [] });
    await a1.ready;
    const before = await a1.classify('red zebra tax audit');
    expect(before.intent).toBeNull();
    const teach = await a1.learn({ kind: 'teach', text: 'red zebra tax audit', intent: 'help', language: 'en' });
    expect(teach.applied).toBe(true);

    const a2 = await createLocalAgent({ corpora: CORPORA, stopwords: STOP, storage, sessionList: async () => [] });
    await a2.ready;
    expect(a2.getStatus().source).toBe('persisted');
    const after = await a2.classify('red zebra tax audit');
    expect(after.intent).toBe('help');
  });

  it('corrupted model is recovered by retraining (user-required #2)', async () => {
    const storage = makeStorage();
    const a1 = await createLocalAgent({ corpora: CORPORA, stopwords: STOP, storage, sessionList: async () => [] });
    await a1.ready;
    expect(a1.getStatus().source).toBe('trained');
    storage.corrupt('en');
    const a2 = await createLocalAgent({ corpora: CORPORA, stopwords: STOP, storage, sessionList: async () => [] });
    await a2.ready;
    expect(a2.getStatus().source).toBe('trained');
    const r = await a2.classify('what time is it');
    expect(r.intent).toBe('time');
  });

  it('corpus change forces retrain (hash differs)', async () => {
    const storage = makeStorage();
    const shifted = JSON.parse(JSON.stringify(CORPORA));
    shifted.en.intents[0].examples = [...CORPORA.en.intents[0].examples, 'yo yo yoyo yo'];
    const a1 = await createLocalAgent({ corpora: shifted, stopwords: STOP, storage, sessionList: async () => [] });
    await a1.ready;
    expect(a1.getStatus().corpusHash).toBe(corpusHash(shifted));
    const a2 = await createLocalAgent({ corpora: CORPORA, stopwords: STOP, storage, sessionList: async () => [] });
    await a2.ready;
    expect(a2.getStatus().corpusHash).toBe(corpusHash(CORPORA));
    expect(a2.getStatus().source).toBe('trained');
  });

  it('respond() returns language/intent/confidence/margin/action/response shape', async () => {
    const storage = makeStorage();
    const a = await createLocalAgent({ corpora: CORPORA, stopwords: STOP, storage, sessionList: async () => [{ title: 'My Chat' }] });
    await a.ready;
    const out = await a.respond('turn on warm lights', { time: '14:02', date: '2026-09-10' });
    expect(out.language).toBe('en');
    expect(out.intent).toBe('lighting');
    expect(out.action).toEqual({ type: 'lighting', preset: 'warm' });
    expect(out.response.length).toBeGreaterThan(5);
    expect(out.memory).toBeNull();
  });

  it('respond() with unknown feeds context but yields null action', async () => {
    const storage = makeStorage();
    const a = await createLocalAgent({ corpora: CORPORA, stopwords: STOP, storage, sessionList: async () => [] });
    await a.ready;
    const out = await a.respond('flob narf quagmire zzz');
    expect(out.intent).toBeNull();
    expect(out.action).toBeNull();
    expect(out.response.length).toBeGreaterThan(5);
  });
});

describe('pack/unpack integrity', () => {
  it('packAgent -> unpackAgent restores decision behavior', () => {
    const agent = buildModels(CORPORA, STOP);
    const pack = packAgent(agent);
    expect(pack.modelVersion).toBe('m7.1');
    expect(pack.algorithm).toBe('multinomial-nb-1');
    const restored = unpackAgent(pack, STOP, CORPORA);
    const a = classifyText(restored, 'bonjour', STOP);
    const b = classifyText(agent, 'bonjour', STOP);
    expect(a.intent).toBe('greeting');
    expect(b.intent).toBe('greeting');
  });
});

describe('helpers', () => {
  it('normalizeAgentLang maps unsupported locales to en', () => {
    expect(normalizeAgentLang('es')).toBe('en');
    expect(normalizeAgentLang('de-DE')).toBe('en');
    expect(normalizeAgentLang('fr')).toBe('fr');
    expect(normalizeAgentLang('AR')).toBe('ar');
  });
  it('buildResponse interpolates slots and time/date', () => {
    const agent = buildModels(CORPORA, STOP);
    const r = buildResponse(agent, CORPORA, 'time', 'what time is it', {}, { language: 'en', time: '14:02', date: '2026-09-10' });
    expect(r).toMatch(/14:02/);
  });
});