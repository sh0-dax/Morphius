import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildModels } from '../js/agent/agent.js';
import { serializeNb, deserializeNb, predictNb } from '../js/agent/nb.js';
import { featuresForText } from '../js/agent/nlp.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const load = (name) => JSON.parse(readFileSync(join(root, name), 'utf8'));
const STOP = load('data/agent/stopwords.json');
const CORPORA = { en: load('data/agent/en.json'), fr: load('data/agent/fr.json'), ar: load('data/agent/ar.json') };

const EPS = 1e-6;

// Probes chosen to exercise normalized Arabic (finally, لا) + accented French.
const PROBES = {
  en: ['what time is it', 'turn off the lights', 'you are amazing'],
  fr: ['bonjour aiface', 'quelle heure est-il', 'éteins la lumière'],
  ar: ['السلام عليكم', 'كم الساعة الآن', 'أطفئ الأنوار', 'أفتح المرآة'],
};

describe('JS <-> Python trainer parity', () => {
  const agent = buildModels(CORPORA, STOP);

  for (const lang of ['en', 'fr', 'ar']) {
    it(lang + ': serialized artifact matches data/agent/weights-' + lang + '.json to 1e-6', () => {
      const js = serializeNb(agent.models[lang]);
      const py = load(`data/agent/weights-${lang}.json`);

      expect(js.schema).toBe(py.schema);
      expect(js.algorithm).toBe(py.algorithm);
      expect(js.classes).toEqual(py.classes);
      expect(js.vocabulary).toEqual(py.vocabulary);

      for (const c of js.classes) {
        expect(Math.abs(js.docsPerClass[c] - (py.docsPerClass[c] || 0))).toBeLessThan(EPS);
        const jc = js.tokenCounts[c];
        const pc = py.tokenCounts[c] || {};
        for (const t of new Set([...Object.keys(jc), ...Object.keys(pc)])) {
          expect(Math.abs((jc[t] || 0) - (pc[t] || 0)), `${lang}.${c}.${t}`).toBeLessThan(EPS);
        }
      }
    });

    it(lang + ': predict parity on probes (Python weights -> JS runtime)', () => {
      const pyModel = deserializeNb(load(`data/agent/weights-${lang}.json`));
      for (const text of PROBES[lang]) {
        const tokens = featuresForText(text, lang, STOP);
        const fromPy = predictNb(pyModel, tokens);
        const fromJs = predictNb(agent.models[lang], tokens);
        expect(fromPy.id, `${text} id`).toBe(fromJs.id);
        expect(Math.abs(fromPy.confidence - fromJs.confidence), `${text} conf`).toBeLessThan(EPS);
        expect(fromPy.scores, `${text} scores`).toEqual(fromJs.scores.map((s) => closeTo(s, EPS)));
      }
    });
  }
});

function closeTo(value, eps) {
  return { asymmetricMatch: (actual) => Math.abs(actual - value) <= eps };
}