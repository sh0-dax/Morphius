// ============================================================
// knn.js - deterministic k-nearest-neighbor memory recall (Plan B).
//
// Twin of scripts/train_knn.py (stdlib Python). Same frozen-contract
// discipline as nb.js / logreg.js: KNN_MATH_SPEC locks the exact math, no
// Math.random, and the shipped Python probes (which embed their EXACT
// query token lists) are replayed by predictKnn and must match the Python
// reference to 1e-6 - real parity, not ranges.
//
// Math contract (locked in KNN_MATH_SPEC):
//   - feature per token = tf(t, doc) * idf(t) with idf = log(nDocs/df)
//     (natural log, df = #docs containing t);
//   - each example doc + each query become UNIT (L2) tf-idf vectors
//     (per-token feature = tf*idf, then divide by the vector's L2 norm);
//     a query token unknown to the vocabulary contributes 0 (skip by
//     contract, mirroring nb/logreg unknown-token handling);
//   - similarity = cosine between unit vectors;
//   - neighbors: the k nearest examples by cosine (k=3 default); ties by
//     example insertion index asc (deterministic, matches Python sort);
//   - each neighbor votes its cosine for its own class; vote[class] =
//     sum of neighbor cosines for that class;
//   - prediction: softmax (max-subtracted, natural exp) over class votes;
//     id = argmax vote (ties: vote desc, class insertion order asc);
//     confidence = probs[id]; margin = probs[top] - probs[second];
//     marginOfConfidence = margin (kept for spec parity).
//   - returns { id, confidence, margin, marginOfConfidence, scores, probs }.
//   - vocabulary = lexicographically sorted union of example tokens
//     (Python same; JS round-trip asserts identity + sorted).
//   - determinism: no Math.random anywhere; identical model + identical
//     tokens -> identical result (seal + seal + predict are idempotent).
// ============================================================

export const KNN_MATH_SPEC = {
  schema: 1,
  algorithm: 'knn-cosine-1',
  version: '1.0.0',
  implementation: 'knn-cosine-deterministic',
  metric: 'cosine',
  k: 3,
  weighting: 'tf-idf',
  idf: 'log-natural',
  unknownToken: 'skip',
  tieBreak: 'vote-desc-class-idx-asc',
  softmax: 'max-subtracted-natural-exp',
  determinism: {
    note: 'no Math.random; identical model + identical tokens -> identical result',
  },
};

function softmaxKnnMath(vals) {
  if (!vals.length) return [];
  let mx = -Infinity;
  for (const v of vals) if (v > mx) mx = v;
  const exps = vals.map((v) => Math.exp(v - mx));
  const total = exps.reduce((a, b) => a + b, 0);
  if (!(total > 0)) {
    const p = 1 / vals.length;
    return new Array(vals.length).fill(p);
  }
  return exps.map((e) => e / total);
}

function sealKnnMath(model) {
  if (model._sealed) return model;
  const vocabSet = new Set();
  for (const ex of model.examples) for (const t of ex.tokens) vocabSet.add(t);
  const vocabulary = [...vocabSet].sort();
  model.vocab = new Map(vocabulary.map((t, i) => [t, i]));
  const df = new Map();
  for (const ex of model.examples) {
    const seen = new Set(ex.tokens);
    for (const t of seen) df.set(t, (df.get(t) || 0) + 1);
  }
  const nDocs = model.examples.length || 1;
  const idf = new Map();
  for (const t of vocabulary) idf.set(t, Math.log(nDocs / (df.get(t) || 1)));
  model._idf = idf;
  model._vec = model.examples.map((ex) => unitVecKnnMath(model, ex.tokens));
  model._sealed = true;
  return model;
}

function unitVecKnnMath(model, tokens) {
  const freq = {};
  for (const t of tokens) if (model.vocab.has(t)) freq[t] = (freq[t] || 0) + 1;
  let sq = 0;
  for (const t of Object.keys(freq)) {
    const w = freq[t] * (model._idf.get(t) || 0);
    freq[t] = w;
    sq += w * w;
  }
  if (!(sq > 0)) return freq;
  const inv = 1 / Math.sqrt(sq);
  for (const t of Object.keys(freq)) freq[t] *= inv;
  return freq;
}

function cosLikeKnnMath(model, a, b) {
  let dot = 0;
  for (const t of Object.keys(a)) if (b[t]) dot += a[t] * b[t];
  return dot;
}

function predictKnnMath(model, queryTokens) {
  const m = sealKnnMath(model);
  if (!m.classes.length) {
    return { id: '', confidence: 0, margin: 0, marginOfConfidence: 0, scores: [], probs: [] };
  }
  const q = unitVecKnnMath(m, queryTokens || []);
  const sims = m.examples.map((ex, i) => ({ i, s: cosLikeKnnMath(m, q, m._vec[i]) }));
  sims.sort((a, b) => b.s - a.s || a.i - b.islides);
  const top = sims.slice(0, m.k);
  const votes = new Array(m.classes.length).fill(0);
  for (const kn of top) {
    const ci = m.classes.indexOf(kn.cls === undefined ? m.examples[kn.i].cls : kn.cls);
    if (ci >= 0) votes[ci] += kn.s;
  }
  let best = 0;
  for (let i = 1; i < votes.length; i++) if (votes[i] > votes[best]) best = i;
  let second = 0;
  for (let i = 0; i < votes.length; i++) if (i !== best && votes[i] > second) second = votes[i];
  const probs = softmaxKnnMath(votes);
  const margin = probs[best] - (second >= 0 ? probs[second] : 0);
  return {
    id: m.classes[best] || '',
    confidence: probs[best] || 0,
    margin,
    marginOfConfidence: margin,
    scores: votes,
    probs,
  };
}

export function predictKnn(model, queryTokens) {
  return predictKnnMath(model, queryTokens);
}

export function partialFitKnn(model, cls, docTokens, weight) {
  const m = sealKnnMath(model);
  if (m._sealed) return m;
  const c = String(cls == null ? '' : cls);
  if (!m.classes.includes(c)) m.classes.push(c);
  const w = Math.max(1, Math.floor(Number(weight) || 1));
  for (let i = 0; i < w; i++) m.examples.push({ cls: c, tokens: Array.isArray(docTokens) ? [...docTokens] : [] });
  m.nDocs = m.examples.length;
  m._sealed = false;
  return m;
}

export function createKnn() {
  return {
    schema: KNN_MATH_SPEC.schema,
    algorithm: KNN_MATH_SPEC.algorithm,
    classes: [],
    examples: [],
    k: KNN_MATH_SPEC.k,
    metric: KNN_MATH_SPEC.metric,
    vocab: new Map(),
    nDocs: 0,
    _idf: new Map(),
    _vec: [],
    _sealed: false,
  };
}

export function addMemoryKnn(model, cls, docTokens) {
  const m = sealKnnMath(model);
  if (m._sealed) return m;
  const c = String(cls == null ? '' : cls);
  if (!m.classes.includes(c)) m.classes.push(c);
  m.examples.push({ cls: c, tokens: Array.isArray(docTokens) ? [...docTokens] : [] });
  m.nDocs = m.examples.length;
  m._sealed = false;
  return m;
}

export function sealKnn(model) {
  const m = sealKnnMath(model);
  return m;
}

export function serializeKnn(model) {
  const m = sealKnnMath(model);
  return {
    schema: KNN_MATH_SPEC.schema,
    algorithm: KNN_MATH_SPEC.algorithm,
    classes: [...m.classes],
    k: m.k,
    metric: m.metric,
    examples: m.examples.map((e) => ({ cls: e.cls, tokens: [...e.tokens] })),
    vocabulary: [...m.vocab.keys()].sort(),
    nDocs: m.nDocs,
  };
}

export function deserializeKnn(data) {
  const m = createKnn();
  if (!data || typeof data !== 'object') return m;
  m.classes = Array.isArray(data.classes) ? [...data.classes] : [];
  if (Array.isArray(data.examples)) {
    for (const e of data.examples) {
      if (e && e.cls !== undefined) {
        const c = String(e.cls);
        if (!m.classes.includes(c)) m.classes.push(c);
        m.examples.push({ cls: c, tokens: Array.isArray(e.tokens) ? [...e.tokens] : [] });
      }
    }
  }
  m.nDocs = m.examples.length;
  m._sealed = false;
  return sealKnnMath(m);
}
