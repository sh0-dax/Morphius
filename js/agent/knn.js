// knn.js - deterministic cosine kNN memory-recall twin (Plan B).
// Twin of scripts/train_knn.py. Same frozen-contract discipline as nb.js /
// logreg.js: the shipped knn-memory-{lang}.json embeds probes whose exact
// token lists are replayed here with predictKnn and must match to 1e-6.
// No Math.random anywhere; results are bit-deterministic given the model.

export const KNN_MATH_SPEC = Object.freeze({
  schema: 1,
  algorithm: 'knn-cosine-1',
  version: '1.0.0',
  metric: 'cosine',
  k: 3,
  idf: 'log(nDocs/df)',
  log: 'natural',
});

function softmaxKnnMath(vals) {
  if (!vals.length) return [];
  const mx = Math.max(...vals);
  const exps = vals.map((v) => Math.exp(v - mx));
  const total = exps.reduce((a, b) => a + b, 0);
  if (!(total > 0)) {
    const p = 1 / vals.length;
    return vals.map(() => p);
  }
  return exps.map((e) => e / total);
}

function tfidfUnitKnnMath(m, tokens) {
  const vocab = new Set(m.vocabulary);
  const freq = new Map();
  for (const t of tokens) if (vocab.has(t)) freq.set(t, (freq.get(t) || 0) + 1);
  const vec = new Map();
  let sq = 0;
  for (const [t, f] of freq) {
    const w = f * m.idf.get(t);
    vec.set(t, w);
    sq += w * w;
  }
  if (!(sq > 0)) return vec;
  const inv = 1 / Math.sqrt(sq);
  for (const [t, w] of vec) vec.set(t, w * inv);
  return vec;
}

function cosLikeKnnMath(m, a, b) {
  let d = 0;
  for (const [t, w] of a) if (b.has(t)) d += w * b.get(t);
  return d;
}

function predictKnnMath(m, queryTokens) {
  const classes = m.classes;
  if (!classes || !classes.length) return { id: '', confidence: 0, margin: 0, scores: [], probs: [] };
  const q = tfidfUnitKnnMath(m, queryTokens);
  const sims = m.examples.map((ex, i) => ({ i, s: cosLikeKnnMath(m, q, tfidfUnitKnnMath(m, ex.tokens)) }));
  sims.sort((a, b) => b.s - a.s || a.i - b.i);
  const top = sims.slice(0, m.k);
  const votes = classes.map(() => 0);
  for (const kn of top) {
    const ci = classes.indexOf(m.examples[kn.i].cls);
    votes[ci] += kn.s;
  }
  const probs = softmaxKnnMath(votes);
  let best = 0;
  for (let i = 1; i < votes.length; i++) if (votes[i] > votes[best]) best = i;
  let secondProb = 0;
  for (let i = 0; i < votes.length; i++) if (i !== best && probs[i] > secondProb) secondProb = probs[i];
  const margin = probs[best] - secondProb;
  return { id: classes[best], confidence: probs[best], margin, scores: votes, probs };
}

export function createKnn() {
  return {
    schema: KNN_MATH_SPEC.schema,
    algorithm: KNN_MATH_SPEC.algorithm,
    classes: [],
    examples: [],
    k: KNN_MATH_SPEC.k,
    metric: KNN_MATH_SPEC.metric,
    sealed: false,
    vocabulary: [],
    idf: new Map(),
    nDocs: 0,
  };
}

export function addMemoryKnn(m, clsId, vi) {
  const memory = {
    cls: String(clsId),
    tokens: vi.map(String),
  };
  m.examples.push(memory);
  if (!m.classes.includes(memory.cls)) m.classes.push(memory.cls);
  return memory;
}

export function partialFitKnn(m, clsId, vi) {
  return addMemoryKnn(m, clsId, vi);
}

export function sealKnn(m) {
  if (m.sealed) return m;
  const vocab = [...new Set([].concat(...m.examples.map((e) => e.tokens)))].sort();
  m.vocabulary = vocab;
  m.nDocs = m.examples.length;
  m.k = KNN_MATH_SPEC.k;
  m.metric = KNN_MATH_SPEC.metric;
  const nDocs = m.nDocs || 1;
  const df = new Map();
  for (const ex of m.examples) for (const t of new Set(ex.tokens)) df.set(t, (df.get(t) || 0) + 1);
  m.idf = new Map(vocab.map((t) => [t, Math.log(nDocs / (df.get(t) || 1))]));
  m.schema = KNN_MATH_SPEC.schema;
  m.algorithm = KNN_MATH_SPEC.algorithm;
  m.sealed = true;
  return m;
}

export function predictKnn(model, queryTokens) {
  return predictKnnMath(model, queryTokens);
}

export function serializeKnn(m) {
  return {
    schema: m.schema,
    algorithm: m.algorithm,
    classes: m.classes.slice(),
    examples: m.examples.map((e) => ({ cls: e.cls, tokens: e.tokens.slice() })),
    vocabulary: (m.vocabulary || [...m.idf.keys()]).slice(),
    nDocs: m.nDocs,
    k: m.k,
    metric: m.metric,
    idf: Object.fromEntries(m.idf),
  };
}

export function deserializeKnn(data) {
  const idf = new Map(Object.entries(data.idf || {}));
  const m = {
    schema: data.schema,
    algorithm: data.algorithm,
    classes: data.classes.slice(),
    examples: (data.examples || []).map((e) => ({ cls: e.cls, tokens: e.tokens.slice() })),
    vocabulary: (data.vocabulary || []).slice(),
    nDocs: data.nDocs ?? data.examples?.length ?? 0,
    k: data.k ?? KNN_MATH_SPEC.k,
    metric: data.metric ?? KNN_MATH_SPEC.metric,
    idf,
    sealed: true,
  };
  if (!m.vocabulary.length) {
    const vocab = [...new Set([].concat(...m.examples.map((e) => e.tokens)))].sort();
    const nDocs = m.examples.length || 1;
    const df = new Map();
    for (const ex of m.examples) for (const t of new Set(ex.tokens)) df.set(t, (df.get(t) || 0) + 1);
    m.vocabulary = vocab;
    m.idf = new Map(vocab.map((t) => [t, Math.log(nDocs / (df.get(t) || 1))]));
  }
  return m;
}