// ============================================================
// M7 — Multinomial Naive Bayes from scratch (mirror of the
// ml-from-scratch school of thought: math -> algorithm -> model).
// Mathematical specification (single spec, shared with the Python
// trainer in scripts/train_agent.py):
//   features  : BagOfWords counts over the normalized tokens
//   vocabulary: unique normalized tokens, sorted lexicographically
//   prior     : P(c) = count(c) / totalSamples
//   likelihood: P(w|c) = (count(w,c) + alpha) / (classTotals(c) + alpha*|V|)
//   alpha     : 1.0 (Laplace smoothing)
//   log base  : natural log (Math.log / Python math.log)
//   unknown   : tokens not in the vocabulary are SKIPPED at predict time
//   argmax    : score(c) = log P(c) + sum_w log P(w|c)
//   tie-break : stable — (score desc, class insertion order asc)
//   confidence: softmax(top1); margin = top1 - top2 (relative scores,
//               NOT calibrated probabilities).
//
// Both implementations must match to 1e-6 (parity test).
// ============================================================

export const NB_MATH_SPEC = {
  schema: 1,
  algorithm: 'multinomial-nb-1',
  alpha: 1.0,
  logBase: 'natural',
  unknownToken: 'skip',
  tieBreak: 'score-desc-class-idx-asc',
  features: 'bag-of-words-normalized',
};

export function createNaiveBayes() {
  return {
    classes: [], // class ids in insertion order
    docsPerClass: {}, // class -> number of training docs
    tokenCounts: {}, // class -> { token: count }
    vocab: new Map(), // token -> index (sorted at predict/serialize time)
    nDocs: 0,
  };
}

// docs: array of [className, tokens[]]. Class names stringified.
export function fitNb(model, docs) {
  for (const [className, tokens] of docs) {
    partialFitNb(model, String(className), tokens, 1);
  }
  sealNb(model);
  return model;
}

// Weighted incremental update. `weight` should be 0..1 so ambiguous
// confirmations contribute less than explicit teachings. Returns model.
export function partialFitNb(model, className, tokens, weight) {
  const c = String(className);
  if (model.docsPerClass[c] === undefined) {
    model.docsPerClass[c] = 0;
    model.tokenCounts[c] = Object.create(null);
    model.classes.push(c);
  }
  const w = isFinite(weight) && weight > 0 ? weight : 1;
  model.docsPerClass[c] += w;
  model.nDocs += w;
  const counts = model.tokenCounts[c];
  for (const t of tokens) {
    if (t) counts[t] = (counts[t] || 0) + w;
    if (!model.vocab.has(t)) model.vocab.set(t, model.vocab.size);
  }
  // Vocabulary/counts changed: any cached per-class log maps are stale.
  model._logCache = null;
  return model;
}

// Sort the vocabulary and index it; called once per inference/serialize.
export function sealNb(model) {
  const sorted = [...model.vocab.keys()].sort();
  model.vocab = new Map(sorted.map((t, i) => [t, i]));
  return model;
}

export function classTotals(model, c) {
  const counts = model.tokenCounts[c] || {};
  let t = 0;
  for (const k in counts) t += counts[k];
  return t;
}

// Log-likelihood features for every vocab token under class c (precomputed
// once per predict call and cached on the model until the next fit).
function ensureLog(model, c) {
  if (model._logCache && model._logCache[c]) return model._logCache[c];
  if (!model._logCache) model._logCache = Object.create(null);
  const V = model.vocab.size;
  const totals = classTotals(model, c);
  const counts = model.tokenCounts[c] || {};
  const denom = totals + NB_MATH_SPEC.alpha * V;
  const out = new Map();
  for (const [t] of model.vocab) out.set(t, Math.log((counts[t] || 0) + NB_MATH_SPEC.alpha) - Math.log(denom));
  model._logCache[c] = out;
  return out;
}

// tokens can be an array or the string key a precomputed log map (internal).
export function predictNb(model, docTokens) {
  sealNb(model);
  const classes = model.classes;
  const totalDocs = model.nDocs || 1;
  const scores = classes.map((c) => {
    const prior = Math.log((model.docsPerClass[c] || 0) / totalDocs);
    const logW = ensureLog(model, c);
    let acc = 0;
    for (const t of docTokens) {
      if (model.vocab.has(t)) acc += logW.get(t); // skip-unknown per spec
    }
    return prior + acc;
  });
  const probs = softmaxLike(scores);
  let bestIdx = 0;
  for (let i = 1; i < scores.length; i++) {
    // tie-break: (score desc, class insertion order asc)
    if (scores[i] > scores[bestIdx]) bestIdx = i;
  }
  return {
    scores,
    probs,
    id: classes[bestIdx],
    confidence: probs[bestIdx],
    margin: marginLike(probs),
  };
}

// Inline small copies to avoid importing nlp.js from this low-level module
// (keeps nb.js self-contained and trivially testable).
function softmaxLike(scores) {
  if (scores.length === 0) return [];
  const max = Math.max(...scores);
  let sum = 0;
  const exps = scores.map((s) => { const e = Math.exp(s - max); sum += e; return e; });
  return exps.map((e) => e / sum);
}
function marginLike(probs) {
  if (probs.length < 2) return 0;
  const sorted = [...probs].sort((a, b) => b - a);
  return sorted[0] - sorted[1];
}

// Serialize to a JSON-safe artifact (counts kept so partial_fit resumes).
export function serializeNb(model) {
  sealNb(model);
  const classes = [...model.classes];
  return {
    schema: NB_MATH_SPEC.schema,
    algorithm: NB_MATH_SPEC.algorithm,
    classes,
    docsPerClass: classes.reduce((o, c) => { o[c] = model.docsPerClass[c] || 0; return o; }, {}),
    tokenCounts: classes.reduce((o, c) => {
      const src = model.tokenCounts[c] || {};
      const dst = {};
      for (const t in src) dst[t] = src[t];
      o[c] = dst;
      return o;
    }, {}),
    vocabulary: [...model.vocab.keys()].sort(),
    nDocs: model.nDocs,
  };
}

export function deserializeNb(data) {
  const m = createNaiveBayes();
  if (!data || typeof data !== 'object') return m;
  m.classes = Array.isArray(data.classes) ? [...data.classes] : [];
  m.docsPerClass = { ...(data.docsPerClass || {}) };
  for (const c of m.classes) {
    m.tokenCounts[c] = { ...(data.tokenCounts?.[c] || {}) };
  }
  if (Array.isArray(data.vocabulary)) {
    for (const t of data.vocabulary.sort()) m.vocab.set(t, m.vocab.size);
  }
  m.nDocs = data.nDocs || 0;
  return m;
}