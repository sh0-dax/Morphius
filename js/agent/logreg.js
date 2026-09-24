// ============================================================
// M8 — Multinomial Softmax Logistic Regression from scratch
// (ml-from-scratch school: math -> algorithm -> model), a PARALLEL
// intent classifier to nb.js. Pure JS, zero deps, deterministic.
//
// Mathematical specification (SINGLE shared spec; mirrors the Python
// twin in scripts/train_logreg.py to 1e-6 on shipped weights):
//   features  : BagOfWords over normalized tokens (same tokenize func
//               as nb.js, from nlp.js)
//   vocabulary: unique normalized tokens, sorted lexicographically
//   model     : per-class weight vector W[c] {token: weight} + bias[c]
//   logit     : score(c) = bias[c] + sum_{t in docTokens, t in vocab} W[c][t]
//   unknown   : tokens absent from the vocabulary are SKIPPED (contribute 0)
//   objective : softmax cross-entropy over classes
//   optimizer : full-batch gradient descent (deterministic: fixed epochs,
//               fixed lr, fixed L2, zeros init, no shuffle)
//   log base  : natural (Math.log / Python math.log)
//   argmax    : (score desc, class insertion order asc) — stable tie
//   confidence: softmax(top1); margin = top1 - top2
//   serialized: {schema, algorithm, classes, weights{class:{token}}, bias{class},
//                vocabulary[], nDocs}
//
// The SHIPPED weights always come from the Python trainer; the JS side
// keeps the same serialization schema and predict math so the parity test
// (tests/logregParity.test.mjs) passes to 1e-6 on those weights.
// ============================================================

export const LOGREG_MATH_SPEC = {
  schema: 1,
  algorithm: 'multinomial-logreg-1',
  epochs: 200,
  lr: 0.05,
  l2: 1e-4,
  logBase: 'natural',
  unknownToken: 'skip',
  tieBreak: 'score-desc-class-idx-asc',
  init: 'zeros',
  features: 'bag-of-words-normalized',
};

export function createLogReg() {
  return {
    schema: LOGREG_MATH_SPEC.schema,
    algorithm: LOGREG_MATH_SPEC.algorithm,
    classes: [],                // class ids in insertion order
    weights: {},                // class -> { token: weight }
    bias: {},                   // class -> number
    vocab: new Map(),           // token -> index (sealed at sealLogReg)
    nDocs: 0,
    _sealed: false,
  };
}

// docs: array of [className, tokens[]]. Class names stringified.
export function fitLogReg(model, docs) {
  for (const [className, tokens] of docs) {
    partialFitLogReg(model, String(className), tokens, 1);
  }
  sealLogReg(model);
  return model;
}

// Bounded incremental teaching. `weight` is 0..1 so ambiguous
// confirmations contribute less than explicit teachings (mirrors nb.js).
// NOTE: training GD happens in the Python twin; this on-device path only
// accumulates counts into the SAME serialization shape so a later
// re-export stays byte-compatible with the trainer.
export function partialFitLogReg(model, className, tokens, weight) {
  const c = String(className);
  if (model.weights[c] === undefined) {
    model.weights[c] = Object.create(null);
    model.bias[c] = 0;
    model.classes.push(c);
  }
  const w = isFinite(weight) && weight > 0 ? weight : 1;
  model.nDocs += w;
  const counts = model.weights[c];
  for (const t of tokens) {
    if (t) {
      counts[t] = (counts[t] || 0) + w;
      if (!model.vocab.has(t)) model.vocab.set(t, model.vocab.size);
    }
  }
  model._sealed = false; // vocab changed: reseal next time
  return model;
}

// Sort the vocabulary and index it; called once per predict/serialize
// (exactly like nb.js sealNb).
export function sealLogReg(model) {
  if (model._sealed) return model;
  const sorted = [...model.vocab.keys()].sort();
  model.vocab = new Map(sorted.map((t, i) => [t, i]));
  model._sealed = true;
  return model;
}

// tokens can be an array of normalized tokens (public).
export function predictLogReg(model, docTokens) {
  sealLogReg(model);
  const classes = model.classes;
  const scores = classes.map((c) => {
    const Wc = model.weights[c] || {};
    const bias = model.bias[c] || 0;
    let acc = bias;
    for (const t of docTokens) {
      const w = Wc[t];
      if (w !== undefined) acc += w; // unknown -> skipped per spec
    }
    return acc;
  });
  const probs = softmaxLike2(scores);
  let bestIdx = 0;
  for (let i = 1; i < scores.length; i++) {
    // tie-break: score desc, then class insertion order asc (stable)
    if (scores[i] > scores[bestIdx]) bestIdx = i;
  }
  return {
    scores,
    probs,
    id: classes[bestIdx],
    confidence: probs[bestIdx],
    margin: marginLike2(probs),
  };
}

function softmaxLike2(scores) {
  if (!Array.isArray(scores) || scores.length === 0) return [];
  const max = Math.max(...scores);
  let sum = 0;
  const exps = scores.map((s) => { const e = Math.exp(s - max); sum += e; return e; });
  return exps.map((e) => e / sum);
}

function marginLike2(probs) {
  if (probs.length < 2) return 0;
  const sorted = [...probs].sort((a, b) => b - a);
  return sorted[0] - sorted[1];
}

// Serialize to a JSON-safe artifact (schema + algorithm for parity).
export function serializeLogReg(model) {
  sealLogReg(model);
  const classes = model.classes;
  return {
    schema: LOGREG_MATH_SPEC.schema,
    algorithm: LOGREG_MATH_SPEC.algorithm,
    classes: [...classes],
    weights: classes.reduce((o, c) => {
      const src = model.weights[c] || {};
      const dst = {};
      for (const t in src) dst[t] = src[t];
      o[c] = dst;
      return o;
    }, {}),
    bias: classes.reduce((o, c) => { o[c] = model.bias[c] || 0; return o; }, {}),
    vocabulary: [...model.vocab.keys()],
    nDocs: model.nDocs,
  };
}

export function deserializeLogReg(data) {
  const m = createLogReg();
  if (!data || typeof data !== 'object') return m;
  m.classes = Array.isArray(data.classes) ? [...data.classes] : [];
  for (const c of m.classes) {
    m.weights[c] = { ...(data.weights?.[c] || {}) };
    m.bias[c] = Number(data.bias?.[c]) || 0;
  }
  if (Array.isArray(data.vocabulary)) {
    const sorted = [...data.vocabulary].sort();
    m.vocab = new Map(sorted.map((t, i) => [t, i]));
  }
  m.nDocs = Number(data.nDocs) || 0;
  m._sealed = true;
  return m;
}
