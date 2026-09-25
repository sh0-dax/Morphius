// kmeans.js - deterministic K-Means + PCA memory organizer (Plan C).
// Twin of scripts/train_kmeans.py. Same frozen-contract discipline as
// nb.js / logreg.js / knn.js: KMEANS_MATH_SPEC locks the math, no
// Math.random anywhere, and the shipped kmeans-pca-{lang}.json embeds
// probes whose EXACT token lists are replayed here with predictKmeans and
// must match the Python reference (clusterId, edge, centroid, projection)
// to 1e-6 - real parity, not ranges.
//
// Math contract (locked in KMEANS_MATH_SPEC):
//   - features = unit (L2) tf-idf vectors (exactly the knn feature model:
//     freq*idf with idf = log(nDocs/df), then L2-normalize); a token
//     unknown to the vocabulary contributes 0 (skip);
//   - PCA: top q=16 principal axes of the centered example vectors
//     (mean-subtracted; deterministic power iteration + deflation on the
//     covariance, start vector = k-th basis vector, fixed iteration cap
//     and cosine-convergence tolerance, mirrors the Python twin);
//   - memory organization: K-Means (k=8) in the q-dim projected space;
//     init = farthest-point seeding (seed 0 = projected point with largest
//     norm, then repeatedly furthest from picked seeds, ties index-asc);
//     Lloyd until assignments stop changing (strict '<' nearest), empty
//     centroids stay put;
//   - predictKmeans: project a query's unit vector into q-dim space, then
//     Euclidean distance to the k centroids; clusterId = argmin (ties via
//     strict '<', lowest index); edge = distance to nearest centroid;
//     centroid = that centroid's coordinates.

export const KMEANS_MATH_SPEC = Object.freeze({
  schema: 1,
  algorithm: 'kmeans-pca-1',
  q: 16,
  k: 8,
  metric: 'cosine-unit-tfidf',
  distance: 'euclidean',
  pca: 'mean-centered-power-iteration-deflation',
  seeds: 'farthest-point-index-asc',
  lloyd: 'strict-less-nearest-index',
  pcaIterations: 100,
  pcaTol: 1e-12,
  lloydIterations: 100,
});

function unitVecKmeansMath(m, tokens) {
  const vocabIndex = m.vocabIndex;
  const idf = m.idf;
  const freq = new Map();
  for (const t of tokens) if (vocabIndex.has(t)) freq.set(t, (freq.get(t) || 0) + 1);
  const v = new Array(m.vocabulary.length).fill(0);
  let sq = 0;
  for (const [t, f] of freq) {
    const w = f * idf.get(t);
    v[vocabIndex.get(t)] = w;
    sq += w * w;
  }
  if (sq > 0) {
    const inv = 1 / Math.sqrt(sq);
    for (const [t] of freq) v[vocabIndex.get(t)] *= inv;
  }
  return v;
}

function covarianceKmeansMath(vs) {
  const d = vs[0] ? vs[0].length : 0;
  const cov = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const v of vs) {
    for (let i = 0; i < d; i++) {
      const vi = v[i];
      if (vi === 0) continue;
      const row = cov[i];
      for (let j = 0; j < d; j++) row[j] += vi * v[j];
    }
  }
  const n = vs.length || 1;
  for (let i = 0; i < d; i++) {
    const row = cov[i];
    for (let j = 0; j < d; j++) row[j] /= n;
  }
  return cov;
}

function dominantAxisKmeansMath(cov, start) {
  const d = cov.length;
  let norm = Math.sqrt(start.reduce((s, x) => s + x * x, 0));
  if (norm === 0) {
    const e = new Array(d).fill(0);
    e[0] = 1;
    return e;
  }
  let v = start.map((x) => x / norm);
  for (let it = 0; it < KMEANS_MATH_SPEC.pcaIterations; it++) {
    const nv = new Array(d).fill(0);
    for (let i = 0; i < d; i++) {
      const row = cov[i];
      let acc = 0;
      for (let j = 0; j < d; j++) acc += row[j] * v[j];
      nv[i] = acc;
    }
    const nn = Math.sqrt(nv.reduce((s, x) => s + x * x, 0));
    if (nn === 0) break;
    for (let i = 0; i < d; i++) nv[i] /= nn;
    let cos = 0;
    for (let i = 0; i < d; i++) cos += v[i] * nv[i];
    v = nv;
    if (cos >= 1 - KMEANS_MATH_SPEC.pcaTol) break;
  }
  return v;
}

function pcaAxesKmeansMath(vs) {
  const cov = covarianceKmeansMath(vs);
  const d = cov.length;
  const axes = [];
  for (let k = 0; k < KMEANS_MATH_SPEC.q; k++) {
    const start = new Array(d).fill(0);
    start[k < d ? k : 0] = 1;
    const a = dominantAxisKmeansMath(cov, start);
    axes.push(a);
    let lam = 0;
    for (let i = 0; i < d; i++) {
      let acc = 0;
      for (let j = 0; j < d; j++) acc += cov[i][j] * a[j];
      lam += a[i] * acc;
    }
    for (let i = 0; i < d; i++) {
      const row = cov[i];
      const ai = a[i];
      for (let j = 0; j < d; j++) row[j] -= lam * ai * a[j];
    }
  }
  return axes;
}

function projectKmeansMath(m, tokens) {
  const v = unitVecKmeansMath(m, tokens);
  const mean = m.mean;
  return m.axes.map((a) => {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += a[i] * (v[i] - mean[i]);
    return s;
  });
}

function l2sqKmeansMath(p) {
  let s = 0;
  for (const x of p) s += x * x;
  return s;
}

function farthestSeedsKmeansMath(points) {
  if (!points.length) return [];
  const k = Math.min(KMEANS_MATH_SPEC.k, points.length);
  const seeds = [0];
  while (seeds.length < k) {
    let bestI = 0;
    let bestD2 = -1;
    for (let i = 0; i < points.length; i++) {
      let d2 = Infinity;
      for (const s of seeds) {
        let d = 0;
        for (let j = 0; j < points[i].length; j++) {
          const diff = points[i][j] - points[s][j];
          d += diff * diff;
        }
        if (d < d2) d2 = d;
      }
      if (d2 > bestD2) {
        bestD2 = d2;
        bestI = i;
      }
    }
    seeds.push(bestI);
  }
  return seeds;
}

function kmeansKmeansMath(points) {
  const n = points.length;
  const k = Math.min(KMEANS_MATH_SPEC.k, n);
  if (!n) return { centroids: [], assign: [] };
  const seeds = farthestSeedsKmeansMath(points);
  const centroids = seeds.map((s) => points[s].slice());
  const assign = new Array(n).fill(0);
  const q = points[0].length;
  for (let it = 0; it < KMEANS_MATH_SPEC.lloydIterations; it++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      let bestC = 0;
      let bestD2 = l2sqKmeansMath(points[i].map((x, j) => x - centroids[0][j]));
      for (let c = 1; c < k; c++) {
        const d2 = l2sqKmeansMath(points[i].map((x, j) => x - centroids[c][j]));
        if (d2 < bestD2) {
          bestD2 = d2;
          bestC = c;
        }
      }
      if (assign[i] !== bestC) {
        assign[i] = bestC;
        changed = true;
      }
    }
    if (!changed) break;
    const newCentroids = Array.from({ length: k }, () => new Array(q).fill(0));
    const counts = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      const c = assign[i];
      counts[c]++;
      for (let j = 0; j < q; j++) newCentroids[c][j] += points[i][j];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) {
        for (let j = 0; j < q; j++) newCentroids[c][j] /= counts[c];
        centroids[c] = newCentroids[c];
      }
    }
  }
  return { centroids, assign };
}

export function predictKmeans(model, queryTokens) {
  if (!model.centroids || !model.centroids.length) {
    return { clusterId: 0, edge: 0, centroid: [], projection: [] };
  }
  const proj = projectKmeansMath(model, queryTokens);
  let bestC = 0;
  let bestD2 = Infinity;
  for (let c = 0; c < model.centroids.length; c++) {
    let d2 = 0;
    const cent = model.centroids[c];
    for (let j = 0; j < proj.length; j++) {
      const diff = proj[j] - cent[j];
      d2 += diff * diff;
    }
    if (d2 < bestD2) {
      bestD2 = d2;
      bestC = c;
    }
  }
  return {
    clusterId: bestC,
    edge: Math.sqrt(bestD2),
    centroid: model.centroids[bestC].slice(),
    projection: proj,
  };
}

export function createKmeans() {
  return {
    schema: KMEANS_MATH_SPEC.schema,
    algorithm: KMEANS_MATH_SPEC.algorithm,
    classes: [],
    vocabulary: [],
    vocabIndex: new Map(),
    idf: new Map(),
    nDocs: 0,
    q: KMEANS_MATH_SPEC.q,
    k: KMEANS_MATH_SPEC.k,
    metric: KMEANS_MATH_SPEC.metric,
    distance: KMEANS_MATH_SPEC.distance,
    mean: [],
    axes: [],
    centroids: [],
    assignments: [],
    sealed: false,
    _pending: [],
  };
}

export function deserializeKmeans(data) {
  const vocabIndex = new Map(data.vocabulary.map((t, i) => [t, i]));
  const idf = new Map(Object.entries(data.idf || {}));
  return {
    schema: data.schema,
    algorithm: data.algorithm,
    classes: (data.classes || []).slice(),
    vocabulary: (data.vocabulary || []).slice(),
    vocabIndex,
    idf,
    nDocs: data.nDocs ?? 0,
    q: data.q ?? KMEANS_MATH_SPEC.q,
    k: data.k ?? KMEANS_MATH_SPEC.k,
    metric: data.metric ?? KMEANS_MATH_SPEC.metric,
    distance: data.distance ?? KMEANS_MATH_SPEC.distance,
    mean: (data.mean || []).slice(),
    axes: (data.axes || []).map((a) => a.slice()),
    centroids: (data.centroids || []).map((c) => c.slice()),
    assignments: (data.assignments || []).slice(),
    sealed: true,
  };
}

export function serializeKmeans(m) {
  return {
    schema: m.schema,
    algorithm: m.algorithm,
    classes: m.classes.slice(),
    vocabulary: m.vocabulary.slice(),
    nDocs: m.nDocs,
    q: m.q,
    k: m.k,
    metric: m.metric,
    distance: m.distance,
    mean: m.mean.slice(),
    axes: m.axes.map((a) => a.slice()),
    centroids: m.centroids.map((c) => c.slice()),
    assignments: (m.assignments || []).slice(),
    idf: Object.fromEntries(m.idf),
  };
}

export function addMemoryKmeans(m, clsId, vi) {
  const tokens = vi.map(String);
  const cls = String(clsId);
  if (!m.classes.includes(cls)) m.classes.push(cls);
  m._pending.push({ cls, tokens });
  return tokens;
}

export function partialFitKmeans(m, clsId, vi) {
  return addMemoryKmeans(m, clsId, vi);
}

export function sealKmeans(m) {
  if (m.sealed) return m;
  const rows = m._pending || [];
  delete m._pending;
  const vocabulary = [...new Set([].concat(...rows.map((r) => r.tokens)))].sort();
  const nDocs = rows.length || 1;
  const df = new Map();
  for (const r of rows) for (const t of new Set(r.tokens)) df.set(t, (df.get(t) || 0) + 1);
  const idf = new Map(vocabulary.map((t) => [t, Math.log(nDocs / (df.get(t) || 1))]));
  const vocabIndex = new Map(vocabulary.map((t, i) => [t, i]));
  const base = { vocabulary, vocabIndex, idf };
  const vecs = rows.map((r) => unitVecKmeansMath(base, r.tokens));
  const d = vocabulary.length;
  const mean = new Array(d).fill(0);
  for (let i = 0; i < d; i++) {
    let s = 0;
    for (const v of vecs) s += v[i];
    mean[i] = s / nDocs;
  }
  const centered = vecs.map((v) => v.map((x, i) => x - mean[i]));
  const axes = pcaAxesKmeansMath(centered);
  m.vocabulary = vocabulary;
  m.vocabIndex = vocabIndex;
  m.idf = idf;
  m.nDocs = rows.length;
  m.mean = mean;
  m.schema = KMEANS_MATH_SPEC.schema;
  m.algorithm = KMEANS_MATH_SPEC.algorithm;
  m.metric = KMEANS_MATH_SPEC.metric;
  m.distance = KMEANS_MATH_SPEC.distance;
  m.q = KMEANS_MATH_SPEC.q;
  m.k = KMEANS_MATH_SPEC.k;
  m.axes = axes;
  const projAll = vecs.map((v) => axes.map((a) => {
    let s = 0;
    for (let i = 0; i < d; i++) s += a[i] * (v[i] - mean[i]);
    return s;
  }));
  const res = kmeansKmeansMath(projAll);
  m.centroids = res.centroids;
  m.assignments = res.assign;
  m.sealed = true;
  return m;
}