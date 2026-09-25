#!/usr/bin/env python3
"""train_kmeans.py - deterministic K-Means + PCA memory organizer (Plan C).

Twin of js/agent/kmeans.js. Same frozen-contract discipline as nb.py /
train_logreg.py / train_knn.py: stdlib only, no third party, no random
anywhere (deterministic farthest-point seeding + fixed-power-iteration
PCA), and the shipped artifact kmeans-pca-{lang}.json embeds probes whose
EXACT token lists are replayed in JS with predictKmeans and must match the
Python reference (projection, clusterId, edge, centroid) to 1e-6 - real
parity, not ranges.

The corpus source is THE SAME as the other twins:
  data/agent/{en,fr,ar}.json (intents[] = {id, examples:[string], ...})

Math contract (locked in KMEANS_MATH_SPEC in the JS twin):
  - features = unit (L2) tf-idf vectors of each intent example (exactly the
    knn feature model: freq*idf, idf=log(nDocs/df), then L2-normalize);
  - PCA: the example-vector covariance is decomposed by deterministic power
    iteration + deflation into the top q=16 principal axes (all centered by
    subtracting the example mean vector);
  - memory organization: K-Means (k=8) runs in the q-dimensional projected
    space; initialization is DIET - like farthest-point seeding (cluster 0
    = the projected point with largest norm, then repeatedly the projected
    point furthest from already-picked seeds; ties by index asc); then
    Lloyd iterations until no assignment changes (centroids = mean of their
    points; empty cluster stays put; assignment ties by nearest index using
    strict '<').

Lifecycle:
  1. run:  python scripts/train_kmeans.py   (writes data/agent/kmeans-pca-*)
  2. test: npx vitest run tests/kmeansParity.test.mjs
  3. predict in JS with deserializeKmeans + predictKmeans.
"""
import json
import math
from pathlib import Path

from train_logreg import tokenize

DATA = Path(__file__).resolve().parents[1] / "data" / "agent"
OUT = DATA
LANGUAGES = ("en", "fr", "ar")
SCHEMA = 1
ALGORITHM = "kmeans-pca-1"
Q = 16
KMEANS_K = 8
PCA_MAX_ITER = 100
PCA_TOL = 1e-12
LLOYD_MAX_ITER = 100

PROBES = {
    "en": [
        ["what", "time", "is", "it"],
        ["turn", "off", "the", "lights"],
        ["you", "are", "amazing"],
        ["hello", "there"],
        ["good", "morning"],
    ],
    "fr": [
        ["bonjour", "aiface"],
        ["quelle", "heure", "est", "il"],
        ["merci", "beaucoup"],
        ["eteins", "la", "lumiere"],
        ["a", "bientot"],
    ],
    "ar": [
        ["as-salamu", "alaykum"],
        ["kam", "assaa", "al-aan"],
        ["shukran", "jazilan"],
        ["afi", "al-anwar"],
        ["ma", "ismi"],
    ],
}


def intent_docs(corpus):
    rows = []  # list of [classId, tokens]
    for it in (corpus.get("intents", []) if isinstance(corpus, dict) else corpus) or []:
        cid = str(it.get("id", ""))
        if not cid:
            continue
        for ex in it.get("examples", []) or []:
            if isinstance(ex, str) and ex.strip():
                toks = [t for t in tokenize(ex) if t]
                if toks:
                    rows.append([cid, toks])
    return rows


def unit_vec(vocab_index, idf, tokens):
    freq = {}
    for t in tokens:
        if t in vocab_index:
            freq[t] = freq.get(t, 0) + 1
    v = [0.0] * len(vocab_index)
    sq = 0.0
    for t, f in freq.items():
        w = float(f) * idf[t]
        v[vocab_index[t]] = w
        sq += w * w
    if sq > 0:
        inv = 1.0 / math.sqrt(sq)
        for t in freq:
            v[vocab_index[t]] *= inv
    return v


def covariance(vs):
    d = len(vs[0]) if vs else 0
    cov = [[0.0] * d for _ in range(d)]
    for v in vs:
        for i in range(d):
            vi = v[i]
            if vi == 0.0:
                continue
            row = cov[i]
            for j in range(d):
                row[j] += vi * v[j]
    n = len(vs) or 1
    for i in range(d):
        row = cov[i]
        for j in range(d):
            row[j] /= n
    return cov


def dominant_axis(cov, start, max_iter=PCA_MAX_ITER, tol=PCA_TOL):
    d = len(cov)
    norm = math.sqrt(sum(x * x for x in start))
    if norm == 0.0:
        return [1.0 if i == 0 else 0.0 for i in range(d)]
    v = [x / norm for x in start]
    for _ in range(max_iter):
        nv = [0.0] * d
        for i in range(d):
            row = cov[i]
            acc = 0.0
            for j in range(d):
                acc += row[j] * v[j]
            nv[i] = acc
        nn = math.sqrt(sum(x * x for x in nv))
        if nn == 0.0:
            break
        nv = [x / nn for x in nv]
        cos = sum(a * b for a, b in zip(v, nv))
        v = nv
        if cos >= 1.0 - tol:
            break
    return v


def pca_axes(vs, q=Q):
    cov = covariance(vs)
    d = len(cov)
    axes = []
    for k in range(q):
        start = [0.0] * d
        if k < d:
            start[k] = 1.0
        else:
            start[0] = 1.0
        a = dominant_axis(cov, start)
        axes.append(a)
        lam = 0.0
        for i in range(d):
            acc = 0.0
            for j in range(d):
                acc += cov[i][j] * a[j]
            lam += a[i] * acc
        for i in range(d):
            row = cov[i]
            ai = a[i]
            for j in range(d):
                row[j] -= lam * ai * a[j]
    return axes


def project(v, mean, axes):
    return [sum(a[i] * (v[i] - mean[i]) for i in range(len(v))) for a in axes]


def l2sq(p):
    return sum(x * x for x in p)


def farthest_seeds(points, k):
    if not points:
        return []
    n = len(points)
    k = min(k, n)
    seeds = [0]
    while len(seeds) < k:
        best_i = 0
        best_d2 = -1.0
        for i in range(n):
            d2 = min(l2sq([points[i][j] - points[s][j] for j in range(len(points[i]))]) for s in seeds)
            if d2 > best_d2:
                best_d2 = d2
                best_i = i
        seeds.append(best_i)
    return seeds


def kmeans(points, k=KMEANS_K):
    n = len(points)
    k = min(k, n)
    if n == 0:
        return [], []
    seeds = farthest_seeds(points, k)
    centroids = [list(points[s]) for s in seeds]
    assign = [0] * n
    q = len(points[0])
    for _ in range(LLOYD_MAX_ITER):
        changed = False
        for i in range(n):
            best_c = 0
            best_d2 = l2sq([points[i][j] - centroids[0][j] for j in range(q)])
            for c in range(1, k):
                d2 = l2sq([points[i][j] - centroids[c][j] for j in range(q)])
                if d2 < best_d2:
                    best_d2 = d2
                    best_c = c
            if assign[i] != best_c:
                assign[i] = best_c
                changed = True
        if not changed:
            break
        new_centroids = [[0.0] * q for _ in range(k)]
        counts = [0] * k
        for i in range(n):
            c = assign[i]
            counts[c] += 1
            for j in range(q):
                new_centroids[c][j] += points[i][j]
        for c in range(k):
            if counts[c] > 0:
                for j in range(q):
                    new_centroids[c][j] /= counts[c]
                centroids[c] = new_centroids[c]
    return centroids, assign


def predict_probe(model, tokens):
    v = unit_vec(model["vocab_index"], model["idf"], tokens)
    proj = project(v, model["mean"], model["axes"])
    q = len(proj)
    best_c = 0
    best_d2 = l2sq([proj[j] - model["centroids"][0][j] for j in range(q)])
    for c in range(1, len(model["centroids"])):
        d2 = l2sq([proj[j] - model["centroids"][c][j] for j in range(q)])
        if d2 < best_d2:
            best_d2 = d2
            best_c = c
    return {
        "clusterId": best_c,
        "edge": math.sqrt(best_d2),
        "centroid": list(model["centroids"][best_c]),
        "projection": proj,
    }


def build(lang):
    path = DATA / f"{lang}.json"
    if not path.exists():
        print(f"[train_kmeans] {lang}: corpus missing, skipped")
        return None
    with open(path, encoding="utf-8") as f:
        corpus = json.load(f)
    rows = intent_docs(corpus)
    nDocs = len(rows)
    if nDocs == 0:
        print(f"[train_kmeans] {lang}: no docs, skipped")
        return None
    classes = []
    seen = set()
    for cid, _ in rows:
        if cid not in seen:
            seen.add(cid)
            classes.append(cid)
    vocab_map = {}
    for _, toks in rows:
        for t in toks:
            vocab_map[t] = vocab_map.setdefault(t, 0) + 1
    vocabulary = sorted(vocab_map)
    vocab_index = {t: i for i, t in enumerate(vocabulary)}
    idf = {t: math.log(nDocs / (vocab_map[t] or 1)) for t in vocabulary}
    vecs = [unit_vec(vocab_index, idf, toks) for _, toks in rows]
    d = len(vocabulary)
    mean = [sum(v[i] for v in vecs) / nDocs for i in range(d)]
    centered = [[v[i] - mean[i] for i in range(d)] for v in vecs]
    axes = pca_axes(centered, Q)
    proj_all = [project(v, mean, axes) for v in vecs]
    centroids, assign = kmeans(proj_all, KMEANS_K)
    model = {
        "schema": SCHEMA,
        "algorithm": ALGORITHM,
        "language": lang,
        "classes": classes,
        "vocabulary": vocabulary,
        "nDocs": nDocs,
        "q": Q,
        "k": len(centroids),
        "metric": "cosine-unit-tfidf",
        "distance": "euclidean",
        "mean": mean,
        "axes": axes,
        "centroids": centroids,
        "assignments": assign,
        "idf": idf,
    }
    model["vocab_index"] = vocab_index
    probes = []
    for toks in PROBES.get(lang, []) or []:
        p = predict_probe(model, toks)
        p["tokens"] = list(toks)
        probes.append(p)
    model["probes"] = probes
    return model


def main():
    for lang in LANGUAGES:
        model = build(lang)
        if model is None:
            continue
        out = OUT / f"kmeans-pca-{lang}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(
            f"[train_kmeans] {lang}: classes={len(model['classes'])} vocab={len(model['vocabulary'])} "
            f"nDocs={model['nDocs']} q={model['q']} k={model['k']} probes={len(model['probes'])} -> {out.name}"
        )


if __name__ == "__main__":
    main()