#!/usr/bin/env python3
"""train_knn.py - deterministic cosine kNN memory recall (Plan B).

Twin of js/agent/knn.js. Same frozen-contract discipline as train_nb.py /
train_logreg.py: stdlib only, no third party, no Math/random anywhere, and
the shipped probes embed their EXACT token lists so the JS predictKnn
replay must match to 1e-6 (real parity, not ranges).

The corpus source is THE SAME shipped intent corpus train_logreg.py reads:
  data/agent/{en,fr,ar}.json  (intents[] = {id, examples:[string], ...})
The "story" (as in Plan B, memory recall): each intent example is a
remembered document; a query is answered by the class whose remembered
documents are most similar (cosine over unit tf-idf vectors, k=3 nearest,
each neighbor votes its cosine, votes softmax -> confidence/margin). This
is the MIRROR of the trainer that produced Plan A's memory-store vibe but
retrieval-driven - a deterministic knn "memory recall" twin.
"""
import json
import math
import re
from train_logreg import normalize_arabic, tokenize
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "agent"
OUT = DATA
LANGUAGES = ("en", "fr", "ar")
K = 3
SCHEMA = 1
ALGORITHM = "knn-cosine-1"

IDF_SPEC = {"metric": "cosine", "k": K, "idf": "log(nDocs/df)", "log": "natural"}


def doc_tokens_corpus(entry):
    if isinstance(entry, str):
        return tokenize(entry)
    if isinstance(entry, dict):
        for key in ("text", "doc", "utterance"):
            v = entry.get(key)
            if isinstance(v, str):
                return tokenize(v)
        toks = entry.get("tokens")
        if isinstance(toks, list):
            return [str(t) for t in toks]
    return []


def build(lang):
    path = DATA / f"{lang}.json"
    if not path.exists():
        print(f"[train_knn] {lang}: corpus missing, skipped")
        return None
    with open(path, encoding="utf-8") as f:
        corpus = json.load(f)
    corpus_lang = corpus.get("language") if isinstance(corpus, dict) else lang
    intents = corpus.get("intents", []) if isinstance(corpus, dict) else corpus
    classes = []
    examples = []  # {cls, tokens}
    for intent in intents:
        if not isinstance(intent, dict):
            continue
        cid = str(intent.get("id", ""))
        if cid and cid not in classes:
            classes.append(cid)
        raw = intent.get("examples", []) if isinstance(intent, dict) else []
        if not isinstance(raw, list) and isinstance(intent, dict):
            raw = intent.get("docs", []) or []
        if not isinstance(raw, list):
            raw = []
        for entry in raw:
            toks = doc_tokens_corpus(entry)
            if toks:
                examples.append({"cls": cid, "tokens": toks})
    nDocs = len(examples)
    vocab_set = set()
    for ex in examples:
        vocab_set.update(ex["tokens"])
    vocabulary = sorted(vocab_set)
    # idf(t) = log(nDocs / df(t)); df over distinct docs containing t
    df_map = {}
    for ex in examples:
        for t in set(ex["tokens"]):
            df_map[t] = df_map.get(t, 0) + 1
    idf = {}
    for t in vocabulary:
        idf[t] = math.log(nDocs / (df_map.get(t, 0) or 1))
    model = {
        "schema": SCHEMA,
        "algorithm": ALGORITHM,
        "classes": classes,
        "examples": [{"cls": e["cls"], "tokens": list(e["tokens"])} for e in examples],
        "vocabulary": vocabulary,
        "nDocs": nDocs,
        "k": K,
        "metric": "cosine",
        "idf": idf,
        "probes": [],
    }
    model["probes"] = probe_predictions(model, corpus_lang)
    return model


def predict_tokens(model, tokens):
    cls_id = model["classes"]
    if not cls_id:
        return {"id": "", "confidence": 0.0, "margin": 0.0, "scores": [], "probs": []}
    vocab_set = set(model["vocabulary"])
    idf = model["idf"]
    # query unit tf-idf vector (unknown query tokens skipped)
    q = tfidf_unit(model, tokens)
    # example vectors
    sims = []
    for i, ex in enumerate(model["examples"]):
        v = tfidf_unit(model, ex["tokens"])
        sims.append({"i": i, "s": cosine(model, q, v)})
    sims.sort(key=lambda p: (-p["s"], p["i"]))
    top = sims[: model["k"]]
    votes = [0.0] * len(cls_id)
    for kn in top:
        ci = cls_id.index(model["examples"][kn["i"]]["cls"])
        votes[ci] += kn["s"]
    probs = softmax(votes)
    best = 0
    for i in range(1, len(votes)):
        if votes[i] > votes[best]:
            best = i
    second = 0
    for i in range(len(votes)):
        if i != best and probs[i] > second:
            second = probs[i]
    return {
        "id": cls_id[best],
        "confidence": probs[best],
        "margin": probs[best] - second,
        "scores": votes,
        "probs": probs,
    }


def tfidf_unit(model, tokens):
    vocab_set = set(model["vocabulary"])
    freq = {}
    for t in tokens:
        if t in vocab_set:
            freq[t] = freq.get(t, 0) + 1
    vec = {}
    sq = 0.0
    for t, f in freq.items():
        w = f * model["idf"][t]
        vec[t] = w
        sq += w * w
    if not sq > 0:
        return vec
    inv = 1.0 / math.sqrt(sq)
    for t in vec:
        vec[t] *= inv
    return vec


def cosine(model, a, b):
    d = 0.0
    for t, w in a.items():
        if t in b:
            d += w * b[t]
    return d


def softmax(vals):
    if not vals:
        return []
    mx = max(vals)
    exps = [math.exp(v - mx) for v in vals]
    total = sum(exps)
    if not total > 0:
        p = 1.0 / len(vals)
        return [p] * len(vals)
    return [e / total for e in exps]


PROBES = {
    "en": ["hello there", "what time is it", "goodbye for now", "thank you very much"],
    "fr": ["bonjour comment ca va", "quelle heure est il", "au revoir a bientot", "merci beaucoup"],
    "ar": ["as-salamu alaykum kayfa haluk", "kamel sa3a kam", "ila al-liqa lahiqan", "shukran jaziylan"],
}


def probe_predictions(model, lang):
    if lang not in PROBES:
        return []
    return [
        {**predict_tokens(model, toks), "tokens": list(toks)}
        for toks in PROBES.get(lang, [])
    ]


def main():
    for lang in LANGUAGES:
        model = build(lang)
        if model is None:
            continue
        out = OUT / f"knn-memory-{lang}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2, sort_keys=True)
        n = len(model["examples"])
        print(
            f"[train_knn] {lang}: classes={len(model['classes'])} vocab={len(model['vocabulary'])} "
            f"nDocs={n} probes={len(model['probes'])} -> {out.name}"
        )


if __name__ == "__main__":
    main()
