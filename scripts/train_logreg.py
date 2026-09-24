#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morphius - Multinomial softmax logistic-regression trainer (Python twin)
========================================================================
Trainer twin of js/agent/logreg.js. Deterministic, stdlib-only (math, json,
re, collections). Mirrors LOGREG_MATH_SPEC exactly so the JS<->Python parity
test (tests/logregParity.test.mjs) passes to 1e-6 on SHIPPED artifacts:

    schema    : 1
    algorithm : multinomial-logreg-1
    features  : bag-of-words-normalized (normalized tokens over vocab)
    init      : zeros
    epochs    : 200
    lr        : 0.05
    l2        : 1e-4
    log base  : natural
    unknown   : tokens not in vocabulary are SKIPPED (contribute 0)
    tieBreak  : (score desc, class insertion order asc)
    confidence: softmax(top1); margin = p(top1) - p(top2)

Training is full-batch softmax GD: W[c][t], bias[c] -> zero init, fixed
epochs/lr/l2, no shuffle, natural exp. Deterministic across machines.

    python scripts/train_logreg.py

Reads: data/agent/{en,fr,ar}.json + stopwords-{lang}.json (tokenization is a
pure function; same as train_agent.py).
Writes: data/agent/logreg-weights-{en,fr,ar}.json in the EXACT serialized
shape awaited by deserializeLogReg (schema/algorithm/classes/weights{class:
{token}}/bias{class}/vocabulary[]/nDocs), plus a `probes` block carrying the
Python reference predictions that the JS parity test re-derives with
predictLogReg and compares to 1e-6.
"""
import json
import math
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "agent"
OUT = DATA
LANGUAGES = ["en", "fr", "ar"]

SCHEMA = 1
ALGORITHM = "multinomial-logreg-1"
EPOCHS = 200
LR = 0.05
L2 = 1e-4

AR_MARKS_RE = re.compile(r"[\u0610-\u061A\u0620\u0640\u064B-\u065F\u0670\u06D6-\u06ED\u08D4-\u08E1]")
AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ئ": "ي", "ؤ": "و"})

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


def has_arabic_script(text):
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def normalize_arabic(text):
    return text.translate(AR_MAP)


def tokenize(text):
    raw = str(text or "")
    if has_arabic_script(raw):
        raw = AR_MARKS_RE.sub("", normalize_arabic(raw))
    else:
        raw = raw.lower()
    return [t for t in re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", raw)]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def intent_docs(corpus, stopwords):
    stops = set(stopwords.get("stopwords", []) or [])
    rows = []
    for it in corpus.get("intents", []) or []:
        cid = it.get("id")
        if not cid:
            continue
        for ex in it.get("examples", []) or []:
            if isinstance(ex, str) and ex.strip():
                toks = [t for t in tokenize(ex) if t and t not in stops]
                rows.append([str(cid), toks])
    return rows


def softmax(scores):
    mx = max(scores) if scores else 0.0
    exps = [math.exp(s - mx) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def predict(model, tokens):
    classes = model["classes"]
    scores = []
    for c in classes:
        w = model["weights"].get(c, {})
        bias = model.get("bias", {}).get(c, 0.0)
        acc = bias
        for t in tokens:
            if t in w:
                acc += w[t]  # unknown skipped
        scores.append(acc)
    probs = softmax(scores)
    best = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best]:
            best = i
    margin = 0.0
    if len(probs) > 1:
        s = sorted(probs, reverse=True)
        margin = s[0] - s[1]
    return {
        "id": classes[best],
        "confidence": probs[best],
        "margin": margin,
        "probs": probs,
        "scores": scores,
    }


def fit(corpus, stopwords):
    rows = intent_docs(corpus, stopwords)
    classes = []
    weights = {}
    bias = {}
    vocab_counter = Counter()
    docs = {}  # class -> list of token-lists
    for cid, toks in rows:
        if cid not in docs:
            docs[cid] = []
            classes.append(cid)
        docs[cid].append(toks)
        for t in toks:
            vocab_counter[t] += 1

    vocabulary = sorted(vocab_counter)
    for c in classes:
        weights[c] = {t: 0.0 for t in vocabulary}
        bias[c] = 0.0

    # full-batch softmax GD (zeros init, fixed epochs/lr/l2, no shuffle)
    for _ in range(EPOCHS):
        gw = {c: {t: 0.0 for t in vocabulary} for c in classes}
        gb = {c: 0.0 for c in classes}
        for cid in classes:
            for toks in docs[cid]:
                scores = [bias[c] + sum(weights[c][t] for t in toks if t in weights[c]) for c in classes]
                probs = softmax(scores)
                target = classes.index(cid)
                for ci, c in enumerate(classes):
                    err = probs[ci] - (1.0 if ci == target else 0.0)
                    gb[c] += err
                    for t in toks:
                        if t in weights[c]:
                            gw[c][t] += err
        for c in classes:
            for t in vocabulary:
                gw[c][t] += L2 * weights[c][t]
        for c in classes:
            for t in vocabulary:
                weights[c][t] -= LR * gw[c][t]
            bias[c] -= LR * gb[c]

    model = {
        "schema": SCHEMA,
        "algorithm": ALGORITHM,
        "classes": classes,
        "weights": {c: dict(w) for c, w in weights.items()},
        "bias": {c: bias[c] for c in classes},
        "vocabulary": vocabulary,
        "nDocs": sum(len(v) for v in docs.values()),
    }
    model["probes"] = [
        {**predict(model, toks), "tokens": toks} for toks in PROBES.get(corpus.get("language", "en"), [])
    ]
    return model


def main():
    for lang in LANGUAGES:
        corpus = load_json(DATA / f"{lang}.json")
        stopwords = {}
        try:
            stopwords = load_json(DATA / f"stopwords-{lang}.json")
        except Exception:
            pass
        model = fit(corpus, stopwords)
        out = OUT / f"logreg-weights-{lang}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2, sort_keys=True)
        n = sum(len(v) for v in corpus.get("intents", []))
        print(f"[train_logreg] {lang}: classes={model['classes']} "
              f"vocab={len(model['vocabulary'])} nDocs={model['nDocs']} probes={len(model['probes'])} -> {out.name}")


if __name__ == "__main__":
    main()
