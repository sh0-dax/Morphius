#!/usr/bin/env python3
"""
Morphius — Local Agent model trainer (reference weights)
===========================================================
Implements the exact same Multinomial Naive Bayes spec as js/agent/nb.js
so the PARITY TEST (tests/agentParity.test.mjs) can diff the two to 1e-6.

Stdlib ONLY (math + json). No numpy, no sklearn, no network.

    python scripts/train_agent.py

Writes data/agent/weights-{en,fr,ar}.json — these are SHIPPED in the repo
(web app is offline-first) and must be committed after every corpus change.
The browser normally trains in-browser from data/agent/*.json; these files
are the reference + a fast-path fallback / corruption recovery seed.

Spec (shared with js/agent/nb.js):
    features  : BagOfWords counts over normalized tokens
    vocabulary: unique normalized tokens, sorted lexicographically
    prior     : P(c) = count(c) / totalSamples
    likelihood: P(w|c) = (count(w,c) + alpha) / (classTotals(c) + alpha*|V|)
    alpha     : 1.0 (Laplace)
    log base  : natural
    unknown   : skipped at predict time
    argmax    : (score desc, class insertion order asc)
"""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "agent"
AGENT_LANGUAGES = ["en", "fr", "ar"]
SCHEMA = 1
ALGORITHM = "multinomial-nb-1"
ALPHA = 1.0

AR_MARKS_RE = re.compile(r"[\u0610-\u061A\u0620\u0640\u064B-\u065F\u0670\u06D6-\u06ED\u08D4-\u08E1\uFB1D]")
AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ئ": "ي", "ؤ": "و"})


def has_arabic_script(text: str) -> bool:
    return bool(re.search("[\\u0600-\\u06FF]", text))


def normalize_arabic(text: str) -> str:
    return text.translate(AR_MAP)


def tokenize(text: str) -> list:
    raw = str(text or "")
    if has_arabic_script(raw):
        normalized = AR_MARKS_RE.sub("", normalize_arabic(raw))
    else:
        normalized = raw.lower()
    tokens = []
    cur = []
    for ch in normalized:
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def filter_stopwords(tokens: list, lang: str, stopwords: dict) -> list:
    s = set(stopwords.get(lang, [])) if stopwords else set()
    kept = [t for t in tokens if t not in s]
    return kept if kept else tokens


def features_for_text(text: str, lang: str, stopwords: dict) -> list:
    return filter_stopwords(tokenize(text), lang, stopwords)


# ---- corpus helpers ----
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def intent_docs(corpus: dict, stopwords: dict):
    rows = []
    for it in corpus.get("intents", []):
        if not it or not it.get("id"):
            continue
        for ex in it.get("examples", []) or []:
            if isinstance(ex, str) and ex.strip():
                rows.append([it["id"], features_for_text(ex, corpus["language"], stopwords)])
    return rows


# ---- Multinomial NB + serialization (mirror of js/agent/nb.js) ----
class NB:
    def __init__(self):
        self.classes = []
        self.docs_per_class = {}
        self.token_counts = {}
        self.n_docs = 0.0

    def partial_fit(self, cls: str, tokens: list, weight: float = 1.0):
        cls = str(cls)
        if cls not in self.docs_per_class:
            self.docs_per_class[cls] = 0.0
            self.token_counts[cls] = {}
            self.classes.append(cls)
        w = weight if math.isfinite(weight) and weight > 0 else 1.0
        self.docs_per_class[cls] += w
        self.n_docs += w
        counts = self.token_counts[cls]
        for t in tokens:
            if t:
                counts[t] = counts.get(t, 0.0) + w

    def fit(self, docs):
        for cls, tokens in docs:
            self.partial_fit(cls, tokens, 1.0)
        return self

    def vocabulary(self) -> list:
        return sorted({t for c in self.classes for t in self.token_counts.get(c, {})})

    def serialize(self) -> dict:
        return {
            "schema": SCHEMA,
            "algorithm": ALGORITHM,
            "classes": list(self.classes),
            "docsPerClass": {c: self.docs_per_class[c] for c in self.classes},
            "tokenCounts": {c: dict(self.token_counts[c]) for c in self.classes},
            "vocabulary": self.vocabulary(),
            "nDocs": self.n_docs,
        }


def build_lang_model(corpus: dict, stopwords: dict) -> NB:
    m = NB()
    m.fit(intent_docs(corpus, stopwords))
    return m


def build_lang_model_nb(corpora: dict, stopwords: dict) -> dict:
    out = {}
    for lang in AGENT_LANGUAGES:
        if lang not in corpora:
            continue
        out[lang] = build_lang_model(corpora[lang], stopwords)
    return out


def main() -> None:
    print(f"[train] reading corpus + stopwords from {DATA} ...")
    stopwords = load_json(DATA / "stopwords.json")
    corpora = {}
    for lang in AGENT_LANGUAGES:
        corpora[lang] = load_json(DATA / f"{lang}.json")

    models = build_lang_model_nb(corpora, stopwords)
    total_examples = 0
    for lang, model in models.items():
        path = DATA / f"weights-{lang}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model.serialize(), f, ensure_ascii=False, sort_keys=False, indent=2)
        n = sum(model.docs_per_class.values())
        total_examples += n
        print(f"  [train] {lang}: {len(model.classes)} intents, "
              f"{len(model.vocabulary())} vocab tokens, {n:.0f} training docs -> {path.name}")
    print(f"[train] done. {total_examples:.0f} total training docs. "
          f"Remember: commit the weights AND run tests/agentParity.test.mjs.")


if __name__ == "__main__":
    main()