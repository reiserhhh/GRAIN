#!/usr/bin/env python3
"""R35/R36 structural reliability multiverse (split-half/bootstrap subspace stability) on real text."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.run_grain_r17_natural_sensitivity import load_material

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R35_STRUCTURAL_MULTIVERSE_PROTOCOL_V1.md"
PROTOCOL_SHA = "4f75dd4e783e58f0232a85e1c0bf1fb68adb3ebaa23472b47e0d3afe71de6238"
RUNTIME = ROOT / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F13/structural_multiverse_r1"
SEED, SPLITS, BOOTS = 20260926, 5, 10


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def svd_basis(x, k, seed):
    return TruncatedSVD(n_components=k, algorithm="randomized", random_state=seed).fit(x).components_


def similarity(a, b):
    k = a.shape[0]
    qa, _ = np.linalg.qr(a.T)
    qb, _ = np.linalg.qr(b.T)
    s = float(np.linalg.norm(qa.T @ qb) ** 2 / k)
    c = np.abs(a @ b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :])
    r, cidx = linear_sum_assignment(-c)
    return s, float(c[r, cidx].mean())


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    material = load_material()
    runtime = joblib.load(RUNTIME)
    rows, transport = [], []
    for corpus, m in material.items():
        texts = m["texts"]
        authors = m["addresses"].author_index.to_numpy()
        ua = np.unique(authors)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, len(corpus)]))
        splits = []
        for _ in range(SPLITS):
            perm = rng.permutation(ua)
            half = set(perm[: len(ua) // 2].tolist())
            mask = np.array([a in half for a in authors])
            splits.append(mask)
        boots = []
        for _ in range(BOOTS):
            pick = rng.choice(ua, len(ua))
            idx = np.concatenate([np.flatnonzero(authors == a) for a in pick])
            boots.append(idx)
        for k, ngram, weight in itertools.product((12, 24, 48), ((1, 1), (1, 2)), ("tfidf", "binary")):
            vec = TfidfVectorizer(max_features=5000, ngram_range=ngram, strip_accents="unicode",
                                  sublinear_tf=weight == "tfidf", use_idf=weight == "tfidf",
                                  binary=weight == "binary")
            x = vec.fit_transform(texts)
            full = svd_basis(x, k, SEED)
            sh = [similarity(svd_basis(x[s], k, SEED), svd_basis(x[~s], k, SEED)) for s in splits]
            bs = [similarity(svd_basis(x[b], k, SEED), full) for b in boots]
            S = np.array([a for a, _ in sh])
            rows.append({"corpus": corpus, "k": k, "ngram": f"{ngram[0]}-{ngram[1]}", "weighting": weight,
                         "vocab": x.shape[1], "units": x.shape[0],
                         "split_half_S_mean": float(S.mean()), "split_half_S_min": float(S.min()),
                         "split_half_congruence_mean": float(np.mean([b for _, b in sh])),
                         "bootstrap_S_mean": float(np.mean([a for a, _ in bs])),
                         "bootstrap_congruence_mean": float(np.mean([b for _, b in bs])),
                         "random_expectation": k / x.shape[1],
                         "admitted": bool(S.mean() >= 0.80),
                         "is_frozen_default": bool(k == 24 and ngram == (1, 2) and weight == "tfidf")})
            print(json.dumps(rows[-1]), flush=True)
        xr = runtime.vectorizer.transform(texts)
        frozen = runtime.svd.components_
        tr = [similarity(svd_basis(xr[s], 24, SEED), frozen) for s in splits]
        transport.append({"corpus": corpus, "frozen_vs_half_S_mean": float(np.mean([a for a, _ in tr])),
                          "frozen_vs_half_congruence_mean": float(np.mean([b for _, b in tr])),
                          "random_expectation": 24 / frozen.shape[1]})
        print(json.dumps(transport[-1]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "multiverse.csv", index=False)
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)),
           "universes": len(df), "admitted": int(df.admitted.sum()), "transport_frozen_structure": transport,
           "neural_calls": 0, "raw_text_in_outputs": False, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
