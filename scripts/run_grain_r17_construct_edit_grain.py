#!/usr/bin/env python3
"""R17: construct-targeted self-reference edit vs nuisance edits, GRAIN lexical reader only."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import eigh

from scripts.run_grain_r17_natural_sensitivity import (
    LB,
    _versions_for,
    load_manifest,
    load_material,
)
from scripts.run_grain_r17_natural_sensitivity import OUT as R17

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R17_CONSTRUCT_EDIT_GRAIN_PROTOCOL_V1.md"
PROTOCOL_SHA = "271d2e56df1690505e02e73958936a13cb27b6d53f4306b5f55179e3ca5a6dfb"
OUT = ROOT / "results/grain_gptpro_foundation_v3/R17/construct_edit_grain_r1"
RUNTIME = ROOT / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"
MAP = {"i'm": "the author is", "i've": "the author has", "i'll": "the author will", "i'd": "the author would",
       "myself": "the author", "mine": "the author's", "my": "the author's", "me": "the author", "i": "the author"}
PAT = re.compile(r"(?<![\w'’])(i['’](?:m|ve|ll|d)|myself|mine|my|me|i)(?![\w'’])", re.IGNORECASE)
BOOT, SEED = 2000, 20260915
TRAITS = ("Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def construct_edit(text):
    n = 0

    def repl(m):
        nonlocal n
        n += 1
        rep = MAP[m.group(0).lower().replace("’", "'")]
        return rep[0].upper() + rep[1:] if m.start() == 0 or text[max(0, m.start() - 2):m.start()].strip() in (".", "!", "?") else rep
    new = PAT.sub(repl, text)
    outside_ok = PAT.split(text)[0::2] == PAT.sub("\x00", text).split("\x00")
    return new, n, outside_ok


def boot_ratio(a, b, rng):
    idx = rng.integers(0, len(a), (BOOT, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return float(a.mean() / b.mean()), *[float(x) for x in np.percentile(r, [2.5, 97.5])]


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(R17)
    material = load_material()
    runtime = joblib.load(RUNTIME)
    rows, results = [], {}
    rng = np.random.default_rng(SEED)
    for corpus in ("pandora", "essays"):
        m = manifest[manifest.corpus == corpus].reset_index(drop=True)
        with np.load(LB / corpus / "private_units.npz", allow_pickle=False) as z:
            cg, cauth = z["g"].astype(float), z["authors"]
        with np.load(LB / corpus / "private_predictions.npz", allow_pickle=False) as z:
            psd = z["grain"].std(0, ddof=1)
        obj = joblib.load(LB / corpus / "private_model_grain.joblib")
        with np.load(LB / corpus / "private_blocks.npz", allow_pickle=False) as z:
            blocks = {k: z[k] for k in z.files}
        keep, cons, nuis, within, head = [], [], [], [], []
        for rec in m.to_dict("records"):
            vs = _versions_for(material, rec)
            new, n, ok = construct_edit(vs["base"])
            rows.append({"corpus": corpus, "author_index": rec["author_index"], "substitutions": n,
                         "applicable": n > 0, "retained": ok})
            if n == 0 or not ok:
                continue
            gb, gc = runtime.transform([vs["base"]])[0], runtime.transform([new])[0]
            gn = runtime.transform([vs["contraction_expand"], vs["sentence_reverse"]])
            cons.append(gc - gb)
            nuis.extend(list(gn - gb))
            within.append(np.linalg.norm(cg[rec["partner_row"]] - cg[rec["unit_row"]]))
            a = int(rec["author_index"])
            urows = np.flatnonzero(cauth == a)
            k = int(np.flatnonzero(urows == rec["unit_row"])[0])

            def score(G, a=a, blocks=blocks, obj=obj):
                b = dict(blocks)
                b["grain"] = blocks["grain"].copy()
                b["grain"][a] = np.r_[G.mean(0), G.std(0, ddof=0)]
                return obj["head"].predict(obj["features"].transform(b, None, [a]))[0]

            G = cg[urows].copy()
            base = score(G)
            G1 = G.copy()
            G1[k] = gc
            G2 = G.copy()
            G2[k] = gn[0]
            G3 = G.copy()
            G3[k] = gn[1]
            head.append({"construct": (score(G1) - base) / psd, "contraction": (score(G2) - base) / psd,
                         "reverse": (score(G3) - base) / psd})
            keep.append(a)
        C, Nn, W = np.array(cons), np.array(nuis), np.array(within)
        dc = np.linalg.norm(C, axis=1)
        dn = np.linalg.norm(Nn, axis=1).reshape(-1, 2).mean(1)
        SC = C.T @ C / len(C)
        SN = Nn.T @ Nn / len(Nn)
        SNr = SN + 1e-3 * np.trace(SN) / SN.shape[0] * np.eye(SN.shape[0])
        lam, vec = eigh(SC, SNr)
        v = vec[:, -1]
        share = float(np.mean((C @ v) ** 2) / np.mean(np.sum(C ** 2, 1)) * (v @ v) ** -1)
        hd = {kk: np.array([x[kk] for x in head]) for kk in ("construct", "contraction", "reverse")}
        trait_rows = {}
        for t, trait in enumerate(TRAITS):
            vals = hd["construct"][:, t]
            idx = rng.integers(0, len(vals), (BOOT, len(vals)))
            trait_rows[trait] = {"construct_mean_sd_units": float(vals.mean()),
                                 "ci": [float(x) for x in np.percentile(vals[idx].mean(1), [2.5, 97.5])],
                                 "contraction_mean": float(hd["contraction"][:, t].mean()),
                                 "reverse_mean": float(hd["reverse"][:, t].mean())}
        results[corpus] = {
            "applicable_units": len(keep), "refused_no_first_person": int(len(m) - len(keep)),
            "construct_over_within": boot_ratio(dc, W, rng),
            "construct_over_nuisance": boot_ratio(dc, dn, rng),
            "generalized_eigen_top3": [float(x) for x in lam[::-1][:3]],
            "generalized_eigen_median": float(np.median(lam)),
            "construct_energy_share_on_top_selective_direction": share,
            "grain_head_shift_by_trait": trait_rows,
        }
    pd.DataFrame(rows).to_csv(OUT / "edit_validation.csv", index=False)
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)), "results": results,
           "neural_calls": 0, "labels_read": False, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "construct_edit.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
