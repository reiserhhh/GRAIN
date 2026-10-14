#!/usr/bin/env python3
"""R17 nuisance vs construct edit battery across readers (protocol v1)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from scripts.run_grain_evidence_calibration import infer_pairs
from scripts.run_grain_gptpro_f08_nli_reader import QUERIES
from scripts.run_grain_gptpro_f11_grain_reader import predict_readout

TT = ROOT / "results/grain_gptpro_foundation_v3/F16/text_transport_full_r1"
RUNTIME = ROOT / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"
HEAD = ROOT / "results/grain_gptpro_foundation_v3/F11/grain_comment_reader_r1/readout_bundle.json"
CAP = 1500


def swap(text, a, b):
    return re.sub(rf"\b({a}|{b})\b", lambda m: b if m.group(1) == a else a, text)


def edits(row):
    t, status = row["text"], row["subject_status"]
    sentences = re.split(r"(?<=\.)\s+", t)
    out = [("N_prefix", "nuisance", "Just to note: " + t), ("N_length", "nuisance", t + " The room had been booked for an hour.")]
    if len(sentences) >= 2:
        out.append(("N_order", "nuisance", " ".join([sentences[-1]] + sentences[:-1])))
    out.append(("N_paraphrase", "nuisance", t.replace("supported", "backed").replace("rejected", "turned down")))
    out.append(("C_pole", "construct_owned" if status in ("actual", "negated") else "construct_unowned", swap(t, "supported", "rejected")))
    if status == "actual":
        m = re.search(r"I actually (supported|rejected) the plan\.", t)
        if m:
            out.append(("C_own", "construct_owned", t.replace(m.group(0), f"My colleague said, 'I actually {m.group(1)} the plan.' That is my colleague's decision, not mine.")))
    elif status == "quoted":
        m = re.search(r"My colleague said, 'I actually (supported|rejected) the plan\.' That is my colleague's decision, not mine\.", t)
        if m:
            out.append(("C_own", "construct_owned", t.replace(m.group(0), f"I actually {m.group(1)} the plan.")))
    return out


def read(texts, ids):
    """GRAIN (lexical runtime + head) and NLI readings; NLI cap enforced before the call."""
    pairs = pd.DataFrame([{"audit_id": f"{i}:{k}", "sentence": t, "item_text": q} for i, t in zip(ids, texts, strict=True) for k, q in QUERIES.items()])
    if len(pairs) > CAP:
        raise RuntimeError(f"NLI budget {len(pairs)} > cap {CAP}")
    runtime = joblib.load(RUNTIME)
    head = json.loads(HEAD.read_text())
    feats = runtime.transform(list(texts))
    _, clipped = predict_readout(feats, head)
    nli = infer_pairs(pairs).set_index("audit_id")
    rows = []
    for j, i in enumerate(ids):
        pos, neg = float(nli.loc[f"{i}:positive", "p_entailment"]), float(nli.loc[f"{i}:negative", "p_entailment"])
        rows.append({"id": i, "grain_pole": float(clipped[j, 0]), "grain_qualification": float(clipped[j, 1]),
                     "nli_pole": pos / (pos + neg), "nli_qualification": float(nli.loc[f"{i}:qualification", "p_entailment"])})
    return pd.DataFrame(rows).set_index("id"), np.asarray(feats), len(pairs)


def main():
    """Run the battery and write metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = pd.read_csv(TT / "reader_scores.csv")
    base = base[base.role == "source"].reset_index(drop=True)
    items = [{"id": r.material_id, "base": r.material_id, "edit": "base", "class": "base", "text": r.text, "status": r.subject_status} for r in base.itertuples()]
    for r in base.to_dict("records"):
        for name, cls, text in edits(r):
            items.append({"id": f"{r['material_id']}|{name}", "base": r["material_id"], "edit": name, "class": cls, "text": text, "status": r["subject_status"]})
    frame = pd.DataFrame(items)
    frame.to_csv(args.output / "edited_texts.csv", index=False)
    scores, feats, pairs = read(frame.text.tolist(), frame.id.tolist())
    scores.to_csv(args.output / "reader_scores_edits.csv")
    arch = base.set_index("material_id")
    replay = {"grain_pole": float(np.abs(scores.loc[arch.index, "grain_pole"] - arch["grain_pole"]).max()),
              "nli_pole": float(np.abs(scores.loc[arch.index, "nli_pole"] - arch["pole_readout"]).max()),
              "nli_qualification": float(np.abs(scores.loc[arch.index, "nli_qualification"] - arch["nli_qualification"]).max())}
    feat = dict(zip(frame.id, feats, strict=True))
    metrics = {}
    for score in ("grain_pole", "grain_qualification", "nli_pole", "nli_qualification"):
        m = {}
        for edit in ("N_prefix", "N_length", "N_order", "N_paraphrase", "C_pole", "C_own"):
            for cls in ("nuisance", "construct_owned", "construct_unowned"):
                sel = frame[(frame.edit == edit) & (frame["class"] == cls)]
                if len(sel):
                    d = np.abs(scores.loc[sel.id, score].to_numpy() - scores.loc[sel.base, score].to_numpy())
                    m[f"{edit}:{cls}"] = {"n": len(sel), "mean_abs_delta": float(d.mean())}
        nuis = np.mean([v["mean_abs_delta"] for k, v in m.items() if k.endswith(":nuisance")])
        target = "C_pole:construct_owned" if "pole" in score else "C_own:construct_owned"
        m["CSNR"] = m[target]["mean_abs_delta"] / nuis if nuis > 0 else float("inf")
        m["nuisance_mean_abs_delta"] = float(nuis)
        metrics[score] = m
    dn = np.array([feat[i] - feat[b] for i, b in zip(frame.id, frame.base, strict=True) if i != b and frame.set_index("id").loc[i, "class"] == "nuisance"])
    dp = np.array([feat[i] - feat[b] for i, b, e, c in zip(frame.id, frame.base, frame.edit, frame["class"], strict=True) if e == "C_pole" and c == "construct_owned"])
    sn = np.cov(dn.T) + 1e-6 * np.trace(np.cov(dn.T)) / dn.shape[1] * np.eye(dn.shape[1])
    sp = dp.T @ dp / len(dp)
    from scipy.linalg import eigh
    vals, vecs = eigh(sp, sn)
    top = vecs[:, -1] / np.linalg.norm(vecs[:, -1])
    head = json.loads(HEAD.read_text())
    wpole = np.asarray(head["coef"])[0] / np.asarray(head["scale"])
    wpole = wpole / np.linalg.norm(wpole)
    eig = {"top_generalized_eigenvalue": float(vals[-1]), "second": float(vals[-2]), "cosine_top_vs_grain_pole_readout": float(abs(top @ wpole)),
           "n_nuisance_deltas": len(dn), "n_owned_pole_deltas": len(dp)}
    out = {"nli_pairs_used": pairs, "cap": CAP, "replay_base_vs_archive_max_abs": replay, "metrics": metrics, "generalized_eigen": eig}
    (args.output / "edit_battery.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
