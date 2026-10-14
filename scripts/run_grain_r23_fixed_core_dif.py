#!/usr/bin/env python
"""R23/R25 fixed-core frame/topic/day response variation (docs/GRAIN_R23_FIXED_CORE_DIF_PROTOCOL_V1.md). Zero calls."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R23_FIXED_CORE_DIF_PROTOCOL_V1.md"
PROTOCOL_SHA = "1ce2af65a233a3d7a72136c8b139b5c1529178dc3bc6f9e3ed5a8b8010b9652e"
R14 = ROOT / "results/grain_gptpro_foundation_v3/R14/same_object_channels_r1"
BANK = ROOT / "results/grain_gptpro_foundation_v3/F16/text_transport_full_r1"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F12/fixed_core_dif_r1"
SEED, B = 20260926, 1000
FRAMES = {"source": ("For the written minutes", "In my own account", "At the review"),
          "pilot": ("To clarify the record", "In this short explanation", "For the follow-up"),
          "application": ("Here is what happened", "To explain the exchange", "In the final account")}
ENDINGS = ("The meeting concerned a proposed timetable.", "The discussion concerned a proposed budget.", "The proposal concerned a shared workspace.",
           "The meeting concerned a proposed schedule.", "The proposal concerned a community project.")
MOMENTS = ("Monday", "Tuesday", "Wednesday", "Thursday")
OUTPUTS = {"nli_p": "pole", "nli_q": "qual", "grain_q_hist": "qual", "lex96_p": "pole", "lex96_q": "qual", "lex80_p": "pole", "lex80_q": "qual", "grainridge_p": "pole"}


def load() -> pd.DataFrame:
    inv = json.loads((R14 / "inventory.json").read_text())
    if hashlib.sha256((R14 / "typed_reader_table.csv").read_bytes()).hexdigest() != inv["typed_table_sha256"]:
        raise RuntimeError("typed table hash mismatch")
    t = pd.read_csv(R14 / "typed_reader_table.csv")
    m = pd.read_csv(BANK / "materials.csv")[["material_id", "role", "variant", "text"]].rename(columns={"variant": "variant_m"})
    d = t.merge(m, on="material_id", validate="one_to_one")
    if not (d.variant.astype(int) == d.variant_m.astype(int)).all():
        raise RuntimeError("variant mismatch between typed table and materials")
    v = d.variant.astype(int)
    d["frame_f"] = [FRAMES[r][x % 3] for r, x in zip(d.role, v)]
    d["day_f"] = [MOMENTS[(x // 15) % 4] for x in v]
    d["topic_f"] = [ENDINGS[(x // 3) % 5] for x in v]
    d["frame_t"] = d.text.str.extract(r"^(.*?) on (?:Mon|Tues|Wednes|Thurs)day,")[0]
    d["day_t"] = d.text.str.extract(r" on ((?:Mon|Tues|Wednes|Thurs)day),")[0]
    d["topic_t"] = d.text.str.extract(r"(The [a-z]+ concerned [^.]+\.)$")[0]
    d["core"] = d.pole.astype(str) + "_" + d.subject_status
    return d


def design(g: pd.DataFrame, key: str | None = None) -> tuple[np.ndarray, list[str]]:
    cols, names = [np.ones(len(g))], ["intercept"]
    for fac in ("core", "frame_f", "topic_f", "day_f"):
        levels = sorted(g[fac].unique())
        for lv in levels[1:]:
            cols.append((g[fac] == lv).to_numpy(float))
            names.append(f"{fac}={lv}")
    if key is not None:
        kk = g[key].to_numpy(float)
        for fac in ("frame_f", "topic_f"):
            for lv in sorted(g[fac].unique())[1:]:
                cols.append(kk * (g[fac] == lv).to_numpy(float))
                names.append(f"{key}x{fac}={lv}")
    return np.column_stack(cols), names


def fit(g, y, key=None):
    X, names = design(g, key)
    rank = int(np.linalg.matrix_rank(X))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    null = []
    if rank < X.shape[1]:
        _, s, vt = np.linalg.svd(X)
        ns = vt[rank:]
        null = [names[j] for j in range(len(names)) if np.abs(ns[:, j]).max() > 1e-8]
    return dict(zip(names, coef)), rank, X.shape[1], null


def ranges(coef: dict, prefix: str) -> float:
    vals = [0.0] + [v for k, v in coef.items() if k.startswith(prefix)]
    return float(max(vals) - min(vals))


def core_contrast(g, out, kind):
    if kind == "pole":
        q = g[g.qualification == 1]
        return float(q[q.pole == 1][out].mean() - q[q.pole == 0][out].mean())
    return float(g[g.qualification == 1][out].mean() - g[g.qualification == 0][out].mean())


def summarize(g, out, kind):
    y = g[out].to_numpy(float)
    key = "pole" if kind == "pole" else "qualification"
    ca, rank_a, p_a, null_a = fit(g, y)
    ci, rank_i, p_i, null_i = fit(g, y, key)
    loading_frame = [ci.get(f"{key}xframe_f={lv}", 0.0) for lv in sorted(g.frame_f.unique())[1:]]
    loading_topic = [ci.get(f"{key}xtopic_f={lv}", 0.0) for lv in sorted(g.topic_f.unique())[1:]]
    return {"core": core_contrast(g, out, kind), "rank_add": rank_a, "p_add": p_a, "null_add": null_a, "rank_int": rank_i, "p_int": p_i, "null_int": null_i,
            "frame_range": ranges(ca, "frame_f="), "topic_range": ranges(ca, "topic_f="), "day_range": ranges(ca, "day_f="),
            "loading_frame_range": float(max([0.0] + loading_frame) - min([0.0] + loading_frame)),
            "loading_topic_range": float(max([0.0] + loading_topic) - min([0.0] + loading_topic)),
            "topic_effects": {k.split("=", 1)[1]: v for k, v in ca.items() if k.startswith("topic_f=")},
            "day_effects": {k.split("=", 1)[1]: v for k, v in ca.items() if k.startswith("day_f=")}}


def replay(d):
    spec = importlib.util.spec_from_file_location("r14", ROOT / "scripts/run_grain_r14_same_object_channels.py")
    r14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r14)
    from grain_research.gptpro_foundation_v3.f02 import generate_world
    from grain_research.measurement_discovery.estimators import ordered_rows
    typed, rd, strata = r14.load_typed()
    world, seed = "c2_xor", r14.SEEDS[0]
    records, _ = generate_world(world, seed, counts=r14.COUNTS)
    _s, fitted, initial, smean = r14.fit_source(records, world)
    chans = ("nli_hard", "hybrid_nlipole_grainq", "grainpole_grainq", "lex96_transported", "lex80_refit")
    results = {}
    for ch in chans:
        effects = []
        for ai, (author, arows) in enumerate(ordered_rows([r for r in records if r["role"] != "source"]).items()):
            if arows[0]["role"] != "application":
                continue
            ys = [r["observation"] for r in arows]
            ev = r14.render_transport([seed, 8, 0, ai], ys, "mixed", "application", strata)
            base = [rd[e["material_id"]][ch] for e in ev]
            tstar, alt = None, None
            for t in range(12, 47):
                e = ev[t]
                cands = [m for m in strata[("application", e["material_pole"], e["status"])] if m != e["material_id"] and rd[m][ch] != rd[e["material_id"]][ch]]
                if cands:
                    tstar, alt = t, sorted(cands)[0]
                    break
            if tstar is None:
                continue
            swapped = list(base)
            swapped[tstar] = rd[alt][ch]

            def losses(seq):
                h = r14.QualifiedHistory(fitted, initial, smean, mode="filtered")
                out = {c: [] for c in ("b1", "b2", "m")}
                for t, y in enumerate(ys):
                    before = h.predict() if t >= 2 else None
                    if t >= tstar + 1:
                        for c in out:
                            out[c].append((before["probabilities"][c] - y) ** 2)
                    h.observe(*seq[t])
                return {c: float(np.mean(v)) for c, v in out.items()}

            lo, ls = losses(base), losses(swapped)
            effects.append({"author": author, "tstar": tstar, "orig_material": ev[tstar]["material_id"], "alt_material": alt,
                            "orig_reading": list(base[tstar]), "alt_reading": list(swapped[tstar]), **{f"d_{c}": ls[c] - lo[c] for c in lo}})
        if not effects:
            results[ch] = {"support": 0, "status": "NO_SUPPORT"}
            continue
        e = pd.DataFrame(effects)
        rng = np.random.default_rng(SEED)
        bs = {c: [e[f"d_{c}"].sample(len(e), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(1000)] for c in ("b1", "b2", "m")}
        results[ch] = {"support": len(e), **{f"mean_d_{c}": float(e[f"d_{c}"].mean()) for c in ("b1", "b2", "m")},
                       **{f"ci_d_{c}": [float(np.percentile(bs[c], 2.5)), float(np.percentile(bs[c], 97.5))] for c in ("b1", "b2", "m")}}
        e.to_csv(OUT / f"replay_{ch}.csv", index=False)
    return results


def main() -> int:
    t0 = time.time()
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    parse_ok = bool((d.frame_f == d.frame_t).all() and (d.day_f == d.day_t).all() and (d.topic_f == d.topic_t).all())
    if not parse_ok:
        (OUT / "parse_check.json").write_text(json.dumps({"status": "STOP_PARSE"}) + "\n")
        return 2
    support = {role: {fac: pd.crosstab(g.core, g[fac]).to_dict() for fac in ("frame_f", "topic_f", "day_f")} for role, g in d.groupby("role")}
    (OUT / "support.json").write_text(json.dumps(support, indent=1) + "\n")
    rows, rng = [], np.random.default_rng(SEED)
    for role, g in d.groupby("role"):
        g = g.reset_index(drop=True)
        for out, kind in OUTPUTS.items():
            s = summarize(g, out, kind)
            defined = abs(s["core"]) >= 0.05
            boots = {k: [] for k in ("frame_range", "topic_range", "day_range", "loading_frame_range", "loading_topic_range", "core")}
            skipped = 0
            for _ in range(B):
                ix = rng.integers(0, len(g), len(g))
                gb = g.iloc[ix].reset_index(drop=True)
                if gb.core.nunique() < 8 or gb.frame_f.nunique() < g.frame_f.nunique() or gb.topic_f.nunique() < g.topic_f.nunique() or gb.day_f.nunique() < g.day_f.nunique():
                    skipped += 1
                    continue
                sb = summarize(gb, out, kind)
                for k in boots:
                    boots[k].append(sb[k])
            row = {"role": role, "output": out, "kind": kind, **{k: v for k, v in s.items() if k not in ("topic_effects", "day_effects")},
                   "core_defined": defined, "boot_used": B - skipped, "boot_skipped_level_loss": skipped}
            for k, v in boots.items():
                row[f"{k}_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
            for k in ("frame_range", "topic_range", "day_range", "loading_frame_range", "loading_topic_range"):
                row[f"{k}_over_core"] = (s[k] / abs(s["core"])) if defined else None
            row["topic_effects"], row["day_effects"] = s["topic_effects"], s["day_effects"]
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_json(OUT / "dif_results.json", orient="records", indent=1)
    # transport of shared factors: source-fit vs application (descriptive)
    trans = []
    for out in OUTPUTS:
        src = res[(res.role == "source") & (res.output == out)].iloc[0]
        app = res[(res.role == "application") & (res.output == out)].iloc[0]
        for lv, v in src.topic_effects.items():
            trans.append({"output": out, "factor": "topic", "level": lv, "source_effect": v, "application_effect_descriptive": app.topic_effects.get(lv)})
        for lv, v in src.day_effects.items():
            trans.append({"output": out, "factor": "day", "level": lv, "source_effect": v, "application_effect_descriptive": app.day_effects.get(lv)})
    pd.DataFrame(trans).to_csv(OUT / "shared_factor_transport.csv", index=False)
    rep = replay(d)
    (OUT / "replay_summary.json").write_text(json.dumps(rep, indent=1) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"seconds": time.time() - t0, "neural_calls": 0, "parse_ok": parse_ok,
                                                  "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}) + "\n")
    print(json.dumps({"seconds": time.time() - t0, "parse_ok": parse_ok}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
