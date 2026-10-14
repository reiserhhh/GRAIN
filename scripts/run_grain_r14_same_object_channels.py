#!/usr/bin/env python
"""R14 same-object channel comparison (docs/GRAIN_R14_SAME_OBJECT_CHANNELS_PROTOCOL_V1.md). Zero neural forwards.

Stages: prepare | replay | main | secondary | analyze. Each stage stops on any hash, role or replay mismatch.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from grain_research.gptpro_foundation_v3 import f08  # noqa: E402
from grain_research.gptpro_foundation_v3.f02 import c2_next_probability, generate_world  # noqa: E402
from grain_research.gptpro_foundation_v3.f02_recovery import fit_observed_source  # noqa: E402
from grain_research.gptpro_foundation_v3.f08_estimators import QualifiedHistory  # noqa: E402
from grain_research.measurement_discovery.estimators import ordered_rows  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R14_SAME_OBJECT_CHANNELS_PROTOCOL_V1.md"
PROTOCOL_SHA = "068cfb7922c1fded08949493eb43619adaecb6f3aaa95173e3a756dc0dd98f48"
RES = ROOT / "results/grain_gptpro_foundation_v3"
BANK = RES / "F16/text_transport_full_r1"
LEX = RES / "F08/reader_recovery_r1"
DOWN = RES / "F08/downstream_full_r2"
OUT = RES / "R14/same_object_channels_r1"
BANK_SHA = "fe5e33cb9ce987ae1f550750005662dbb02e8ecc1f305aa308cfc06264634da2"
LEX_SCORES_SHA = "92efe929d06a5d97626f82b7bb7d99415f4a37c5f0229d201c91c787bf84c924"
SEEDS = (20260915, 20260916, 20260917)
WORLDS = ("c2_xor", "c2_iid")
COUNTS = {"source": 300, "validation": 100, "application": 200}
STATUSES = ("actual", "quoted", "hypothetical", "negated")
CHANNELS = ("oracle", "nli_soft", "nli_hard", "hybrid_nlipole_grainq", "grainpole_grainq", "lex96_transported", "lex80_refit")
HARD = CHANNELS[2:]
CANDS = ("b1", "b2", "m", "source_mean", "personal_mean")
MSTREAM = 1601
CAP_S = 90 * 60


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wj(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n")


def role_key(role: str, partition) -> str:
    if role == "pilot":
        return f"pilot_{partition}"
    return role


# ---------------------------------------------------------------- prepare
def prepare() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    stop = []
    if sha(BANK / "reader_scores.csv") != BANK_SHA:
        stop.append("bank reader_scores hash")
    if sha(LEX / "reader_scores.csv") != LEX_SCORES_SHA:
        stop.append("canonical lexical reader_scores hash")
    bank = pd.read_csv(BANK / "reader_scores.csv")
    mats = pd.read_csv(BANK / "materials.csv")
    if not (bank.material_id.tolist() == mats.material_id.tolist() and (bank.text_sha256 == mats.text_sha256).all()):
        stop.append("bank/materials ID or text hash mismatch")
    z = np.load(BANK / "grain_features.npz")
    feats, raw = z["features"], z["raw"]
    if feats.shape != (380, 24) or len(bank) != 380:
        stop.append("feature rows")
    if not np.allclose(np.clip(raw[:, 1], 0, 1), bank.grain_qualification.to_numpy(), atol=1e-12):
        stop.append("GRAIN feature row order does not reproduce cached grain_qualification")
    if not np.allclose(bank.grain_pole.to_numpy(), 0.5, atol=1e-9):
        stop.append("historical grain_pole not constant .5")
    bank["rk"] = [role_key(r, p) for r, p in zip(bank.role, bank.pilot_partition)]
    counts = bank.rk.value_counts().to_dict()
    if counts != {"application": 200, "pilot_fit": 60, "pilot_validation": 40, "source": 80}:
        stop.append(f"role counts {counts}")
    hashes_by_role = bank.groupby("rk").text_sha256.apply(set)
    overlap = sum(len(a & b) for i, a in enumerate(hashes_by_role) for b in list(hashes_by_role)[i + 1:])
    if overlap:
        stop.append("text hashes shared across roles")
    qmeta = bank.subject_status.isin(["actual", "negated"]).astype(int)
    if not (qmeta == bank.qualification).all():
        stop.append("status/qualification metadata mismatch")
    # R21 collision check on cached GRAIN features
    keys = [hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() for f in feats]
    bank["grain_class"] = keys
    cls = bank.groupby("grain_class").pole.agg(["nunique", "size", "sum"])
    bayes_acc = float((np.maximum(cls["sum"], cls["size"] - cls["sum"])).sum() / len(bank))
    collisions = {"distinct_feature_vectors": int(len(cls)), "classes_with_both_poles": int((cls["nunique"] == 2).sum()),
                  "items_in_both_pole_classes": int(cls.loc[cls["nunique"] == 2, "size"].sum()),
                  "bayes_optimal_pole_accuracy_from_features": bayes_acc,
                  "note": "any function of these GRAIN features gives equal pole output within a class"}
    # fixed GRAIN pole readout (source-80 only)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    src = (bank.rk == "source").to_numpy()
    scaler = StandardScaler().fit(feats[src])
    ridge = Ridge(alpha=1.0).fit(scaler.transform(feats[src]), bank.pole[src].to_numpy(float))
    graw = ridge.predict(scaler.transform(feats))
    bank["grainridge_p_raw"] = graw
    bank["grainridge_p"] = np.clip(graw, 0, 1)
    # lexical transported (canonical F08 reader) and refit (source-80)
    with (LEX / "reader.pkl").open("rb") as fh:
        lex96 = pickle.load(fh)
    texts = bank.text.tolist()
    p96, q96 = f08.predict_reader(lex96, texts)
    lex80 = f08.fit_reader([texts[i] for i in np.where(src)[0]], bank.pole[src].astype(int).tolist(), bank.qualification[src].astype(int).tolist())
    p80, q80 = f08.predict_reader(lex80, texts)
    bank["lex96_p"], bank["lex96_q"], bank["lex80_p"], bank["lex80_q"] = p96, q96, p80, q80
    with (OUT / "lex80_refit_reader.pkl").open("wb") as fh:
        pickle.dump(lex80, fh)
    with (OUT / "grain_pole_readout.pkl").open("wb") as fh:
        pickle.dump({"scaler": scaler, "ridge": ridge}, fh)
    typed = bank[["material_id", "rk", "pole", "qualification", "subject_status", "variant", "text_sha256", "grain_class"]].copy()
    typed["nli_p"], typed["nli_q"] = bank.pole_readout, bank.nli_qualification
    typed["grain_q_hist"], typed["grain_pole_hist"] = bank.grain_qualification, bank.grain_pole
    for c in ("grainridge_p_raw", "grainridge_p", "lex96_p", "lex96_q", "lex80_p", "lex80_q"):
        typed[c] = bank[c]
    typed.to_csv(OUT / "typed_reader_table.csv", index=False)
    # reader-level accuracy per role
    acc = []
    pairs = {"nli": ("nli_p", "nli_q"), "hybrid": ("nli_p", "grain_q_hist"), "grainpole": ("grainridge_p", "grain_q_hist"),
             "lex96": ("lex96_p", "lex96_q"), "lex80": ("lex80_p", "lex80_q")}
    for rk, g in typed.groupby("rk"):
        for name, (pc, qc) in pairs.items():
            qual = g[g.qualification == 1]
            acc.append({"role": rk, "reader": name, "n": len(g),
                        "pole_acc_qualified": float(((qual[pc] >= .5).astype(int) == qual.pole).mean()),
                        "pole_brier_qualified": float(((qual[pc] - qual.pole) ** 2).mean()),
                        "q_acc": float(((g[qc] >= .5).astype(int) == g.qualification).mean()),
                        "q_brier": float(((g[qc] - g.qualification) ** 2).mean())})
    pd.DataFrame(acc).to_csv(OUT / "reader_accuracy_by_role.csv", index=False)
    inv = {"status": "PASS" if not stop else "STOP_INPUT", "stop_reasons": stop, "protocol_sha256": PROTOCOL_SHA,
           "inputs": {"bank_reader_scores": sha(BANK / "reader_scores.csv"), "bank_materials": sha(BANK / "materials.csv"),
                      "grain_features": sha(BANK / "grain_features.npz"), "lex96_reader_pkl": sha(LEX / "reader.pkl"),
                      "lex96_reader_scores": sha(LEX / "reader_scores.csv"),
                      "grain_head_bundle": sha(RES / "F11/grain_comment_reader_r1/readout_bundle.json")},
           "role_counts": counts, "cross_role_text_overlap": overlap, "grain_collisions": collisions,
           "grain_ridge": {"fit_rows": int(src.sum()), "alpha": 1.0, "raw_range": [float(graw.min()), float(graw.max())],
                           "coef_norm": float(np.linalg.norm(ridge.coef_))},
           "lex80_fit": {"fit_rows": int(src.sum()), "n_features": int(lex80["n_features"]),
                         "fit_text_sha256": hashlib.sha256("\n".join(texts[i] for i in np.where(src)[0]).encode()).hexdigest()},
           "typed_table_sha256": sha(OUT / "typed_reader_table.csv"), "neural_forwards": 0}
    wj(OUT / "inventory.json", inv)
    return inv


# ---------------------------------------------------------------- shared loop
def author_metrics(readings_by_channel, ys, fitted, initial, source_mean, panel, world, card=None):
    """Canonical QualifiedHistory update loop; returns per-channel per-candidate (brier, logloss, nextp_se)."""
    tw = "xor" if world == "c2_xor" else "iid"
    out = {}
    for ch, readings in readings_by_channel.items():
        h = QualifiedHistory(fitted, initial, source_mean, mode="independent" if panel == "complete" else "filtered")
        losses = {c: [] for c in CANDS}
        nextp = {c: [] for c in CANDS}
        for t, y in enumerate(ys):
            before = h.predict() if t >= 2 else None
            if t >= 12:
                pt = c2_next_probability(ys[t - 2], ys[t - 1], 1, tw)
                for c, p in before["probabilities"].items():
                    # identical to the canonical F08 downstream: a (brier, logloss) 2-tuple list averaged with np.mean(axis=0)
                    losses[c].append(((p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log1p(-p))))
                    nextp[c].append((p - pt) ** 2)
            h.observe(*readings[t])
            if card is not None and t in card["ts"]:
                card["updates"].setdefault(t, {})[ch] = {"reading": list(readings[t]), "before": None if before is None else before["probabilities"],
                                                          "after": h.predict()["probabilities"] if t >= 1 else None}
        out[ch] = {c: (*np.mean(losses[c], axis=0), float(np.mean(nextp[c]))) for c in CANDS}
    return out


def source_state(records):
    source = [r for r in records if r["role"] == "source"]
    fitted = fit_observed_source(source, None or records[0]["world"] if "world" in records[0] else None) if False else None
    return source


def fit_source(records, world):
    source = [r for r in records if r["role"] == "source"]
    fitted = fit_observed_source(source, world)
    initial = np.ones((2, 2))
    for group in ordered_rows(source).values():
        initial[group[0]["observation"], group[1]["observation"]] += 1
    initial /= initial.sum()
    return source, fitted, initial, float(np.mean([r["observation"] for r in source]))


# ---------------------------------------------------------------- replay gate
def replay() -> dict:
    cache = {(r["role"], int(r["pole"]), r["subject_status"], int(r["variant"])): r for r in csv.DictReader((LEX / "reader_scores.csv").open())}
    arch = pd.read_csv(DOWN / "metrics_by_author.csv", float_precision="round_trip")  # exact float parsing for the exact gate
    rows = []
    seed = SEEDS[0]
    for wi, world in enumerate(WORLDS):
        records, _ = generate_world(world, seed, counts=COUNTS)
        _s, fitted, initial, smean = fit_source(records, world)
        for ai, (author, arows) in enumerate(ordered_rows([r for r in records if r["role"] != "source"]).items()):
            role = arows[0]["role"]
            rng = np.random.default_rng(np.random.SeedSequence([seed, 8, wi, ai]))
            statuses, variants, anc = rng.integers(0, 4, 48), rng.integers(0, 4, 48), rng.integers(0, 2, 48)
            ys = [r["observation"] for r in arows]
            for panel in ("complete", "mixed"):
                rd = {c: [] for c in ("oracle", "soft", "hard")}
                for t, y in enumerate(ys):
                    status = STATUSES[statuses[t]] if panel == "mixed" else ("actual" if statuses[t] % 2 == 0 else "negated")
                    q = int(status in ("actual", "negated"))
                    mp = y if q else int(anc[t])
                    m = cache[(role, mp, status, int(variants[t]))]
                    p, qual = float(m["pole_score"]), float(m["qualification_score"])
                    r = {"oracle": (float(mp), float(q)), "soft": (p, qual), "hard": (float(p >= .5), float(qual >= .5))}
                    if panel == "complete":
                        r = {c: (v[0], 1.0) for c, v in r.items()}
                    for c in rd:
                        rd[c].append(r[c])
                res = author_metrics(rd, ys, fitted, initial, smean, panel, world)
                for ch, cands in res.items():
                    for cand, (b, ll, _np) in cands.items():
                        rows.append({"world": world, "seed": seed, "author_id": author, "panel": panel, "channel": ch, "candidate": cand, "brier_new": b, "ll_new": ll})
    new = pd.DataFrame(rows)
    m = new.merge(arch, on=["world", "seed", "author_id", "panel", "channel", "candidate"], how="left", validate="one_to_one")
    missing = int(m.brier.isna().sum())
    maxb = float(np.nanmax(np.abs(m.brier_new - m.brier)))
    maxl = float(np.nanmax(np.abs(m.ll_new - m.log_loss)))
    res = {"rows": len(m), "missing_archive_rows": missing, "max_abs_brier_diff": maxb, "max_abs_logloss_diff": maxl,
           "status": "PASS" if missing == 0 and maxb == 0.0 and maxl == 0.0 else "STOP_REPLAY", "scope": "seed 20260915, both worlds, both panels, oracle/soft/hard, validation+application authors"}
    wj(OUT / "replay_check.json", res)
    return res


# ---------------------------------------------------------------- transport rendering
def load_typed():
    typed = pd.read_csv(OUT / "typed_reader_table.csv")
    rd = {}
    for r in typed.itertuples(index=False):
        rd[r.material_id] = {
            "oracle": (float(r.pole), float(r.qualification)),
            "nli_soft": (float(r.nli_p), float(r.nli_q)),
            "nli_hard": (float(r.nli_p >= .5), float(r.nli_q >= .5)),
            "hybrid_nlipole_grainq": (float(r.nli_p >= .5), float(r.grain_q_hist >= .5)),
            "grainpole_grainq": (float(r.grainridge_p >= .5), float(r.grain_q_hist >= .5)),
            "lex96_transported": (float(r.lex96_p >= .5), float(r.lex96_q >= .5)),
            "lex80_refit": (float(r.lex80_p >= .5), float(r.lex80_q >= .5)),
        }
    strata = {}
    for r in typed.sort_values("material_id").itertuples(index=False):
        strata.setdefault((r.rk, int(r.pole), r.subject_status), []).append(r.material_id)
    return typed, rd, strata


def render_transport(seq_ids, ys, panel, rk, strata):
    rng = np.random.default_rng(np.random.SeedSequence(seq_ids))
    statuses, _variants, anc = rng.integers(0, 4, 48), rng.integers(0, 4, 48), rng.integers(0, 2, 48)
    u = np.random.default_rng(np.random.SeedSequence([*seq_ids, MSTREAM])).random(48)
    events = []
    for t, y in enumerate(ys):
        status = STATUSES[statuses[t]] if panel == "mixed" else ("actual" if statuses[t] % 2 == 0 else "negated")
        q = int(status in ("actual", "negated"))
        mp = y if q else int(anc[t])
        mats = strata[(rk, mp, status)]
        events.append({"t": t, "status": status, "q_meta": q, "material_pole": mp, "material_id": mats[int(u[t] * len(mats))]})
    return events


def readings_for(events, rd, panel, channels):
    out = {}
    for ch in channels:
        seq = [rd[e["material_id"]][ch] for e in events]
        if panel == "complete":
            seq = [(p, 1.0) for p, _q in seq]
        out[ch] = seq
    return out


def run_world_seed(world, seed):
    wi = WORLDS.index(world)
    typed, rd, strata = load_typed()
    records, _ = generate_world(world, seed, counts=COUNTS)
    source, fitted, initial, smean = fit_source(records, world)
    rows, cards = [], []
    for ai, (author, arows) in enumerate(ordered_rows([r for r in records if r["role"] != "source"]).items()):
        role = arows[0]["role"]
        rk = "application" if role == "application" else "pilot_validation"
        ys = [r["observation"] for r in arows]
        for panel in ("complete", "mixed"):
            events = render_transport([seed, 8, wi, ai], ys, panel, rk, strata)
            card = None
            if world == "c2_xor" and seed == SEEDS[0] and role == "application" and panel == "mixed" and not cards:
                first = {}
                for e in events:
                    if e["t"] >= 2 and e["status"] not in first:
                        first[e["status"]] = e["t"]
                card = {"author_id": author, "ts": set(first.values()), "events": {t: events[t] for t in first.values()}, "updates": {}}
            res = author_metrics(readings_for(events, rd, panel, CHANNELS), ys, fitted, initial, smean, panel, world, card)
            if card is not None:
                cards.append(card)
            for ch, cands in res.items():
                for cand, (b, ll, npse) in cands.items():
                    rows.append({"world": world, "seed": seed, "author_id": author, "role": role, "panel": panel, "channel": ch, "candidate": cand,
                                 "brier": b, "log_loss": ll, "nextp_se": npse})
    bundle = {"world": world, "seed": seed, "fitted": fitted, "initial": initial, "source_mean": smean}
    return rows, cards, bundle


def main_stage():
    inv = json.loads((OUT / "inventory.json").read_text())
    rp = json.loads((OUT / "replay_check.json").read_text())
    if inv["status"] != "PASS" or rp["status"] != "PASS":
        raise RuntimeError("inventory or replay gate did not pass")
    t0 = time.time()
    jobs = [(w, s) for w in WORLDS for s in SEEDS]
    results = Parallel(n_jobs=2)(delayed(run_world_seed)(w, s) for w, s in jobs)
    rows, cards = [], []
    (OUT / "source_bundles").mkdir(exist_ok=True)
    for (w, s), (r, c, b) in zip(jobs, results):
        rows += r
        cards += c
        with (OUT / "source_bundles" / f"oracle_source_{w}_{s}.pkl").open("wb") as fh:
            pickle.dump(b, fh)
    pd.DataFrame(rows).to_csv(OUT / "metrics_by_author.csv", index=False)
    typed = pd.read_csv(OUT / "typed_reader_table.csv").set_index("material_id")
    out_cards = []
    for c in cards:
        for t, e in sorted(c["events"].items()):
            mid = e["material_id"]
            out_cards.append({"author_id": c["author_id"], "t": t, "status": e["status"], "q_meta": e["q_meta"], "material_pole": e["material_pole"],
                              "material_id": mid, "text": pd.read_csv(BANK / "materials.csv").set_index("material_id").loc[mid, "text"],
                              "grain_class": typed.loc[mid, "grain_class"][:12],
                              "channel_updates": {ch: c["updates"][t][ch] for ch in c["updates"][t]}})
    wj(OUT / "evidence_cards.json", out_cards)
    return {"rows": len(rows), "seconds": time.time() - t0}


# ---------------------------------------------------------------- secondary axis
def emission_law_pilot(typed, ch_cols):
    """P(r | y), r in {missing, read0, read1}, under the render law, uniform over pilot-fit-60 stratum materials."""
    pc, qc = ch_cols
    g = typed[typed.rk == "pilot_fit"]
    law = np.zeros((2, 3))
    for y in (0, 1):
        for status in STATUSES:
            poles = (y,) if status in ("actual", "negated") else (0, 1)
            for pole in poles:
                s = g[(g.pole == pole) & (g.subject_status == status)]
                w = 0.25 / len(poles)
                qread = s[qc] >= .5
                law[y, 0] += w * float((~qread).mean())
                law[y, 1] += w * float((qread & (s[pc] < .5)).mean())
                law[y, 2] += w * float((qread & (s[pc] >= .5)).mean())
    return law


SEC_COLS = {"nli_hard": ("nli_p", "nli_q"), "hybrid_nlipole_grainq": ("nli_p", "grain_q_hist"), "grainpole_grainq": ("grainridge_p", "grain_q_hist"),
            "lex96_transported": ("lex96_p", "lex96_q"), "lex80_refit": ("lex80_p", "lex80_q")}


def run_secondary(world, seed):
    from run_grain_gptpro_f08_mnar_source import em_fit, fitted_from_counts
    wi = WORLDS.index(world)
    typed, rd, strata = load_typed()
    records, _ = generate_world(world, seed, counts=COUNTS)
    source, ofit, oinit, omean = fit_source(records, world)
    src_groups = ordered_rows(source)
    src_events = {a: render_transport([seed, 8, 100 + wi, si], [r["observation"] for r in rows], "mixed", "source", strata)
                  for si, (a, rows) in enumerate(src_groups.items())}
    rows_out, diags = [], []
    apps = [(ai, a, r) for ai, (a, r) in enumerate(ordered_rows([r for r in records if r["role"] != "source"]).items()) if r[0]["role"] == "application"]
    for ch, cols in SEC_COLS.items():
        law = emission_law_pilot(typed, cols)
        codes = []
        for a in src_groups:
            code = []
            for e in src_events[a]:
                p, q = rd[e["material_id"]][ch]
                code.append(0 if q < .5 else 1 + int(p >= .5))
            codes.append(code)
        em = em_fit(np.asarray(codes), law)
        fitted = fitted_from_counts(em["counts"], world)
        init = em["init"].reshape(2, 2)
        diags.append({"world": world, "seed": seed, "channel": ch, "law_pilot_fit60": law.tolist(), "iterations": em["iterations"],
                      "max_abs_b1_shift_vs_oracle": float(np.abs(fitted["b1"]["source_p"] - ofit["b1"]["source_p"]).max()),
                      "y_mean": em["y_mean"], "oracle_y_mean": omean})
        for ai, a, arows in apps:
            ys = [r["observation"] for r in arows]
            ev = render_transport([seed, 8, wi, ai], ys, "mixed", "application", strata)
            res = author_metrics(readings_for(ev, rd, "mixed", [ch]), ys, fitted, init, em["y_mean"], "mixed", world)
            for cand, (b, ll, npse) in res[ch].items():
                rows_out.append({"world": world, "seed": seed, "author_id": a, "channel": ch, "arm": "em_pilot_law", "candidate": cand,
                                 "brier": b, "log_loss": ll, "nextp_se": npse})
    return rows_out, diags


def secondary_stage():
    t0 = time.time()
    jobs = [(w, s) for w in WORLDS for s in SEEDS]
    res = Parallel(n_jobs=2)(delayed(run_secondary)(w, s) for w, s in jobs)
    rows = [r for rr, _d in res for r in rr]
    diags = [d for _r, dd in res for d in dd]
    pd.DataFrame(rows).to_csv(OUT / "secondary_em_metrics_by_author.csv", index=False)
    wj(OUT / "secondary_em_diagnostics.json", {"disclosure": "pooled E-step (not author-level); M known limited under pooled E-step; law fit on pilot-fit-60 only",
                                               "diagnostics": diags})
    return {"rows": len(rows), "seconds": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "replay", "main", "secondary"], required=True)
    st = ap.parse_args().stage
    t0 = time.time()
    out = {"prepare": prepare, "replay": replay, "main": main_stage, "secondary": secondary_stage}[st]()
    runtime = json.loads((OUT / "runtime.json").read_text()) if (OUT / "runtime.json").exists() else {}
    runtime[st] = {"seconds": time.time() - t0, "finished": time.strftime("%H:%M:%S")}
    wj(OUT / "runtime.json", runtime)
    print(json.dumps(out, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
