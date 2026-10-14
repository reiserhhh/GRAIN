#!/usr/bin/env python
"""R14 soft-channel and event-trace companion (docs/GRAIN_R14_SOFT_CHANNEL_TRACE_PROTOCOL_V1.md). Zero neural calls.

The r1 same-object runner is imported read-only for its reader table, rendering and source fits. Nothing in
same_object_channels_r1 is modified.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("r14", ROOT / "scripts/run_grain_r14_same_object_channels.py")
r14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r14)
from grain_research.gptpro_foundation_v3.f02 import c2_next_probability, generate_world  # noqa: E402
from grain_research.gptpro_foundation_v3.f08_filter import filter_step  # noqa: E402
from grain_research.measurement_discovery.estimators import ordered_rows  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R14_SOFT_CHANNEL_TRACE_PROTOCOL_V1.md"
PROTOCOL_SHA = "2ff7a5d9d24195702bfc5c068291602922baa1a443d14451fe4eafe181d79f67"
R1 = ROOT / "results/grain_gptpro_foundation_v3/R14/same_object_channels_r1"
OUT = ROOT / "results/grain_gptpro_foundation_v3/R14/soft_channel_trace_r1"
ARCHIVED = r14.CHANNELS
SOFT = {"hybrid_soft": ("nli_p", "grain_q_hist"), "grainpole_soft": ("grainridge_p", "grain_q_hist"),
        "lex96_soft": ("lex96_p", "lex96_q"), "lex80_soft": ("lex80_p", "lex80_q")}
NOINFO = "grain_noinfo_reference"
ALL = (*ARCHIVED, *SOFT, NOINFO)
CANDS = r14.CANDS


class NoInfoHistory(r14.QualifiedHistory):
    """Advance the event clock and prior law; add no observation-driven statistics (declared reference)."""

    def observe(self, pole, qualification):  # noqa: ARG002 - readings deliberately ignored
        if self.t >= 2:
            self.pair = filter_step(self.pair, self.transition, 0.5, 0.0)["pair"]
        self.t += 1


def readings(events, rd, typed_idx, panel):
    out = {ch: [rd[e["material_id"]][ch] for e in events] for ch in ARCHIVED}
    for ch, (pc, qc) in SOFT.items():
        out[ch] = [(float(typed_idx.at[e["material_id"], pc]), float(typed_idx.at[e["material_id"], qc])) for e in events]
    out[NOINFO] = [(0.5, 0.0) for _ in events]
    if panel == "complete":
        out = {ch: [(p, 1.0) for p, _q in seq] for ch, seq in out.items()}
    return out


def run_world_seed(world, seed):
    wi = r14.WORLDS.index(world)
    typed, rd, strata = r14.load_typed()
    tix = typed.set_index("material_id")
    text_sha = tix["text_sha256"].str[:12].to_dict()
    tw = "xor" if world == "c2_xor" else "iid"
    records, _ = generate_world(world, seed, counts=r14.COUNTS)
    _src, fitted, initial, smean = r14.fit_source(records, world)
    rows, means = [], []
    for ai, (author, arows) in enumerate(ordered_rows([r for r in records if r["role"] != "source"]).items()):
        role = arows[0]["role"]
        rk = "application" if role == "application" else "pilot_validation"
        ys = [r["observation"] for r in arows]
        for panel in ("complete", "mixed"):
            events = r14.render_transport([seed, 8, wi, ai], ys, panel, rk, strata)
            rds = readings(events, rd, tix, panel)
            for ch in ALL:
                cls = NoInfoHistory if ch == NOINFO else r14.QualifiedHistory
                h = cls(fitted, initial, smean, mode="independent" if panel == "complete" else "filtered")
                losses, nextp = {c: [] for c in CANDS}, {c: [] for c in CANDS}
                for t, y in enumerate(ys):
                    before = h.predict() if t >= 2 else None
                    pt = c2_next_probability(ys[t - 2], ys[t - 1], 1, tw) if t >= 2 else float("nan")
                    if t >= 12:
                        for c, p in before["probabilities"].items():
                            losses[c].append(((p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log1p(-p))))
                            nextp[c].append((p - pt) ** 2)
                    p_read, q_read = rds[ch][t]
                    h.observe(p_read, q_read)
                    e = events[t]
                    row = {"world": world, "seed": seed, "role": role, "author_id": author, "panel": panel, "channel": ch, "t": t,
                           "event_id": arows[t]["event_id"], "material_id": e["material_id"], "text_sha12": text_sha[e["material_id"]],
                           "status": e["status"], "q_meta": e["q_meta"], "material_pole": e["material_pole"], "p_read": p_read, "q_read": q_read,
                           "y": y, "p_true": pt, "evidence_mass_after": float(h.mass), "transition_mass_after": float(h.counts.sum()),
                           "pred_status": None if before is None else before["status"]}
                    for c in CANDS:
                        row[f"pred_{c}"] = float("nan") if before is None else before["probabilities"][c]
                    rows.append(row)
                for c in CANDS:
                    b, ll = np.mean(losses[c], axis=0)
                    means.append({"world": world, "seed": seed, "author_id": author, "role": role, "panel": panel, "channel": ch, "candidate": c,
                                  "brier": b, "log_loss": ll, "nextp_se": float(np.mean(nextp[c]))})
    df = pd.DataFrame(rows)
    df["address_sha16"] = [hashlib.sha256(f"{w}|{s}|{a}|{p}|{t}|{m}".encode()).hexdigest()[:16]
                           for w, s, a, p, t, m in zip(df.world, df.seed, df.author_id, df.panel, df.t, df.material_id)]
    return df, pd.DataFrame(means), {"world": world, "seed": seed, "fitted_b1": np.asarray(fitted["b1"]["source_p"]).tolist(), "initial": initial.tolist()}


def main() -> int:
    t0 = time.time()
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    inv = json.loads((R1 / "inventory.json").read_text())
    if hashlib.sha256((R1 / "typed_reader_table.csv").read_bytes()).hexdigest() != inv["typed_table_sha256"]:
        raise RuntimeError("typed reader table hash mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(w, s) for w in r14.WORLDS for s in r14.SEEDS]
    res = Parallel(n_jobs=2)(delayed(run_world_seed)(w, s) for w, s in jobs)
    trace = pd.concat([r[0] for r in res], ignore_index=True)
    means = pd.concat([r[1] for r in res], ignore_index=True)
    srcs = [r[2] for r in res]
    trace.to_parquet(OUT / "event_trace.parquet", compression="zstd", index=False)
    means.to_csv(OUT / "metrics_by_author.csv", index=False)
    (OUT / "source_priors.json").write_text(json.dumps(srcs, indent=1) + "\n")
    # ---- check 1: reconstruction vs archived r1 author means (tolerance 1e-12)
    arch = pd.read_csv(R1 / "metrics_by_author.csv", float_precision="round_trip")
    m = means[means.channel.isin(ARCHIVED)].merge(arch, on=["world", "seed", "author_id", "role", "panel", "channel", "candidate"], suffixes=("", "_arch"), validate="one_to_one")
    rec = {"rows_compared": len(m), "archived_rows": len(arch),
           "max_abs_diff": {k: float(np.abs(m[k] - m[f"{k}_arch"]).max()) for k in ("brier", "log_loss", "nextp_se")}}
    tr = trace[(trace.t >= 12) & trace.channel.isin(ARCHIVED)].copy()
    for c in CANDS:
        tr[f"b_{c}"] = (tr[f"pred_{c}"] - tr.y) ** 2
    fromtrace = tr.groupby(["world", "seed", "author_id", "panel", "channel"])[[f"b_{c}" for c in CANDS]].mean().reset_index().melt(
        id_vars=["world", "seed", "author_id", "panel", "channel"], var_name="candidate", value_name="brier_trace")
    fromtrace["candidate"] = fromtrace.candidate.str[2:]
    m2 = fromtrace.merge(arch, on=["world", "seed", "author_id", "panel", "channel", "candidate"], validate="one_to_one")
    rec["trace_brier_max_abs_diff"] = float(np.abs(m2.brier_trace - m2.brier).max())
    rec["status"] = "PASS" if rec["rows_compared"] == len(arch) and max(rec["max_abs_diff"].values()) <= 1e-12 and rec["trace_brier_max_abs_diff"] <= 1e-12 else "STOP_RECONSTRUCTION"
    # ---- check 2: flatness
    flat = {}
    for q in (0.0, 0.3, 1.0):
        like = q * np.array([0.5, 0.5]) + (1 - q) / 2
        flat[str(q)] = float(abs(like[0] - like[1]))
    mix = trace[(trace.panel == "mixed") & trace.channel.isin(["grainpole_soft", NOINFO]) & (trace.t >= 2)]
    piv = mix.pivot_table(index=["world", "seed", "author_id", "t"], columns="channel", values="pred_b1")
    flat["mixed_B1_soft05_minus_noinfo_max_abs"] = float(np.abs(piv["grainpole_soft"] - piv[NOINFO]).max())
    # ---- check 3: independent prior-law propagation for 10 declared authors (first 10 application authors, xor seed 1)
    indep = []
    s0 = srcs[0]
    T = np.asarray(s0["fitted_b1"])  # P(next=1 | a,b), index 2a+b
    ni = trace[(trace.channel == NOINFO) & (trace.world == s0["world"]) & (trace.seed == s0["seed"]) & (trace.role == "application") & (trace.panel == "mixed")]
    for author in sorted(ni.author_id.unique())[:10]:
        pair = np.asarray(s0["initial"], float)
        g = ni[ni.author_id == author].sort_values("t")
        for t in range(48):
            if t >= 2:
                pred = float(pair.ravel() @ T)
                indep.append(abs(np.clip(pred, 1e-6, 1 - 1e-6) - g.pred_b1.iloc[t]))
                new = np.zeros((2, 2))
                for a in (0, 1):
                    for b in (0, 1):
                        p1 = T[2 * a + b]
                        new[b, 0] += pair[a, b] * (1 - p1)
                        new[b, 1] += pair[a, b] * p1
                pair = new / new.sum()
    # ---- check 4: render
    rend = {"application_materials_only": bool(trace[trace.role == "application"].material_id.str.startswith("transport_application").all()),
            "validation_materials_pilot_only": bool(trace[trace.role == "validation"].material_id.str.startswith("transport_pilot").all()),
            "status_counts_application_mixed": trace[(trace.role == "application") & (trace.panel == "mixed") & (trace.channel == "oracle")].status.value_counts().to_dict()}
    checks = {"reconstruction": rec, "flatness": flat, "independent_prior_law_max_abs": float(max(indep)), "render": rend,
              "trace_rows": len(trace), "trace_sha256": hashlib.sha256((OUT / "event_trace.parquet").read_bytes()).hexdigest()}
    checks["status"] = "PASS" if (rec["status"] == "PASS" and flat["mixed_B1_soft05_minus_noinfo_max_abs"] <= 1e-15 and checks["independent_prior_law_max_abs"] <= 1e-12
                                  and rend["application_materials_only"] and rend["validation_materials_pilot_only"]) else "FAIL"
    (OUT / "checks.json").write_text(json.dumps(checks, indent=1, default=str) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"seconds": time.time() - t0, "neural_calls": 0, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}) + "\n")
    print(json.dumps(checks, indent=1, default=str))
    return 0 if checks["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
