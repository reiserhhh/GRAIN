#!/usr/bin/env python3
"""GRAIN-C2-01 runner: source-calibrated evidence-unit law, cold finite-mixture world."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import evidence_unit as EU
from grain_research.gptpro_foundation_v3 import f08

PY = sys.executable
PROTOCOL = ROOT / "docs/GRAIN_C2_01_EVIDENCE_UNIT_PROTOCOL_V1.json"
F08 = ROOT / "results/grain_gptpro_foundation_v3/F08"
OUT = ROOT / "results/grain_construction_v1/C2/evidence_unit_r1"
SEEDS = (20261021, 20261022, 20261023)
KAPPAS = (1.0, 0.5, 0.25)
N_AUTH, N_EV = 200, 24
DIR_SEED, BOOT_SEED, B_AUTH, B_MAT = 20261020, 20261019, 500, 2000
LEX, NLI = EU.READERS


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def tv(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum(-1)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    t0, c0 = time.time(), time.process_time()
    proto = json.loads(PROTOCOL.read_text())
    (OUT / "protocol_before_run.json").write_text(json.dumps({**proto, "protocol_sha256": sha(PROTOCOL)}, indent=1))

    # ---------------------------------------------------------------- 1 inventory
    lex = pd.read_csv(F08 / "reader_recovery_r1/reader_scores.csv")
    nli = pd.read_csv(F08 / "nli_reader_r1/reader_scores.csv")
    design = ["material_id", "role", "family", "pole", "subject_status", "qualification", "variant", "text"]
    match = {c: bool((lex[c] == nli[c]).all()) for c in design}
    if not all(match.values()):
        raise ValueError(f"cached banks do not match: {match}")
    if not (lex.family == lex.role.map(EU.FAMILY_OF_ROLE)).all():
        raise ValueError("family-role mapping changed")
    if not ((lex.qualification == lex.subject_status.isin(EU.QUALIFIED).astype(int)).all()):
        raise ValueError("qualification label inconsistent with status")
    files = {k: {"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for k, p in {
        "lexical_scores": F08 / "reader_recovery_r1/reader_scores.csv",
        "lexical_bank": F08 / "reader_recovery_r1/grammar_bank.csv",
        "lexical_reader_pkl": F08 / "reader_recovery_r1/reader.pkl",
        "nli_scores": F08 / "nli_reader_r1/reader_scores.csv",
        "nli_bank": F08 / "nli_reader_r1/grammar_bank.csv",
        "nli_protocol": F08 / "nli_reader_r1/protocol_before_inference.json"}.items()}
    lb, nb = pd.read_csv(F08 / "reader_recovery_r1/grammar_bank.csv"), pd.read_csv(F08 / "nli_reader_r1/grammar_bank.csv")
    bank_content_equal = bool(lb[design].equals(nb[design]))

    # lexical provenance + leave-one-variant-out source readings (frozen f08 constants)
    src = lex[lex.role == "source"].reset_index(drop=True)
    full = f08.fit_reader(src.text.tolist(), src.pole.tolist(), src.qualification.tolist())
    pf, qf = f08.predict_reader(full, src.text.tolist())
    repro = {"max_abs_diff_pole_vs_cache": float(np.max(np.abs(pf - src.pole_score))),
             "max_abs_diff_qual_vs_cache": float(np.max(np.abs(qf - src.qualification_score)))}
    oof_p, oof_q = np.zeros(len(src)), np.zeros(len(src))
    for v in sorted(src.variant.unique()):
        tr, te = src.variant != v, src.variant == v
        rd = f08.fit_reader(src.text[tr].tolist(), src.pole[tr].tolist(), src.qualification[tr].tolist())
        p, q = f08.predict_reader(rd, src.text[te].tolist())
        oof_p[te.to_numpy()], oof_q[te.to_numpy()] = p, q
    src_lex_oof = src.assign(pole_score=oof_p, qualification_score=oof_q)

    def rows_of(df):
        return [{"material_id": r.material_id, "role": r.role, "pole": int(r.pole), "subject_status": r.subject_status,
                 "variant": int(r.variant), "category": int(EU.category(r.pole_score, r.qualification_score))}
                for r in df.itertuples()]

    source_rows = {LEX: rows_of(src_lex_oof), NLI: rows_of(nli[nli.role == "source"]),
                   "lexical_insample": rows_of(src)}
    manifest = {"task": "GRAIN-C2-01", "materials": 96, "roles": EU.FAMILY_OF_ROLE, "files": files,
                "design_fields_match_across_caches": match,
                "grammar_bank_files": {"content_equal_on_design_fields": bank_content_equal,
                                       "byte_hash_equal": files["lexical_bank"]["sha256"] == files["nli_bank"]["sha256"]},
                "per_role_status_counts": {r: g.subject_status.value_counts().to_dict() for r, g in lex.groupby("role")},
                "variants_per_class_per_family": 4,
                "missing_fields": ["writer_id (none; material_id/variant are wrapper IDs)", "author or person ID",
                                   "graded intensity", "register", "assigned condition", "multiple propositions",
                                   "shared representation with C1 vectors"],
                "lexical_reader_reproduction": repro,
                "lexical_source_law": "leave-one-variant-out refit (f08.fit_reader constants); in-sample cache kept as ablation",
                "label_kinds": {"designed": ["pole", "subject_status", "qualification", "family", "variant"],
                                "reader_score": ["pole_score", "qualification_score"],
                                "personal_observation": "none in the bank; simulated author events in the finite world",
                                "unobserved": ["writer", "author propensity outside the world", "exposure/opportunity"]},
                "reuse_note": "cached readings reused; not a new replication; no neural call in this packet"}
    (OUT / "pilot_manifest.json").write_text(json.dumps(manifest, indent=1))

    # ---------------------------------------------------------------- 2 observation law + bundles
    law, bundles, bundle_sha = {}, {}, {}
    for key, rows in source_rows.items():
        reader = LEX if key == "lexical_insample" else key
        b = EU.build_bundle(rows, reader, DIR_SEED, {"source_rows": len(rows), "variant": key})
        name = {LEX: "bundle_lexical.json", NLI: "bundle_nli.json", "lexical_insample": "bundle_lexical_insample_ablation.json"}[key]
        bundle_sha[key] = EU.save_bundle(b, OUT / name)
        bundles[key] = b
    diag_channels = {}
    for reader, df in ((LEX, lex), (NLI, nli)):
        fam = {}
        for role in ("source", "validation", "application"):
            rr = rows_of(df[df.role == role]) if not (reader == LEX and role == "source") else source_rows[LEX]
            n = EU.channel_counts(rr)
            freq = n / n.sum(-1, keepdims=True)
            sv = np.linalg.svd(freq.reshape(8, 4), compute_uv=False)
            fam[role] = {"counts": n.tolist(), "rank": int(np.sum(sv > 1e-8)), "singular_values": sv.round(6).tolist(),
                         "tv_actual_vs_quoted_by_pole": tv(freq[:, 0], freq[:, 1]).round(4).tolist(),
                         "tv_negated_vs_hypothetical_by_pole": tv(freq[:, 3], freq[:, 2]).round(4).tolist(),
                         "tv_pole1_vs_pole0_by_status": tv(freq[1], freq[0]).round(4).tolist()}
            diag_channels[(reader, role)] = freq
        for role in ("validation", "application"):
            d = tv(diag_channels[(reader, "source")], diag_channels[(reader, role)])
            fam[role]["tv_to_source_by_row(pole x status)"] = d.round(4).tolist()
            fam[role]["tv_to_source_mean"] = float(d.mean())
        fam["role_of_held_channels"] = "EVALUATOR DIAGNOSTIC ONLY (designed labels); never a scoring input"
        law[reader] = fam
    law["lexical_insample_source"] = {"counts": bundles["lexical_insample"]["channel_counts"]}
    law["bundles_sha256"] = bundle_sha
    (OUT / "observation_law.json").write_text(json.dumps(law, indent=1))

    # ---------------------------------------------------------------- 3 held material checks
    held, contr = [], []
    rng = np.random.default_rng(np.random.SeedSequence([DIR_SEED, 7]))
    for reader, df in ((LEX, lex), (NLI, nli)):
        for r in df.itertuples():
            c = int(EU.category(r.pole_score, r.qualification_score))
            held.append({"reader": reader, "material_id": r.material_id, "role": r.role, "family": r.family,
                         "pole": r.pole, "subject_status": r.subject_status, "qualification": r.qualification,
                         "variant": r.variant, "pole_score": r.pole_score, "qualification_score": r.qualification_score,
                         "category": c, "pole_correct": int((c >= 2) == bool(r.pole)),
                         "qualification_correct": int((c % 2 == 1) == bool(r.qualification))})
        for role in ("source", "validation", "application"):
            g = df[df.role == role].reset_index(drop=True)
            if reader == LEX and role == "source":
                g = src_lex_oof
            stats = {
                "qualified_pole_contrast": lambda h: h[(h.qualification == 1) & (h.pole == 1)].pole_score.mean() - h[(h.qualification == 1) & (h.pole == 0)].pole_score.mean(),
                "unqualified_pole_contrast(leakage at material level)": lambda h: h[(h.qualification == 0) & (h.pole == 1)].pole_score.mean() - h[(h.qualification == 0) & (h.pole == 0)].pole_score.mean(),
                "qualification_direction": lambda h: h[h.qualification == 1].qualification_score.mean() - h[h.qualification == 0].qualification_score.mean(),
                "pole_brier": lambda h: float(np.mean((h.pole_score - h.pole) ** 2)),
                "qualification_brier": lambda h: float(np.mean((h.qualification_score - h.qualification) ** 2)),
            }
            for name, f in stats.items():
                boots = []
                for _ in range(B_MAT):
                    idx = []
                    for _, cell in g.groupby(["pole", "subject_status"]):
                        idx += list(rng.choice(cell.index, len(cell), replace=True))
                    boots.append(f(g.loc[idx]))
                contr.append({"reader": reader, "role": role, "family": EU.FAMILY_OF_ROLE[role], "statistic": name,
                              "value": float(f(g)), "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
                              "source_readings": "leave-one-variant-out" if (reader == LEX and role == "source") else "cached"})
    write_csv(OUT / "held_material_checks.csv", held)
    write_csv(OUT / "held_material_contrasts.csv", contr)
    stage1 = {"wall_s": time.time() - t0}

    # ---------------------------------------------------------------- 4 finite evidence-mixture world (evaluator)
    look = {}
    for reader, df in ((LEX, lex), (NLI, nli)):
        for r in df.itertuples():
            look[(reader, r.role, int(r.pole), r.subject_status, int(r.variant))] = (r.pole_score, r.qualification_score,
                                                                                       int(EU.category(r.pole_score, r.qualification_score)))
    sims = {}
    for seed in SEEDS:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2201]))
        pi = rng.beta(2, 2, N_AUTH)
        u_st, u_sub, u_pole = rng.random((N_AUTH, N_EV)), rng.random((N_AUTH, N_EV)), rng.random((N_AUTH, N_EV))
        var = rng.integers(0, 4, (N_AUTH, N_EV))
        for kappa in KAPPAS:
            qual = u_st < kappa
            status = np.where(qual, np.where(u_sub < 0.5, "actual", "negated"), np.where(u_sub < 0.5, "quoted", "hypothetical"))
            pole = np.where(qual, u_pole < pi[:, None], u_pole < 0.5).astype(int)
            for role in ("validation", "application"):
                for reader in (LEX, NLI):
                    cats = np.vectorize(lambda b, s, v: look[(reader, role, b, s, v)][2])(pole, status, var)
                    counts = np.stack([np.bincount(row, minlength=4) for row in cats]).astype(float)
                    sims[(seed, kappa, role, reader)] = {"counts": counts, "pi": pi,
                                                         "matched": (np.sum(qual & (pole == 1), 1), np.sum(qual & (pole == 0), 1)),
                                                         "pole": pole, "status": status, "var": var}

    # ---------------------------------------------------------------- 5 scoring (estimator inputs: counts only)
    chans = {}
    for key, b in bundles.items():
        n = np.asarray(b["channel_counts"])
        chans[(key, "point")] = EU.log_probabilities(EU.point_channel(n)[None])
        chans[(key, "dirichlet")] = EU.log_probabilities(EU.dirichlet_channels(n, DIR_SEED))
    for key, rows in source_rows.items():
        if key != "lexical_insample":
            chans[(key, "bootstrap")] = EU.log_probabilities(EU.variant_bootstrap_channels(rows, BOOT_SEED))
    other = {LEX: NLI, NLI: LEX}
    cold, results = [], {}
    for kappa in KAPPAS:
        for role in ("validation", "application"):
            for reader in (LEX, NLI):
                counts = np.concatenate([sims[(s, kappa, role, reader)]["counts"] for s in SEEDS])
                arms = {"calibrated_point": chans[(reader, "point")], "calibrated_dirichlet": chans[(reader, "dirichlet")],
                        "calibrated_variant_bootstrap": chans[(reader, "bootstrap")],
                        "wrong_reader_law": chans[(other[reader], "dirichlet")]}
                if reader == LEX:
                    arms["calibrated_dirichlet_insample"] = chans[("lexical_insample", "dirichlet")]
                for arm, logps in arms.items():
                    ppi, pka = EU.marginal_posteriors(counts, logps=logps)
                    m, v, lo, hi = EU.summarize_grid(ppi)
                    results[(kappa, role, reader, arm)] = (m, v, lo, hi, pka @ EU.KAPPA_GRID)
                m, v, lo, hi = EU.naive_scores(counts)
                results[(kappa, role, reader, "naive_uncalibrated")] = (m, v, lo, hi, (counts[:, 1] + counts[:, 3]) / N_EV)
                pos = np.concatenate([sims[(s, kappa, role, reader)]["matched"][0] for s in SEEDS])
                neg = np.concatenate([sims[(s, kappa, role, reader)]["matched"][1] for s in SEEDS])
                m, v, lo, hi = EU.beta_summary(pos, neg)
                results[(kappa, role, reader, "matched_observation_PRIVILEGED")] = (m, v, lo, hi, (pos + neg) / N_EV)
                for arm in [a for (k, r_, rd, a) in results if (k, r_, rd) == (kappa, role, reader)]:
                    m, v, lo, hi, km = results[(kappa, role, reader, arm)]
                    for i in range(len(m)):
                        cold.append([reader, arm, role, kappa, SEEDS[i // N_AUTH], i % N_AUTH, m[i], v[i], lo[i], hi[i], km[i]])
    with open(OUT / "cold_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reader", "arm", "family_role", "kappa", "seed", "author_index", "pi_mean", "pi_var", "pi_lo", "pi_hi", "kappa_mean"])
        w.writerows(cold)
    truth_rows = [[s, i, sims[(s, 1.0, "application", LEX)]["pi"][i]] for s in SEEDS for i in range(N_AUTH)]
    with open(OUT / "evaluator_truth.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "author_index", "pi_true"])
        w.writerows(truth_rows)

    # ---------------------------------------------------------------- 6 downstream comparison (evaluator)
    truth = np.concatenate([sims[(s, 1.0, "application", LEX)]["pi"] for s in SEEDS])
    rng = np.random.default_rng(np.random.SeedSequence([DIR_SEED, 9]))
    idx = [rng.integers(0, len(truth), len(truth)) for _ in range(B_AUTH)]
    down, cal, pairs = [], [], []
    for (kappa, role, reader, arm), (m, v, lo, hi, km) in results.items():
        err = m - truth
        cover = (truth >= lo) & (truth <= hi)
        mb = [np.mean(err[i] ** 2) for i in idx]
        down.append({"reader": reader, "arm": arm, "family_role": role, "kappa": kappa, "authors": len(m), "events_per_author": N_EV,
                     "sd_score": float(np.std(m)), "r": float(np.corrcoef(m, truth)[0, 1]), "mse": float(np.mean(err ** 2)),
                     "mse_lo": float(np.percentile(mb, 2.5)), "mse_hi": float(np.percentile(mb, 97.5)),
                     "bias": float(err.mean()), "bias_lo": float(np.percentile([err[i].mean() for i in idx], 2.5)),
                     "bias_hi": float(np.percentile([err[i].mean() for i in idx], 97.5)),
                     "coverage": float(cover.mean()), "mean_width": float(np.mean(hi - lo)),
                     "kappa_hat_mean": float(np.mean(km)), "kappa_error": float(np.mean(km) - kappa)})
        cal.append({"reader": reader, "arm": arm, "family_role": role, "kappa": kappa, "mean_width": float(np.mean(hi - lo)),
                    "coverage": float(cover.mean()), "mean_posterior_sd": float(np.mean(np.sqrt(v)))})
    for kappa in KAPPAS:
        for role in ("validation", "application"):
            for reader in (LEX, NLI):
                m0 = results[(kappa, role, reader, "calibrated_dirichlet")][0]
                for ref in ("naive_uncalibrated", "calibrated_point", "wrong_reader_law", "matched_observation_PRIVILEGED"):
                    m1 = results[(kappa, role, reader, ref)][0]
                    d = (m0 - truth) ** 2 - (m1 - truth) ** 2
                    db = [d[i].mean() for i in idx]
                    pairs.append({"reader": reader, "family_role": role, "kappa": kappa, "primary": "calibrated_dirichlet",
                                  "reference": ref, "mse_primary_minus_ref": float(d.mean()),
                                  "ci_lo": float(np.percentile(db, 2.5)), "ci_hi": float(np.percentile(db, 97.5)),
                                  "mean_abs_score_change": float(np.mean(np.abs(m0 - m1)))})
                ml, mn = results[(kappa, role, LEX, "calibrated_dirichlet")][0], results[(kappa, role, NLI, "calibrated_dirichlet")][0]
                if reader == NLI:
                    pairs.append({"reader": "lexical_vs_nli", "family_role": role, "kappa": kappa, "primary": "calibrated_dirichlet",
                                  "reference": "reader_version_effect", "mse_primary_minus_ref": float(np.mean((mn - truth) ** 2) - np.mean((ml - truth) ** 2)),
                                  "ci_lo": None, "ci_hi": None, "mean_abs_score_change": float(np.mean(np.abs(mn - ml)))})
    write_csv(OUT / "downstream_comparison.csv", down + [{"section": "paired"} | p for p in pairs])
    write_csv(OUT / "paired_differences.csv", pairs)
    write_csv(OUT / "calibration_uncertainty.csv", cal)

    # ---------------------------------------------------------------- 7 cold CLI replay
    replay = {}
    for reader, key in ((LEX, LEX), (NLI, NLI)):
        s = sims[(SEEDS[0], 0.5, "application", reader)]
        recs = []
        for i in range(N_AUTH):
            evs = [{"event_id": f"e{i}_{t}", "pole_score": float(look[(reader, "application", int(s["pole"][i, t]), s["status"][i, t], int(s["var"][i, t]))][0]),
                    "qualification_score": float(look[(reader, "application", int(s["pole"][i, t]), s["status"][i, t], int(s["var"][i, t]))][1])}
                   for t in range(N_EV)]
            recs.append({"author_id": f"sim_{SEEDS[0]}_{i}", "reader_id": reader, "events": evs})
        recs.append({"author_id": "no_evidence_author", "reader_id": reader, "events": []})
        inp = OUT / f"cold_input_{'lexical' if reader == LEX else 'nli'}.jsonl"
        outp = OUT / f"cold_cli_output_{'lexical' if reader == LEX else 'nli'}.jsonl"
        inp.write_text("".join(json.dumps(r) + "\n" for r in recs))
        bname = "bundle_lexical.json" if reader == LEX else "bundle_nli.json"
        subprocess.run([PY, str(ROOT / "scripts/score_grain_evidence_unit_cold.py"), "--bundle", str(OUT / bname),
                        "--sha256", bundle_sha[key], "--input", str(inp), "--output", str(outp)], check=True)
        cli = [json.loads(x) for x in outp.read_text().splitlines()]
        warm = results[(0.5, "application", reader, "calibrated_dirichlet")][0][:N_AUTH]
        replay[reader] = {"max_abs_diff_vs_warm": float(np.max(np.abs(np.array([c["pi_mean"] for c in cli[:N_AUTH]]) - warm))),
                          "no_evidence_author": cli[-1]}
    (OUT / "cold_replay.json").write_text(json.dumps(replay, indent=1))

    # ---------------------------------------------------------------- 8 frozen acceptance
    dfd = pd.DataFrame(down)
    dfc = pd.DataFrame(contr)
    dfp = pd.DataFrame(pairs)
    acc = {}
    for reader in (LEX, NLI):
        ap = dfc[(dfc.reader == reader) & (dfc.role == "application")].set_index("statistic")
        p2 = bool(ap.loc["qualified_pole_contrast", "ci_lo"] > 0 and ap.loc["qualification_direction", "ci_lo"] > 0)
        per_k = {}
        for kappa in KAPPAS:
            row = dfd[(dfd.reader == reader) & (dfd.arm == "calibrated_dirichlet") & (dfd.family_role == "application") & (dfd.kappa == kappa)].iloc[0]
            pt = dfd[(dfd.reader == reader) & (dfd.arm == "calibrated_point") & (dfd.family_role == "application") & (dfd.kappa == kappa)].iloc[0]
            pn = dfp[(dfp.reader == reader) & (dfp.family_role == "application") & (dfp.kappa == kappa) & (dfp.reference == "naive_uncalibrated")].iloc[0]
            per_k[str(kappa)] = {"P1_nonconstant": bool(row.sd_score >= 0.056), "P3_propagated": bool(row.mean_width > pt.mean_width),
                                 "P4_noninferior_to_naive": bool(pn.ci_lo <= 0),
                                 "transport_coverage_ok": bool(row.coverage >= 0.932), "transport_bias_ci_covers_0": bool(row.bias_lo <= 0 <= row.bias_hi),
                                 "sd": row.sd_score, "coverage": row.coverage, "bias": row.bias, "mse": row.mse}
        p5 = replay[reader]["max_abs_diff_vs_warm"] <= 1e-12
        positive = p2 and p5 and all(v["P1_nonconstant"] and v["P3_propagated"] and v["P4_noninferior_to_naive"] for v in per_k.values())
        transport = all(v["transport_coverage_ok"] and v["transport_bias_ci_covers_0"] for v in per_k.values())
        acc[reader] = {"P2_held_contrasts": p2, "P5_cold_replay": p5, "per_kappa": per_k,
                       "positive_construction": positive, "transport_claim": transport}
    acc["positive_overall"] = any(acc[r]["positive_construction"] for r in (LEX, NLI))
    (OUT / "acceptance.json").write_text(json.dumps(acc, indent=1, default=float))
    run = {"stage1_wall_s": stage1["wall_s"], "total_wall_s": time.time() - t0, "total_cpu_s": time.process_time() - c0,
           "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
           "code_sha256": {p: sha(ROOT / p) for p in ("grain_research/construction/evidence_unit.py", "scripts/run_grain_evidence_unit.py",
                                                       "scripts/score_grain_evidence_unit_cold.py", "grain_research/gptpro_foundation_v3/f08.py")},
           "simulated_authors": len(SEEDS) * N_AUTH, "materials": 96}
    (OUT / "run_log.json").write_text(json.dumps(run, indent=1))
    print(json.dumps({"acceptance": {r: {k: acc[r][k] for k in ("P2_held_contrasts", "P5_cold_replay", "positive_construction", "transport_claim")} for r in (LEX, NLI)},
                      "positive_overall": acc["positive_overall"], "cpu_s": run["total_cpu_s"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
