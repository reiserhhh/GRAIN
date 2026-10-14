#!/usr/bin/env python3
"""F11 source-refit uncertainty for MTMM trait intervals (protocol v1)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f11_mtmm import fit_mtmm, predict_traits

RUN = ROOT / "results/grain_gptpro_foundation_v3/F11/mtmm_full_r1"
SEEDS = (20260915, 20260916, 20260917)
WORLDS = ("base", "shared_event", "conditional_loading")
PANELS = {"all_two_occasions": None, "same_event_three_readers": np.arange(0, 18, 2), "one_reader_two_occasions": np.array([0, 1, 6, 7, 12, 13])}
Z = 1.959963984540054


def load_fit(path):
    """Stored fit json -> model dict with arrays."""
    d = json.loads(path.read_text())
    return {k: np.asarray(v) if isinstance(v, list) else v for k, v in d.items()}


def main():
    """Write author_coverage.csv, checks.json, bootstrap_fits.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boot", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stored_metrics = {(r["seed"], r["world"], r["method"], r["role"], r["panel"]): r for r in csv.DictReader((RUN / "metrics.csv").open())}
    checks = {"replay_max_abs_diff": 0.0, "plugin_coverage_vs_metrics_max_diff": 0.0}
    rows, fits_log, hashes = [], [], {}
    for seed in SEEDS:
        for wi, world in enumerate(WORLDS):
            conditional, event = world == "conditional_loading", world != "base"
            source = np.load(RUN / f"data_{seed}_{world}_source.npz")["observed"]
            app = np.load(RUN / f"data_{seed}_{world}_application.npz")["observed"]
            truth = np.load(RUN / f"truth_{seed}_{world}_application.npz")["trait"]
            for p in (f"data_{seed}_{world}_source.npz", f"data_{seed}_{world}_application.npz", f"fit_{seed}_{world}_reml.json", f"fit_{seed}_{world}_oracle.json"):
                hashes[p] = hashlib.sha256((RUN / p).read_bytes()).hexdigest()
            reml = load_fit(RUN / f"fit_{seed}_{world}_reml.json")
            oracle = load_fit(RUN / f"fit_{seed}_{world}_oracle.json")
            boots = []
            for b in range(args.boot):
                idx = np.random.default_rng(np.random.SeedSequence([seed, 11, 40, wi, b])).integers(0, len(source), len(source))
                try:
                    fit = fit_mtmm(source[idx], conditional, event)
                except ValueError as exc:
                    fits_log.append({"seed": seed, "world": world, "b": b, "success": False, "method": "", "event": "", "error": "", "gradient_max": "", "exception": repr(exc)[:120]})
                    continue
                fits_log.append({"seed": seed, "world": world, "b": b, "success": fit["success"], "method": fit["method"], "event": fit["event"], "error": fit["error"], "gradient_max": fit["gradient_max"], "exception": ""})
                if fit["success"]:
                    boots.append(fit)
            for panel, cells in PANELS.items():
                pred, var = predict_traits(reml, app, conditional, cells)
                stored = np.load(RUN / f"pred_{seed}_{world}_reml_application_{panel}.npz")
                checks["replay_max_abs_diff"] = max(checks["replay_max_abs_diff"], float(np.abs(pred - stored["prediction"]).max()), float(np.abs(var - stored["variance"]).max()))
                se = np.sqrt(np.diag(var))
                plug = np.abs(pred - truth) <= 1.96 * se
                checks["plugin_coverage_vs_metrics_max_diff"] = max(checks["plugin_coverage_vs_metrics_max_diff"], abs(float(plug.mean()) - float(stored_metrics[(str(seed), world, "reml", "application", panel)]["coverage"])))
                bp = np.array([predict_traits(f, app, conditional, cells)[0] for f in boots])
                bv = np.array([np.diag(predict_traits(f, app, conditional, cells)[1]) for f in boots])
                source_var = bp.var(axis=0)
                aug_se = np.sqrt(se**2 + source_var)
                aug = np.abs(pred - truth) <= Z * aug_se
                mix_mean = bp.mean(axis=0)
                mix_se = np.sqrt(bv.mean(axis=0) + source_var)
                mix = np.abs(mix_mean - truth) <= Z * mix_se
                opred, ovar = predict_traits(oracle, app, conditional, cells)
                orc = np.abs(opred - truth) <= 1.96 * np.sqrt(np.diag(ovar))
                for a in range(len(app)):
                    rows.append({"seed": seed, "world": world, "panel": panel, "author": a,
                                 "plugin": float(plug[a].mean()), "augmented": float(aug[a].mean()), "mixture": float(mix[a].mean()), "oracle": float(orc[a].mean()),
                                 "plugin_width": float(2 * 1.96 * se.mean()), "augmented_width": float(2 * Z * aug_se[a].mean()), "mixture_width": float(2 * Z * mix_se[a].mean()),
                                 "source_var_share": float((source_var[a] / (se**2 + source_var[a])).mean())})
            fail = args.boot - len(boots)
            checks.setdefault("failures_by_cell", {})[f"{seed}_{world}"] = fail
            checks.setdefault("unreliable_cells", [])
            if fail > 0.1 * args.boot:
                checks["unreliable_cells"].append(f"{seed}_{world}")
            print(json.dumps({"seed": seed, "world": world, "boot_ok": len(boots)}), flush=True)
    for name, data in (("author_coverage.csv", rows), ("bootstrap_fits.csv", fits_log)):
        with (args.output / name).open("x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    checks["bootstrap_failures"] = sum(not r["success"] for r in fits_log)
    (args.output / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    (args.output / "input_manifest.json").write_text(json.dumps(hashes | {"code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
