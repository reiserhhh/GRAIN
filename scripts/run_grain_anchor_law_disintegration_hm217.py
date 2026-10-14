#!/usr/bin/env python3
"""One common-catalogue anchor-law construction and finite mechanism audit."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_anchor_law_disintegration as M
from grain_research.construction import math_source_anchor_exchange as A
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_source_reference_action_hm215 import select

CONFIG = "configs/grain_anchor_law_disintegration_hm217.json"
DEFAULT = ROOT / "results/grain_math_response_v1/anchor_law_disintegration_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_ANCHOR_LAW_DISINTEGRATION_PLAN_HM217.md",
    "grain_research/construction/math_anchor_law_disintegration.py",
    "scripts/run_grain_anchor_law_disintegration_hm217.py",
    "scripts/score_grain_anchor_law_cold.py",
    "tests/test_grain_anchor_law_disintegration.py",
    "grain_research/construction/math_diagnostic_action_bridge.py",
    "grain_research/construction/math_source_anchor_exchange.py",
    "grain_research/construction/math_source_catalogue_response.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_source_reference_action_hm215.py",
)


def prepare(out):
    """Authenticate common source, fixed edits and inherited assignments once."""
    cfg = R.read(ROOT / CONFIG)
    cp, ap, np_ = (
        ROOT / cfg[k] for k in ("catalogue_parent", "anchor_parent", "native_parent")
    )
    for path in (cp, ap, np_):
        R.check(path, ("computed_seal.json", "verification_seal.json"))
    z = select(
        cp / "observations.npz",
        ("source_means", "source_indices", "codes", "variance", "points", "kernels"),
    )
    a = select(
        ap / "observations.npz",
        ("source_means", "source_indices", "labels", "panel_indices", "vector_actions"),
    )
    n = select(
        np_ / "observations.npz",
        (
            "source_means",
            "source_indices",
            "panel_indices",
            "vector_actions",
            "response",
        ),
    )
    assignments = select(ap / "analysis.npz", ("indices",))["indices"]
    if any(
        not np.array_equal(z[k], a[k]) or not np.array_equal(z[k], n[k])
        for k in ("source_means", "source_indices")
    ):
        raise ValueError("Different SOURCE catalogue")
    if not np.array_equal(z["codes"], a["labels"] @ 2 ** np.arange(5)) or set(
        z["source_indices"]
    ) & set(a["panel_indices"]):
        raise ValueError("Changed target anchors or SOURCE/panel overlap")
    if not np.array_equal(a["panel_indices"], n["panel_indices"]) or not np.array_equal(
        a["vector_actions"][:, :, 0], n["vector_actions"]
    ):
        raise ValueError("Different SAME-event panel operations")
    dx = np.stack(
        [z["points"][:, v, 0, v] - z["points"][:, v, 1, v] for v in (0, 1)], 1
    )
    error = float(np.max(abs(dx - n["vector_actions"])))
    if error > cfg["comparison_tolerance"] or assignments.shape != (5, 199, 386):
        raise ValueError("Changed fixed edit or assignment family")
    for q in range(5):
        A.validate_assignments(a["labels"], q, assignments[q])
    response = n["response"][:, :, cfg["reader_modes"], 2, :, 2]
    example = (
        ROOT
        / "results/grain_math_response_v1/diagnostic_action_bridge_r1/cold_example.json"
    )
    inputs = [
        p / f
        for p in (cp, ap, np_)
        for f in ("observations.npz", "manifest.json", "verification_seal.json")
    ]
    inputs += [ap / "analysis.npz", example]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        **z,
        source_labels=a["labels"],
        panel_indices=a["panel_indices"],
        assignments=assignments,
        native_response=response,
    )
    R.write(out / "cold_example.json", R.read(example))
    R.write(
        out / "input_quality.json",
        {
            "source_authors": 386,
            "panel_authors": 24,
            "dimensions": 1024,
            "same_event_error": error,
            "new_neural_API_private_train_fit_draws": 0,
            "historical_source_labels_only": True,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_example.json", "input_quality.json"),
    )
    print("PREPARED HM217", error, flush=True)


def contacts(actions, response):
    """Cross all reader targets at equal-author weight; keep repeated views paired."""
    x, y = np.asarray(actions, float), np.asarray(response, float)
    xx, yy = x - x.mean(1, keepdims=True), y - y.mean(0, keepdims=True)
    per_view = np.einsum("pavlc,avrt->plcrvt", xx, yy) / (len(y) - 1)
    mean = np.einsum("palc,art->plcrt", xx.mean(2), yy.mean(1)) / (len(y) - 1)
    return np.concatenate((per_view, mean[..., None, :]), 4)


def compute(out):
    """Evaluate the declared law and existing label orbit without model selection."""
    cfg, z = R.check(out)["config"], R.prepared(out)
    start = time.monotonic()
    observed = M.accounts(z["kernels"], z["codes"])
    cc = np.stack(
        [contacts(observed[..., q], z["native_response"]) for q in range(5)], -2
    )
    refs = np.empty((5, 199, 2, 3, 3, 2, 3, 5))
    samples = np.empty((5, 199, 2, 24, 2, 3, 3))
    pp = M.priors(z["codes"])
    for q in range(5):
        base = M.orbit_density(z["kernels"], z["codes"], q)
        for j, ix in enumerate(z["assignments"][q]):
            law = M.densities(z["kernels"], z["codes"][ix])
            for p in range(2):
                full, association = M.action(law, pp[p], q), M.action(base, pp[p], q)
                samples[q, j, p] = np.stack((full, association, full - association), -1)
            refs[q, j] = contacts(samples[q, j], z["native_response"])
            if time.monotonic() - start > cfg["max_compute_seconds"]:
                R.write(
                    out / "compute_failure.json",
                    {"finite_cap": True, "no_retry": True, "target": q, "reference": j},
                )
                raise RuntimeError("Finite cap; preserve partial state")
        print(f"HM217 target{q}: 199 inherited assignments evaluated", flush=True)
    primary, rows = [], []
    for p, law, part, reader, view, q, target in np.ndindex(cc.shape):
        value = float(cc[p, law, part, reader, view, q, target])
        row = {
            "reference": cfg["references"][p],
            "law": cfg["laws"][law],
            "part": cfg["parts"][part],
            "reader": cfg["reader_names"][reader],
            "view": ("view0", "view1", "mean")[view],
            "source_target": cfg["targets"][q],
            "reader_target": cfg["targets"][target],
            "covariance": value,
        } | M.rank(value, refs[q, :, p, law, part, reader, view, target])
        rows.append(row)
        if (p, law, part, reader, view) == (0, 0, 2, 1, 2) and q == target:
            primary.append(row.copy())
    adjusted = A.holm(np.array([r["two_sided_rank"] for r in primary]))
    for row, value in zip(primary, adjusted, strict=True):
        row["holm5_rank"] = float(value)
    summary = {
        "packet": "HM217",
        "primary": primary,
        "full_association_departure_action_RMS": np.sqrt(
            np.mean(observed**2, axis=(1, 2))
        ).tolist(),
        "additive_action_error": float(
            np.max(abs(observed[..., 0, :] - observed[..., 1, :] - observed[..., 2, :]))
        ),
        "new_capability": "common_SOURCE_positive_anchor_assignment_vs_joint_profile_association_cold_action_accounts",
        "new_model_API_training_fit_draws": 0,
        "reused_assignments": 995,
        "historical_conditional_diagnostic_not_population_p_or_accuracy": True,
        "original_scope_complete": False,
        "elapsed_seconds": time.monotonic() - start,
    }
    R.archive(
        out / "analysis.npz",
        actions=observed,
        contacts=cc,
        reference_actions=samples,
        reference_contacts=refs,
    )
    R.table(out / "contact_accounts.csv", rows)
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        ("analysis.npz", "contact_accounts.csv", "summary.json"),
    )
    print(json.dumps(summary, allow_nan=False), flush=True)


def explicit_orbit(k, g, q):
    """Alternate pair-inclusion expectation, without a covariance formula."""
    out = np.empty((len(k), 2, 3, 2, 32))
    for c in np.unique(g & ~(1 << q)):
        ix = np.flatnonzero((g & ~(1 << q)) == c)
        a, size = k[..., ix], len(ix)
        for code in (int(c), int(c | (1 << q))):
            n = int(np.sum(g == code))
            marginal = a.sum(-1) / size
            joint = np.einsum("...i,...i->...", a[..., 0, :], a[..., 1, :]) / size
            include = np.full((size, size), n * (n - 1) / (size * (size - 1)))
            np.fill_diagonal(include, n / size)
            product = (
                np.einsum("...i,ij,...j->...", a[..., 0, :], include, a[..., 1, :])
                / n**2
            )
            single = np.stack((marginal[:, 0, :, 0], marginal[:, 1, :, 1]), 1)
            out[..., code] = np.stack((single, joint, product), 2)
    return out


def literal(k, g, p, q):
    """Independent sum-ratio score on all observed same-event kernels."""
    values = np.empty((len(k), 2, 3, 2))
    for u, v, arm in np.ndindex(len(k), 2, 2):
        weights = np.zeros((3, 2))
        for code in range(32):
            a = k[u, v, arm, :, g == code].T
            # Boolean advanced indexing may put the author axis first.
            if a.shape[0] != 2:
                raise ValueError("Literal paired author axis")
            aa, bb = a
            component = (
                sum(a[v]) / len(aa),
                sum(float(x * y) for x, y in zip(aa, bb, strict=True)) / len(aa),
                sum(float(x * y) for x in aa for y in bb) / len(aa) ** 2,
            )
            weights[:, (code >> q) & 1] += p[code] * np.asarray(component)
        values[u, v, :, arm] = np.log(weights[:, 1] / weights[:, 0])
    return values[..., 0] - values[..., 1]


def verify(out):
    """Replay alternate arithmetic, source-event identity and sealed cold scorer."""
    R.check(out, ("computed_seal.json",))
    z, a = (
        R.prepared(out),
        select(
            out / "analysis.npz",
            ("actions", "reference_actions", "contacts", "reference_contacts"),
        ),
    )
    start, errors, checked = time.monotonic(), [], 0
    k = np.empty_like(z["kernels"])
    for v in (0, 1):
        distance = np.sum(
            (z["points"][..., v, None, :] - z["source_means"][:, v]) ** 2, -1
        )
        k[..., v, :] = np.exp(-distance / (2 * float(z["variance"])))
    errors.append(float(np.max(abs(k - z["kernels"]))))
    pp = M.priors(z["codes"])
    for q in range(5):
        baseline = explicit_orbit(k, z["codes"], q)
        errors.append(
            float(np.max(abs(baseline - M.orbit_density(z["kernels"], z["codes"], q))))
        )
        for j in (-1, 0, 198):
            g = z["codes"] if j == -1 else z["codes"][z["assignments"][q, j]]
            for p in range(2):
                full, association = (
                    literal(k, g, pp[p], q),
                    M.action(baseline, pp[p], q),
                )
                account = np.stack((full, association, full - association), -1)
                expected = (
                    a["actions"][p, ..., q]
                    if j == -1
                    else a["reference_actions"][q, j, p]
                )
                errors.append(float(np.max(abs(account - expected))))
                checked += int(account.size)
                for reader, view in np.ndindex(2, 3):
                    xx = account[:, view] if view < 2 else account.mean(1)
                    yy = (
                        z["native_response"][:, view, reader]
                        if view < 2
                        else z["native_response"][:, :, reader].mean(1)
                    )
                    cov = (xx - xx.mean(0)).reshape(24, -1).T @ (yy - yy.mean(0)) / 23
                    cov = cov.reshape(3, 3, 5)
                    expected = (
                        a["contacts"][p, :, :, reader, view, q]
                        if j == -1
                        else a["reference_contacts"][q, j, p, :, :, reader, view]
                    )
                    errors.append(float(np.max(abs(cov - expected))))
                    checked += int(cov.size)
    if max(errors) > 2e-11:
        R.write(
            out / "verification_failure.json",
            {"max_error": max(errors), "preserve_no_retry": True},
        )
        raise ValueError("Alternate arithmetic discrepancy")
    result = {
        "max_error": max(errors),
        "literal_account_contact_values": checked,
        "kernel_values": int(k.size),
        "replayed_assignments_per_target": ["observed", "first", "last"],
        "elapsed_seconds": time.monotonic() - start,
        "same_agent_alternate_arithmetic_not_blind": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result), flush=True)


def cold(out):
    """Actually run a new process with SOURCE bank and vector-only event input."""
    R.check(out, ("computed_seal.json", "verification_seal.json"))
    cmd = [
        sys.executable,
        str(ROOT / "scripts/score_grain_anchor_law_cold.py"),
        "--bank",
        str(out / "observations.npz"),
        "--record",
        str(out / "cold_example.json"),
    ]
    run = subprocess.run(
        cmd, cwd=ROOT, text=True, capture_output=True, timeout=60, check=False
    )
    if run.returncode:
        R.write(
            out / "cold_failure.json",
            {"returncode": run.returncode, "stderr": run.stderr, "no_retry": True},
        )
        raise RuntimeError("Cold process failure")
    result = json.loads(run.stdout)
    expected = select(out / "analysis.npz", ("actions",))["actions"][:, 0, 0]
    error = float(np.max(abs(np.asarray(result["actions"]) - expected)))
    if error > 2e-11:
        R.write(out / "cold_failure.json", {"max_error": error, "no_retry": True})
        raise ValueError("Cold action mismatch")
    R.write(out / "cold_result.json", result | {"replay_error": error, "command": cmd})
    R.seal(out, "cold_seal.json", ("cold_result.json",))
    print("COLD HM217", error, flush=True)


def main():
    """Scientific stages create immutable outputs once; check is read-only."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "compute", "verify", "cold", "check"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage == "check":
        R.check(
            args.out, ("computed_seal.json", "verification_seal.json", "cold_seal.json")
        )
        print("ALL HM217 SEALS PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
