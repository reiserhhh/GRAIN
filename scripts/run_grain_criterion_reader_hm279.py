#!/usr/bin/env python3
"""One immutable same-evidence criterion/observer experiment, no precision grid."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_pair_criterion as A
from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_criterion_reader as M
from grain_research.construction import math_same_regime_reader as O
from grain_research.construction import math_source_anchor_exchange as E
from grain_research.construction import math_source_event_coupling as P
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_criterion_reader_hm279.json"
DEFAULT = ROOT / "results/grain_math_response_v1/criterion_reader_r1"
SPACES = ("E5", "GRAIN15")
BANKS = ("bank0", "bank1", "pooled")
PARTS = ("join", "order", "single", "raw")
TARGETS = ("E", "N", "A", "C", "O")
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CRITERION_READER_PLAN_HM279.md",
    "grain_research/construction/math_criterion_reader.py",
    "grain_research/construction/math_criterion_components.py",
    "grain_research/construction/math_author_pair_criterion.py",
    "grain_research/construction/math_same_regime_reader.py",
    "grain_research/construction/math_source_event_coupling.py",
    "grain_research/construction/math_source_anchor_exchange.py",
    "grain_research/construction/math_behavior_predictive.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_criterion_reader_hm279.py",
    "scripts/score_grain_criterion_pair_cold.py",
    "tests/test_grain_criterion_reader.py",
)


def load(path):
    """Load only an existing numerical derivative, without pickle or raw text."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prepare(out):
    """Bind same actual event addresses, source order and old observer outputs."""
    if out.exists():
        raise FileExistsError("Retain old/failed results without replacement")
    cfg = R.read(ROOT / CONFIG)
    parents = {k: ROOT / cfg[k + "_parent"] for k in ("source", "criterion", "reader")}
    for kind, path in parents.items():
        R.check(
            path,
            ("collected_seal.json", "analyzed_seal.json")
            if kind == "reader"
            else ("computed_seal.json", "verification_seal.json"),
        )
    source = load(parents["source"] / "source_pair_bank.npz")
    criterion = load(parents["criterion"] / "scores.npz")
    inference = load(parents["criterion"] / "inference.npz")
    panel = load(parents["reader"] / "numeric_pairs.npz")
    if (
        panel["points"].shape != (48, 2, 1024)
        or inference["assignments"].shape != (5, cfg["assignments"], 386)
        or not np.array_equal(source["source_indices"], criterion["source_indices"])
        or not np.array_equal(source["folds"], criterion["banks"])
        or not np.array_equal(P.BITS[source["codes"]], criterion["labels"])
        or not np.array_equal(
            panel["panel_edges"],
            np.array([22 * u + 11 * w + 5 for u in range(24) for w in (0, 1)]),
        )
    ):
        raise ValueError("Changed source or SAME48 adjacent evidence identity")
    bindings = R.read(parents["reader"] / "event_bindings.json")
    if len(bindings) != 48 or any(
        b["unit_id"] != i or b["panel_edge"] != int(panel["panel_edges"][i])
        for i, b in enumerate(bindings)
    ):
        raise ValueError("Numeric pairs and archived neural units are not aligned")
    files = {}
    for kind, names in {
        "source": (
            "manifest.json",
            "computed_seal.json",
            "verification_seal.json",
            "source_pair_bank.npz",
            "qualified_panel_pairs.npz",
        ),
        "criterion": (
            "manifest.json",
            "computed_seal.json",
            "verification_seal.json",
            "scores.npz",
            "inference.npz",
        ),
        "reader": (
            "manifest.json",
            "collected_seal.json",
            "analyzed_seal.json",
            "numeric_pairs.npz",
            "event_bindings.json",
            "observed_sums.npz",
            "source_readings.npz",
            "bank.json",
        ),
    }.items():
        files.update(
            {
                str((parents[kind] / n).relative_to(ROOT)): R.digest(parents[kind] / n)
                for n in names
            }
        )
    old_panel = load(parents["source"] / "qualified_panel_pairs.npz")
    first = bindings[0]
    record = {
        "frame_key": "HM279-same-HM274-unit0",
        "sampling_regime": "next_cached_record_within_registered_window",
        "events": [
            {
                "event_id": str(first["comment_ids"][k]),
                "vector": panel["points"][0, k].tolist(),
                "time": float(first["times"][k]),
                "window": 0,
                "condition": int(old_panel["conditions"][0, 5 + k]),
            }
            for k in (0, 1)
        ],
    }
    out.mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": files,
            "original_scope_complete": False,
        },
    )
    R.write(out / "example_pair.json", record)
    R.write(
        out / "input_quality.json",
        {
            "source_authors": 386,
            "application_authors": 24,
            "actual_pairs": 48,
            "old_criterion_assignments_reused": 995,
            "new_neural_API_private_training": 0,
            "application_labels_read": False,
            "source_bank_reference_retained": True,
            "historical_not_untouched_validation": True,
        },
    )
    R.seal(out, "prepared_seal.json", ("example_pair.json", "input_quality.json"))
    print("PREPARED HM279: same48 evidence, no new model observations", flush=True)


def expired(_signum, _frame):
    """Retain any partial output and stop at the original finite CPU budget."""
    raise TimeoutError("Original120CPU-second cap reached")


def compute(out):
    """Connect all criterion components to the identical archived behavior reading."""
    cfg = R.check(out)["config"]
    source = load(ROOT / cfg["source_parent"] / "source_pair_bank.npz")
    criterion = load(ROOT / cfg["criterion_parent"] / "scores.npz")
    reference = load(ROOT / cfg["criterion_parent"] / "inference.npz")
    reader_parent = ROOT / cfg["reader_parent"]
    panel, sums = (
        load(reader_parent / "numeric_pairs.npz"),
        load(reader_parent / "observed_sums.npz"),
    )
    old = load(reader_parent / "source_readings.npz")
    began = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(cfg["max_seconds"])
    reader = O.cold(
        {
            "version": "HM274-v1",
            "bank": R.read(reader_parent / "bank.json"),
            **{k: v.tolist() for k, v in sums.items()},
        }
    )
    behavior = np.array(
        [
            reader["accounts"][scale]["balanced_key"]["values"]
            for scale in ("sum", "per_token")
        ]
    )
    points = np.stack([panel["points"], panel["points"][:, ::-1]], 1).reshape(
        96, 2, 1024
    )
    labels, means = criterion["labels"], criterion["source_means"]
    d = np.zeros((2, 3, 48, 2, 386))
    scores = np.empty((2, 3, 48, 2, 3, 5))
    all_contacts = np.empty((2, 3, 3, 2, 3, 4, 5, 5))
    rows, old_error, reconstruction = [], 0.0, 0.0
    for sp, space in enumerate(SPACES):
        x, z = points, source["prototypes"]
        if space == "GRAIN15":
            x, z = x @ source["projection"], z @ source["projection"]
        logs = P.event_logs(x, z, float(source["variance"]))
        for b, mask in enumerate(
            (source["folds"] == 0, source["folds"] == 1, np.ones(386, bool))
        ):
            w = A.source_author_weights(logs[:, mask], source["relink"][mask])
            delta = (w[:, 0] - w[:, 1]).reshape(48, 2, -1)
            d[sp, b][:, :, mask] = delta
            value = M.pair_components(delta, labels[mask], means[mask])
            scores[sp, b] = value
            old_targets = old[f"actual/{space}/{BANKS[b]}/targets"]
            old_error = max(
                old_error,
                float(
                    np.max(
                        abs(
                            value[:, :, 0]
                            - (old_targets[:, :, 0] - old_targets[:, :, 1])
                        )
                    )
                ),
            )
            reconstruction = max(
                reconstruction,
                float(np.max(abs(value[:, :, 0] - value[:, :, 1:].sum(2)))),
            )
            for window, ix in enumerate(
                (panel["windows"] == 0, panel["windows"] == 1, np.ones(48, bool))
            ):
                for scale in range(2):
                    c = M.contacts(value[ix], behavior[scale, ix])
                    all_contacts[sp, b, window, scale] = c
                    for channel, name in enumerate(C.CHANNELS):
                        for part, part_name in enumerate(PARTS):
                            for q, target in enumerate(TARGETS):
                                for r, observed_target in enumerate(TARGETS):
                                    rows.append(
                                        {
                                            "space": space,
                                            "source_bank": BANKS[b],
                                            "window": ("0", "1", "pooled")[window],
                                            "scale": ("sum", "per_token")[scale],
                                            "channel": name,
                                            "part": part_name,
                                            "source_target": target,
                                            "reader_target": observed_target,
                                            "contact": float(c[channel, part, q, r]),
                                        }
                                    )
    alpha = np.stack([M.observer_pullback(d[sp, 2], behavior[0]) for sp in range(2)])
    obs, null, var, nominal = M.conditional_reference(
        alpha, labels, means, criterion["groups"], reference["assignments"]
    )
    expected = all_contacts[:, 2, 2, 0, 2, 0]
    pullback_error = float(np.max(abs(obs - expected)))
    if max(old_error, reconstruction, pullback_error) > 1e-12:
        raise ArithmeticError("Old observation or component meaning changed; no rescue")
    diagonal = np.diagonal(nominal, axis1=1, axis2=2)
    holm10 = np.array(E.holm(diagonal.ravel())).reshape(2, 5)
    holm50 = np.array(E.holm(nominal.ravel())).reshape(2, 5, 5)
    primary = []
    for sp, space in enumerate(SPACES):
        for q, target in enumerate(TARGETS):
            primary.append(
                {
                    "space": space,
                    "target": target,
                    "total_contact": float(all_contacts[sp, 2, 2, 0, 0, 0, q, q]),
                    "composition_contact": float(all_contacts[sp, 2, 2, 0, 1, 0, q, q]),
                    "within_contact": float(obs[sp, q, q]),
                    "conditional_p": float(nominal[sp, q, q]),
                    "Holm10_p": float(holm10[sp, q]),
                    "exact_orbit_variance": float(var[sp, q, q]),
                }
            )
    R.archive(
        out / "readings.npz",
        weights=d,
        scores=scores,
        behavior=behavior,
        contacts=all_contacts,
        windows=panel["windows"],
        labels=labels,
        means=means,
        groups=criterion["groups"],
        banks=source["folds"],
    )
    R.archive(
        out / "inference.npz",
        alpha=alpha,
        observed=obs,
        null=null,
        variance=var,
        nominal=nominal,
        holm10=holm10,
        holm50=holm50,
        assignments=reference["assignments"],
    )
    R.table(out / "all_contacts.csv", rows)
    R.table(out / "primary.csv", primary)
    cold = {}
    for sp, space in enumerate(SPACES):
        result = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    str(ROOT / "scripts/score_grain_criterion_pair_cold.py"),
                    "--bundle",
                    str(ROOT / cfg["source_parent"] / "source_pair_bank.npz"),
                    "--record",
                    str(out / "example_pair.json"),
                    "--space",
                    space,
                ],
                text=True,
            )
        )
        for b, label in enumerate(BANKS):
            np.testing.assert_allclose(
                result["source_readings"][label], scores[sp, b, 0], atol=1e-12, rtol=0
            )
        cold[space] = result
    R.write(out / "cold_readback.json", cold)
    R.write(
        out / "summary.json",
        {
            "CPU_wall_seconds": time.monotonic() - began,
            "primary": primary,
            "old_total_error": old_error,
            "component_reconstruction_error": reconstruction,
            "observer_pullback_error": pullback_error,
            "source_within_bank_window_matching": all_contacts[:, :, :, 0, 2, 0]
            .diagonal(axis1=-2, axis2=-1)
            .tolist(),
            "all_accounts": len(rows),
            "model_readings_are_not_actual_human_actions": True,
            "contact_sign_is_not_correctness": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "computed_seal.json",
        (
            "readings.npz",
            "inference.npz",
            "all_contacts.csv",
            "primary.csv",
            "cold_readback.json",
            "summary.json",
        ),
    )
    signal.alarm(0)
    print(json.dumps(R.read(out / "summary.json"), indent=2), flush=True)


def verify(out):
    """Recompute SOURCE/observer products and orbit variance by different arithmetic."""
    R.check(out, ("computed_seal.json",))
    data, inf = load(out / "readings.npz"), load(out / "inference.npz")
    errors = {
        "components": 0.0,
        "contacts": 0.0,
        "assignment_contacts": 0.0,
        "orbit_variance": 0.0,
    }
    y, m = data["labels"], data["means"]
    values = (y, m, y - m)
    for sp in range(2):
        for b in range(3):
            for component, v in enumerate(values):
                direct = np.einsum("ioj,jq->ioq", data["weights"][sp, b], v)
                errors["components"] = max(
                    errors["components"],
                    float(np.max(abs(direct - data["scores"][sp, b, :, :, component]))),
                )
            for window, ix in enumerate(
                (data["windows"] == 0, data["windows"] == 1, np.ones(48, bool))
            ):
                for scale in range(2):
                    r = data["behavior"][scale, ix]
                    re = (r[:, 0] + r[:, 1]) / 2
                    ro = (r[:, 0] - r[:, 1]) / 2
                    rs = (r[:, 2] + r[:, 3]) / 2
                    for c in range(3):
                        s = data["scores"][sp, b, ix, :, c]
                        se, so = (s[:, 0] + s[:, 1]) / 2, (s[:, 0] - s[:, 1]) / 2
                        direct = np.stack(
                            [
                                np.einsum("iq,ir->qr", se, re - rs) / len(s),
                                np.einsum("iq,ir->qr", so, ro) / len(s),
                                np.einsum("iq,ir->qr", se, rs) / len(s),
                                (
                                    np.einsum("iq,ir->qr", s[:, 0], r[:, 0])
                                    + np.einsum("iq,ir->qr", s[:, 1], r[:, 1])
                                )
                                / (2 * len(s)),
                            ]
                        )
                        errors["contacts"] = max(
                            errors["contacts"],
                            float(
                                np.max(
                                    abs(
                                        direct
                                        - data["contacts"][sp, b, window, scale, c]
                                    )
                                )
                            ),
                        )
    r = data["behavior"][0]
    rj = (r[:, 0] + r[:, 1] - r[:, 2] - r[:, 3]) / 2
    for q in range(5):
        for draw, permutation in enumerate(inf["assignments"][q]):
            edited = y.copy()
            edited[:, q] = y[permutation, q]
            codes = (edited.astype(int) * (1 << np.arange(5))).sum(1)
            for bank in (0, 1):
                mask = data["banks"] == bank
                if not np.array_equal(
                    np.bincount(codes[mask], minlength=32),
                    np.bincount(
                        (y[mask].astype(int) * (1 << np.arange(5))).sum(1), minlength=32
                    ),
                ):
                    raise ValueError("Changed full32 SOURCE profile count")
            for sp in range(2):
                readings = data["weights"][sp, 2].mean(1) @ (edited[:, q] - m[:, q])
                contact = np.einsum("i,ir->r", readings, rj) / 48
                errors["assignment_contacts"] = max(
                    errors["assignment_contacts"],
                    float(np.max(abs(contact - inf["null"][draw, sp, q]))),
                )
        for sp in range(2):
            variance = np.zeros(5)
            for cell in np.unique(data["groups"][:, q]):
                mask = data["groups"][:, q] == cell
                rows = inf["alpha"][sp, mask]
                n, p = len(rows), y[mask, q].mean()
                if n > 1:
                    variance += (
                        p * (1 - p) * n / (n - 1) * ((rows - rows.mean(0)) ** 2).sum(0)
                    )
            errors["orbit_variance"] = max(
                errors["orbit_variance"],
                float(np.max(abs(variance - inf["variance"][sp, q]))),
            )
    if max(errors.values()) > 1e-12:
        raise ArithmeticError("Alternate arithmetic failed; preserve result")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "alternate_errors": errors,
            "all_contacts": 10800,
            "conditional_assignments": 995,
            "same_agent_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(R.read(out / "verification.json"), indent=2), flush=True)


def main():
    """Run a finite immutable stage, never overwrite a scientific failure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "compute", "verify", "check"))
    p.add_argument("--output", type=Path, default=DEFAULT)
    a = p.parse_args()
    try:
        if a.mode == "check":
            R.check(a.output, ("computed_seal.json", "verification_seal.json"))
            print("CHECKED HM279", flush=True)
        else:
            globals()[a.mode](a.output)
    except BaseException as exc:
        if a.output.exists() and not (a.output / "failure.json").exists():
            R.write(a.output / "failure.json", {"stage": a.mode, "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
