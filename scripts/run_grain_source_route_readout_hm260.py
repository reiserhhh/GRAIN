#!/usr/bin/env python3
"""One CPU-only SOURCE/relative-readout discriminator on saved full-text states."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_continuation_routes as C
from grain_research.construction import math_source_route_readout as F
from scripts import run_grain_continuation_routes_hm259 as P
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts import run_grain_source_meaning_bridge_hm233 as S

CONFIG = "configs/grain_source_route_readout_hm260.json"
DEFAULT = ROOT / "results/grain_math_response_v1/source_route_readout_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_SOURCE_ROUTE_READOUT_PLAN_HM260.md",
    "grain_research/construction/math_source_route_readout.py",
    "scripts/run_grain_source_route_readout_hm260.py",
    "tests/test_grain_source_route_readout.py",
)


def prepare(out):
    """Bind sealed actual observations and predeclared references without scoring."""
    if out.exists():
        raise FileExistsError("Do not replace completed or partial observation work")
    cfg = R.read(ROOT / CONFIG)
    base, old = ROOT / cfg["base"], ROOT / cfg["analysis"]
    manifest = R.check(
        old, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    cases = R.read(base / "cases.json")
    if len(cases) != 48 or cfg["references"] != ["occurrence", "unique"]:
        raise ValueError("Finite scope or reference contract changed")
    if any(c["unit_id"] != i for i, c in enumerate(cases)):
        raise ValueError("Actual evidence order changed")
    out.mkdir(parents=True, exist_ok=False)
    inputs = dict(manifest["inputs"])
    for name in (
        "manifest.json",
        "analyzed_seal.json",
        "verification_seal.json",
        "actions.npz",
        "contacts.json",
    ):
        path = old / name
        inputs[str(path.relative_to(ROOT))] = R.digest(path)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {**manifest["code"], **{p: R.digest(ROOT / p) for p in SCIENCE}},
            "inputs": inputs,
            "post_HM259_outcome_followup": True,
            "no_neural_model_rawtext_label_reads": True,
            "original_scope_complete": False,
        },
    )
    R.write(out / "cases.json", cases)
    R.archive(
        out / "source_observations.npz",
        **S.selected(
            base / "source_observations.npz", ("fields", "choice", "coordinates")
        ),
    )
    R.seal(out, "prepared_seal.json", ("cases.json", "source_observations.npz"))
    H.status(
        out,
        {
            "state": "PREPARED",
            "done_units": 0,
            "cpu_wall_seconds": 0,
            "new_model_calls": 0,
        },
    )
    print("PREPARED_SOURCE_READOUT", flush=True)


def family(case, route, base, parent):
    """Read ragged native/new blocks with full address and shape validation."""
    by_key = {k: [] for k in ("states", "chosen", "partition")}
    unit, length = case["unit_id"], len(case["target_ids"])
    for mode in range(2):
        chunks = {k: [] for k in by_key}
        for local, start in enumerate(range(0, 11, 4)):
            path = (
                parent / f"captures/{6 * unit + 3 * mode + local:04d}.npz"
                if route == "native"
                else base / P.capture_name(unit, mode, route, start)
            )
            with np.load(path, allow_pickle=False) as z:
                if (int(z["unit"]), int(z["mode"]), int(z["start"])) != (
                    unit,
                    mode,
                    start,
                ):
                    raise ValueError("Captured author/context/query address changed")
                if route != "native" and str(z["route"]) != route:
                    raise ValueError("Captured computation route changed")
                for key, blocks in chunks.items():
                    value = z[key].astype(float)
                    shape = (min(4, 11 - start), length)
                    if key == "states":
                        shape += (1536,)
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError("Missing/nonfinite or partial observation")
                    blocks.append(value)
        for key, context_blocks in by_key.items():
            context_blocks.append(np.concatenate(chunks[key]))
    return {key: np.stack(blocks) for key, blocks in by_key.items()}


def run(out):
    """Compile every signed readout and its SOURCE relation once, no fit or selection."""
    manifest = R.check(out)
    cfg = manifest["config"]
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No automatic restart of a partial or terminal run")
    base = ROOT / cfg["base"]
    parent = ROOT / R.read(base / "manifest.json")["config"]["parent"]
    cases = R.read(out / "cases.json")
    start_time = time.monotonic()
    done, compiled, token_rows, field_rows = 0, {}, [], []
    errors = {"total_recovery": 0.0, "reference_reallocation": 0.0, "head": 0.0}

    def mark(state, error=None):
        """Keep a resumable count without altering scientific observations."""
        H.status(
            out,
            {
                "state": state,
                "done_units": done,
                "cpu_wall_seconds": time.monotonic() - start_time,
                "new_model_calls": 0,
                "error": error,
            },
        )

    try:
        mark("RUNNING")
        for case in cases:
            if time.monotonic() - start_time >= cfg["max_cpu_seconds"]:
                raise TimeoutError("One CPU budget exhausted; no automatic retry")
            basis = S.selected(
                parent / f"captures/basis_{case['unit_id']:02d}.npz",
                ("covectors", "bias"),
            )
            refs = F.reference_covectors(
                case["target_ids"], basis["covectors"], basis["bias"]
            )
            per_reference = {
                r: {k: {} for k in ("shape", "calibration", "total")} for r in refs
            }
            delta_states = {}
            for route in ("closed", "direct", "relay", "native"):
                value = family(case, route, base, parent)
                h = value["states"]
                replay = (
                    np.einsum(
                        "kbtd,td->kbt",
                        h,
                        basis["covectors"].astype(float),
                        optimize=False,
                    )
                    + basis["bias"]
                )
                errors["head"] = max(
                    errors["head"], float(np.max(abs(replay - value["chosen"])))
                )
                delta_states[route] = F.state_difference(h)
                for ref, (wmean, bmean) in refs.items():
                    reference = (
                        np.einsum("kbtd,d->kbt", h, wmean, optimize=False) + bmean
                    )
                    pieces = F.relative_parts(
                        value["chosen"], value["partition"], reference
                    )
                    for key in pieces:
                        per_reference[ref][key][route] = pieces[key]
                del value, h
            relative_delta = {
                "native": delta_states["native"],
                "interaction": delta_states["native"]
                - delta_states["relay"]
                - delta_states["direct"]
                + delta_states["closed"],
            }
            for term, delta in relative_delta.items():
                widths = F.support_oscillation(delta, basis["covectors"])
                for q, target in enumerate(cfg["targets"]):
                    for t, width in enumerate(widths[q]):
                        field_rows.append(
                            {
                                "unit": case["unit_id"],
                                "term": term,
                                "target": target,
                                "position": t,
                                "observed_support_oscillation": float(width),
                            }
                        )
            terms = {
                ref: {key: C.decompose(routes) for key, routes in parts.items()}
                for ref, parts in per_reference.items()
            }
            for ref, parts in terms.items():
                for term in parts["total"]:
                    shape, cal, total = [
                        parts[key][term] for key in ("shape", "calibration", "total")
                    ]
                    errors["total_recovery"] = max(
                        errors["total_recovery"],
                        float(np.max(abs(shape + cal - total))),
                    )
                    shift = shape - terms["occurrence"]["shape"][term]
                    opposite = cal - terms["occurrence"]["calibration"][term]
                    errors["reference_reallocation"] = max(
                        errors["reference_reallocation"],
                        float(np.max(abs(shift + opposite))),
                    )
                    for key, array in (
                        ("shape", shape),
                        ("calibration", cal),
                        ("total", total),
                    ):
                        compiled.setdefault(f"{ref}/{key}/{term}", []).append(
                            array.sum(-1)
                        )
                    for q, target in enumerate(cfg["targets"]):
                        for t in range(shape.shape[-1]):
                            token_rows.append(
                                {
                                    "unit": case["unit_id"],
                                    "reference": ref,
                                    "term": term,
                                    "target": target,
                                    "position": t,
                                    "shape": float(shape[q, t]),
                                    "calibration": float(cal[q, t]),
                                    "total": float(total[q, t]),
                                }
                            )
            done += 1
            mark("RUNNING")
        arrays = {k: np.stack(v) for k, v in compiled.items()}
        old = S.selected(
            ROOT / cfg["analysis"] / "actions.npz",
            tuple(
                C.decompose(
                    {k: np.zeros((1,)) for k in ("closed", "direct", "relay", "native")}
                )
            ),
        )
        errors["old_total"] = max(
            float(np.max(abs(v - old[k.split("/")[-1]])))
            for k, v in arrays.items()
            if "/total/" in k
        )
        if (
            max(v for k, v in errors.items() if k != "head") > 1e-9
            or errors["head"] > 1e-3
        ):
            raise ArithmeticError(
                "Declared signed account or inherited head check failed"
            )
        R.archive(out / "readouts.npz", **arrays)
        R.table(out / "token_accounts.csv", token_rows)
        R.table(out / "relative_field.csv", field_rows)
        source = S.selected(
            out / "source_observations.npz", ("fields", "choice", "coordinates")
        )
        lengths = np.asarray([len(c["target_ids"]) for c in cases])
        contacts, intervals, rows = {}, {}, []
        for name, action in arrays.items():
            for scale, y in (("sum", action), ("per_token", action / lengths[:, None])):
                for part, x in (
                    ("full", source["fields"]),
                    ("choice", source["choice"]),
                    ("within", source["fields"] - source["choice"]),
                    ("GRAIN15", source["coordinates"]),
                ):
                    xx, yy = x.reshape(2, 24, -1), y.reshape(2, 24, 5)
                    for direction in ("pooled", 0, 1):
                        a, b = (
                            (xx.mean(0), yy.mean(0))
                            if direction == "pooled"
                            else (xx[direction], yy[direction])
                        )
                        contacts[f"{name}/{part}/{scale}/{direction}"] = C.contacts(
                            a, b
                        ).tolist()
            diagonal = np.diag(contacts[f"{name}/full/sum/pooled"])
            for q, target in enumerate(cfg["targets"]):
                rows.append(
                    {
                        "account": name,
                        "target": target,
                        "contact": float(diagonal[q]),
                        "forward": contacts[f"{name}/full/sum/0"][q][q],
                        "reverse": contacts[f"{name}/full/sum/1"][q][q],
                    }
                )
            if name.endswith("/interaction"):
                intervals[name] = C.diagonal_intervals(
                    source["fields"].reshape(2, 24, 5).mean(0),
                    action.reshape(2, 24, 5).mean(0),
                    cfg["bootstrap_seed"],
                    cfg["bootstrap_draws"],
                ).tolist()
        R.write(out / "contacts.json", contacts)
        R.write(
            out / "intervals.json",
            {"primary_J": intervals, "marginal_conditional_exploratory": True},
        )
        R.table(out / "source_contacts.csv", rows)
        summary = {
            "authors": 24,
            "directional_uses": done,
            "target_order": cfg["targets"],
            "primary_J_contacts": {
                k: np.diag(contacts[f"{k}/full/sum/pooled"]).tolist() for k in intervals
            },
            "primary_J_intervals": intervals,
            "errors": errors,
            "relative_field": {
                term: {
                    "positions": sum(r["term"] == term for r in field_rows),
                    "mean_oscillation": float(
                        np.mean(
                            [
                                r["observed_support_oscillation"]
                                for r in field_rows
                                if r["term"] == term
                            ]
                        )
                    ),
                    "max_oscillation": max(
                        r["observed_support_oscillation"]
                        for r in field_rows
                        if r["term"] == term
                    ),
                }
                for term in ("interaction", "native")
            },
            "neural_model_API_private_training": 0,
            "cpu_wall_seconds": time.monotonic() - start_time,
            "reference_is_observed_token_support_not_psychological_population": True,
            "not_new_math_personality_accuracy_or_unique_causal_split": True,
            "original_scope_complete": False,
        }
        if summary["cpu_wall_seconds"] > cfg["max_cpu_seconds"]:
            raise TimeoutError("One CPU budget exceeded; preserve partial outputs")
        R.write(out / "summary.json", summary)
        R.seal(
            out,
            "analyzed_seal.json",
            (
                "readouts.npz",
                "token_accounts.csv",
                "relative_field.csv",
                "contacts.json",
                "intervals.json",
                "source_contacts.csv",
                "summary.json",
            ),
        )
        mark("COMPLETE")
        print(__import__("json").dumps(summary), flush=True)
    except Exception as exc:
        mark("FAILED", f"{type(exc).__name__}: {exc}")
        raise


def verify(out):
    """Re-sum signed SOURCE contacts by independent scalar loops, without models."""
    R.check(out, ("analyzed_seal.json",))
    arrays = dict(np.load(out / "readouts.npz", allow_pickle=False))
    source = S.selected(
        out / "source_observations.npz", ("fields", "choice", "coordinates")
    )
    contacts = R.read(out / "contacts.json")
    errors, count = [], 0
    for name, action in arrays.items():
        y = action.reshape(2, 24, 5).mean(0)
        x = source["fields"].reshape(2, 24, 5).mean(0)
        for p in range(5):
            for q in range(5):
                xm, ym = (
                    sum(float(v) for v in x[:, p]) / 24,
                    sum(float(v) for v in y[:, q]) / 24,
                )
                value = (
                    sum(
                        (float(x[i, p]) - xm) * (float(y[i, q]) - ym) for i in range(24)
                    )
                    / 24
                )
                errors.append(abs(value - contacts[f"{name}/full/sum/pooled"][p][q]))
                count += 1
    recovery = max(
        float(
            np.max(
                abs(
                    arrays[f"{ref}/shape/{term}"]
                    + arrays[f"{ref}/calibration/{term}"]
                    - arrays[f"{ref}/total/{term}"]
                )
            )
        )
        for ref in ("occurrence", "unique")
        for term in (
            "closed",
            "direct_increment",
            "relay_increment",
            "interaction",
            "native",
            "relay_with_direct",
            "direct_with_relay",
        )
    )
    if max(errors) > 1e-10 or recovery > 1e-9:
        raise ArithmeticError("Alternate signed account check failed")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "pooled_full_contacts": count,
            "contact_error": max(errors),
            "recovery_error": recovery,
            "same_agent_not_blind": True,
            "new_model_calls": 0,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED_SOURCE_READOUT", flush=True)


def main():
    """Run individually sealed stages; no collect, resume or accuracy options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, ("analyzed_seal.json", "verification_seal.json"))
        print("ALL_SEALS_PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
