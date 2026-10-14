#!/usr/bin/env python3
"""One finite same-evidence full-text route study, no accuracy selection."""

import argparse
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_continuation_routes as C
from grain_research.construction import math_late_condition as L
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_late_condition_hm258 as P
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts import run_grain_source_meaning_bridge_hm233 as S

CONFIG = "configs/grain_continuation_routes_hm259.json"
DEFAULT = ROOT / "results/grain_math_response_v1/continuation_routes_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CONTINUATION_ROUTES_PLAN_HM259.md",
    "grain_research/construction/math_continuation_routes.py",
    "scripts/run_grain_continuation_routes_hm259.py",
    "scripts/score_grain_continuation_routes_cold.py",
    "tests/test_grain_continuation_routes.py",
    "grain_research/construction/math_late_condition.py",
    "grain_research/construction/math_author_predictive_link.py",
    "scripts/run_grain_late_condition_hm258.py",
    "scripts/run_grain_conditional_likelihood_hm152.py",
    "scripts/run_grain_joint_persona_style_hm150.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "scripts/run_grain_source_meaning_bridge_hm233.py",
    "grain_research/vector_mechanisms/trace.py",
)


def prepare(out):
    """Bind exact cached public token/event addresses; do not reread raw bodies."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No replacement of partial or completed science")
    cfg = R.read(ROOT / CONFIG)
    if (
        cfg["authors"],
        cfg["directional_units"],
        cfg["hypotheses"],
        cfg["batch_size"],
    ) != (24, 48, 11, 4):
        raise ValueError("Finite registered scope changed")
    parent = ROOT / cfg["parent"]
    pm = R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    cases = R.read(parent / "cases.json")
    tok = AutoTokenizer.from_pretrained(
        J.MODEL, local_files_only=True, trust_remote_code=False
    )
    header, suffixes = L.observation_parts(tok, [])
    for i, case in enumerate(cases):
        if case["unit_id"] != i or case["suffixes"] != suffixes:
            raise ValueError("Source order or observation wording changed")
        if i >= 24 and case["author"] != cases[i - 24]["author"]:
            raise ValueError("Two uses must bind the same 24 authors")
        for mode, prefix in enumerate(case["prefixes"]):
            if prefix[: len(header)] != header or len(prefix) <= len(header):
                raise ValueError("Exact generic prefix/body boundary missing")
            arrays = S.selected(
                parent / f"prefixes/{i:02d}_{mode}.npz", ("keys", "values")
            )
            if (
                arrays["keys"].shape[-2] != len(prefix)
                or arrays["values"].shape != arrays["keys"].shape
            ):
                raise ValueError("Cached source and token address do not agree")
        case["body_start"] = len(header)
    if len(cases) != 48:
        raise ValueError("All directional units required")
    inputs = [
        parent / p
        for p in (
            "manifest.json",
            "prepared_seal.json",
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
            "cases.json",
            "source_observations.npz",
            "collection_quality.json",
        )
    ]
    inputs += sorted((parent / "prefixes").glob("*.npz"))
    inputs += sorted((parent / "captures").glob("*.npz"))
    out.mkdir(parents=True, exist_ok=False)
    (out / "captures").mkdir()
    (out / "guards").mkdir()
    source = S.selected(
        parent / "source_observations.npz", ("fields", "choice", "coordinates")
    )
    R.archive(out / "source_observations.npz", **source)
    R.write(out / "cases.json", cases)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "model_weights_sha256": pm["model_weights_sha256"],
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_scope_complete": False,
        },
    )
    quality = {
        "authors": 24,
        "directional_uses": 48,
        "new_batches": 864,
        "new_conditional_sequences": 3168,
        "reused_native_sequences": 1056,
        "new_guard_sequences": 8,
        "guard_prefix_encodes": 2,
        "body_start": len(header),
        "same_text_query_cache_SOURCE": True,
        "first_target_prediction_in_direct": True,
        "raw_application_labels_new_people_generation_API_training": 0,
        "historical_exploration_not_untouched_validation": True,
        "discovery_overlap_authors": 17,
        "exact_subreddit_matches": 0,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("cases.json", "source_observations.npz", "input_quality.json"),
    )
    H.status(
        out,
        {"state": "PREPARED", "done_batches": 0, "model_seconds": 0, "pending": None},
    )
    print(json.dumps(quality), flush=True)


def capture_name(unit, mode, route, start):
    """A scientific file address binds author-use, context, route and query group."""
    return f"captures/{unit:02d}_{mode}_{route}_{start:02d}.npz"


def collect(out):
    """Run only three new graphs with a single cumulative cap; preserve failures."""
    import torch

    manifest = R.check(out)
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No automatic retry of partial or terminal experiment")
    cfg, cases = manifest["config"], R.read(out / "cases.json")
    parent = ROOT / cfg["parent"]
    seconds, done, pending = 0.0, 0, None
    head_error, native_error, closed_error, mutation_error = 0.0, 0.0, 0.0, 0.0
    files, receipts, guards = [], [], []
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))

    def mark(state):
        """Checkpoint the actual finite operation rather than only process life."""
        H.status(
            out,
            {
                "state": state,
                "pid": os.getpid(),
                "done_batches": done,
                "model_seconds": seconds,
                "pending": pending,
            },
        )

    def timed(operation):
        """Charge failed calls as well; never renew the scientific budget."""
        nonlocal seconds
        if seconds >= cfg["max_model_seconds"]:
            raise TimeoutError("Original single model-second cap exhausted")
        start = time.monotonic()
        try:
            value = operation()
            torch.mps.synchronize()
            return value
        finally:
            seconds += time.monotonic() - start

    try:
        mark("RUNNING")
        observer = J.offline_model(manifest)
        for case in cases:
            unit, body = case["unit_id"], case["body_start"]
            with np.load(
                parent / f"captures/basis_{unit:02d}.npz", allow_pickle=False
            ) as z:
                basis = {k: z[k].copy() for k in ("covectors", "bias")}
            for mode, prefix in enumerate(case["prefixes"]):
                arrays = S.selected(
                    parent / f"prefixes/{unit:02d}_{mode}.npz", ("keys", "values")
                )
                cache = C.restore_cache(observer, arrays)
                for route in cfg["new_routes"]:
                    for start in range(0, 11, 4):
                        suffixes = case["suffixes"][start : start + 4]
                        pending = {
                            "unit": unit,
                            "context": mode,
                            "route": route,
                            "start": start,
                        }
                        mark("RUNNING")
                        result = timed(
                            partial(
                                C.observe,
                                observer,
                                cache,
                                suffixes,
                                case["target_ids"],
                                body,
                                route,
                            )
                        )
                        if any(not np.array_equal(result[k], basis[k]) for k in basis):
                            raise ValueError("Actual output-token covectors changed")
                        replay = (
                            np.sum(
                                result["states"].astype(float)
                                * basis["covectors"].astype(float),
                                -1,
                            )
                            + basis["bias"]
                        )
                        head_error = max(
                            head_error, float(np.max(abs(replay - result["chosen"])))
                        )
                        name = capture_name(unit, mode, route, start)
                        R.archive(
                            out / name,
                            **{k: result[k] for k in ("states", "chosen", "partition")},
                            unit=unit,
                            mode=mode,
                            route=route,
                            start=start,
                        )
                        files.append(name)
                        receipts.extend({**pending, **r} for r in result["receipts"])
                        done += 1
                        if unit == 0 and route == "closed" and start == 0:
                            pending = {
                                "unit": unit,
                                "context": mode,
                                "operation": "native_and_closed_guards",
                            }
                            mark("RUNNING")
                            opened = timed(
                                partial(
                                    C.observe,
                                    observer,
                                    cache,
                                    suffixes[:2],
                                    case["target_ids"],
                                    body,
                                    "native",
                                )
                            )
                            old = S.selected(
                                parent / f"captures/{3 * mode:04d}.npz",
                                ("chosen", "partition"),
                            )
                            delta = (
                                opened["chosen"].astype(float)
                                - opened["partition"].astype(float)
                            ) - (
                                old["chosen"][:2].astype(float)
                                - old["partition"][:2].astype(float)
                            )
                            ne = float(np.max(abs(delta)))
                            native_error = max(native_error, ne)
                            shifted = (
                                prefix[:body]
                                + prefix[body + 1 :]
                                + prefix[body : body + 1]
                            )
                            other_cache = timed(
                                partial(L.encode_context, observer, shifted)
                            )
                            other = timed(
                                partial(
                                    C.observe,
                                    observer,
                                    other_cache,
                                    suffixes[:2],
                                    case["target_ids"],
                                    body,
                                    "closed",
                                )
                            )
                            ce = max(
                                float(
                                    np.max(
                                        abs(
                                            other[k].astype(float)
                                            - result[k][:2].astype(float)
                                        )
                                    )
                                )
                                for k in ("chosen", "partition")
                            )
                            closed_error = max(closed_error, ce)
                            gname = f"guards/{mode}.npz"
                            R.archive(
                                out / gname,
                                native_delta=delta,
                                closed_chosen=other["chosen"],
                                closed_partition=other["partition"],
                                reference_chosen=result["chosen"][:2],
                                reference_partition=result["partition"][:2],
                            )
                            files.append(gname)
                            guards.append(
                                {
                                    "context": mode,
                                    "native_error": ne,
                                    "closed_error": ce,
                                }
                            )
                            if max(ne, ce) > cfg["native_tolerance"]:
                                raise ArithmeticError(
                                    "Declared graph/native guard failed"
                                )
                            del other_cache, other, opened
                        pending = None
                        mark("RUNNING")
                restored = L.cache_arrays(cache)
                mutation_error = max(
                    mutation_error,
                    max(float(np.max(abs(restored[k] - arrays[k]))) for k in arrays),
                )
                if mutation_error != 0:
                    raise ArithmeticError("Source prefix mutated")
                del cache, restored, arrays
            print(
                json.dumps({"unit": unit, "batches": done, "model_seconds": seconds}),
                flush=True,
            )
        quality = {
            "batches": done,
            "conditional_sequences": 3168,
            "guard_sequences": 8,
            "guard_prefix_encodes": 2,
            "model_seconds": seconds,
            "head_error": head_error,
            "native_error": native_error,
            "closed_guard_error": closed_error,
            "prefix_mutation_error": mutation_error,
            "forbidden_edge_error": max(
                max(r["closed_relay_weight"], r["closed_direct_weight"])
                for r in receipts
            ),
        }
        R.write(out / "collection_quality.json", quality)
        R.table(out / "route_receipts.csv", receipts)
        R.table(out / "guard_receipts.csv", guards)
        R.seal(
            out,
            "collected_seal.json",
            files
            + ["collection_quality.json", "route_receipts.csv", "guard_receipts.csv"],
        )
        mark("COMPLETE")
        lock.unlink()
        print(json.dumps(quality), flush=True)
    except Exception as exc:
        mark("PARTIAL_BUDGET" if isinstance(exc, TimeoutError) else "FAILED")
        R.write(
            out / "failure.json",
            {"error": repr(exc), "pending": pending, "no_automatic_retry": True},
        )
        raise


def observations(out):
    """Read all four complete graph families, including inherited native arrays."""
    cfg = R.read(out / "manifest.json")["config"]
    cases = R.read(out / "cases.json")
    _, native = P.observations(ROOT / cfg["parent"])
    values = []
    for case, old in zip(cases, native, strict=True):
        unit_values = {"native": old}
        for route in cfg["new_routes"]:
            by_key = {k: [] for k in ("chosen", "partition")}
            for mode in range(2):
                chunks = {k: [] for k in by_key}
                for start in range(0, 11, 4):
                    with np.load(
                        out / capture_name(case["unit_id"], mode, route, start),
                        allow_pickle=False,
                    ) as z:
                        if (
                            int(z["unit"]),
                            int(z["mode"]),
                            str(z["route"]),
                            int(z["start"]),
                        ) != (case["unit_id"], mode, route, start):
                            raise ValueError(
                                "Changed actual computation/evidence address"
                            )
                        for k, values in chunks.items():
                            values.append(z[k].astype(float))
                for k, values in by_key.items():
                    values.append(np.concatenate(chunks[k]))
            unit_values[route] = {k: np.stack(v) for k, v in by_key.items()}
        values.append(unit_values)
    return cases, values


def analyze(out):
    """Export all signed actions and SOURCE links, never rank routes by output."""
    manifest = R.check(out, ("collected_seal.json",))
    cfg = manifest["config"]
    cases, readings = observations(out)
    effects, numerators, normalizers, cold_units, token_rows = [], [], [], [], []
    for case, family in zip(cases, readings, strict=True):
        accounts = {
            k: L.accounts(v["chosen"], v["partition"]) for k, v in family.items()
        }
        terms = {
            k: C.decompose({r: a[k] for r, a in accounts.items()})
            for k in ("effect", "numerator", "normalizer")
        }
        effects.append({k: v.sum(-1) for k, v in terms["effect"].items()})
        numerators.append({k: v.sum(-1) for k, v in terms["numerator"].items()})
        normalizers.append({k: v.sum(-1) for k, v in terms["normalizer"].items()})
        cold_units.append(
            {
                "unit": case["unit_id"],
                "routes": {
                    r: {k: v[k].sum(-1).tolist() for k in ("chosen", "partition")}
                    for r, v in family.items()
                },
            }
        )
        for name, value in terms["effect"].items():
            for q, target in enumerate(cfg["targets"]):
                for t, number in enumerate(value[q]):
                    token_rows.append(
                        {
                            "unit": case["unit_id"],
                            "term": name,
                            "target": target,
                            "position": t,
                            "effect": float(number),
                            "numerator": float(terms["numerator"][name][q, t]),
                            "normalizer": float(terms["normalizer"][name][q, t]),
                        }
                    )
    terms = {k: np.stack([v[k] for v in effects]) for k in effects[0]}
    R.archive(
        out / "actions.npz",
        **terms,
        **{"numerator_" + k: np.stack([v[k] for v in numerators]) for k in terms},
        **{"normalizer_" + k: np.stack([v[k] for v in normalizers]) for k in terms},
    )
    R.table(out / "token_accounts.csv", token_rows)
    R.write(out / "cold_input.json", {"version": "HM259-v1", "units": cold_units})
    cold = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(ROOT / "scripts/score_grain_continuation_routes_cold.py"),
                "--input",
                str(out / "cold_input.json"),
            ],
            text=True,
        )
    )
    if cold["neural_runtime_loaded"]:
        raise ValueError("Cold account loaded a neural runtime")
    R.write(out / "cold_output.json", cold)
    source = S.selected(
        out / "source_observations.npz", ("fields", "choice", "coordinates")
    )
    contacts, intervals, rows = {}, {}, []
    lengths = np.asarray([len(c["target_ids"]) for c in cases])
    for term, action in terms.items():
        pooled = action.reshape(2, 24, 5).mean(0)
        xfull = source["fields"].reshape(2, 24, 5).mean(0)
        intervals[term] = C.diagonal_intervals(
            xfull, pooled, cfg["bootstrap_seed"], cfg["bootstrap_draws"]
        ).tolist()
        for scale, y in (("sum", action), ("per_token", action / lengths[:, None])):
            for part, x in (
                ("full", source["fields"]),
                ("choice", source["choice"]),
                ("within", source["fields"] - source["choice"]),
                ("GRAIN15", source["coordinates"]),
            ):
                for direction in ("pooled", 0, 1):
                    xx, yy = x.reshape(2, 24, -1), y.reshape(2, 24, 5)
                    xx, yy = (
                        (xx.mean(0), yy.mean(0))
                        if direction == "pooled"
                        else (xx[direction], yy[direction])
                    )
                    contacts[f"{term}/{part}/{scale}/{direction}"] = C.contacts(
                        xx, yy
                    ).tolist()
        matrix = np.asarray(contacts[f"{term}/full/sum/pooled"])
        for q, target in enumerate(cfg["targets"]):
            rows.append(
                {
                    "term": term,
                    "target": target,
                    "source_contact": float(matrix[q, q]),
                    "source_CI_low": intervals[term][q][0],
                    "source_CI_high": intervals[term][q][1],
                    "action_RMS": float(np.sqrt(np.mean(action[:, q] ** 2))),
                    "forward_contact": contacts[f"{term}/full/sum/0"][q][q],
                    "reverse_contact": contacts[f"{term}/full/sum/1"][q][q],
                }
            )
    R.write(out / "contacts.json", contacts)
    R.write(
        out / "author_intervals.json",
        {"intervals": intervals, "marginal_not_familywise": True},
    )
    R.table(out / "meaning_vs_action.csv", rows)
    reconstruction = terms["native"] - sum(
        terms[k]
        for k in ("closed", "direct_increment", "relay_increment", "interaction")
    )
    summary = {
        "authors": 24,
        "directional_uses": 48,
        "target_order": cfg["targets"],
        "diagonal_full_contacts": {
            k: np.diag(contacts[f"{k}/full/sum/pooled"]).tolist() for k in terms
        },
        "primary_interaction_intervals": intervals["interaction"],
        "signed_reconstruction_error": float(np.max(abs(reconstruction))),
        "collection": R.read(out / "collection_quality.json"),
        "not_accuracy_r_information_fraction_or_new_graph_theorem": True,
        "same_agent_not_blind": True,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "actions.npz",
            "token_accounts.csv",
            "cold_input.json",
            "cold_output.json",
            "contacts.json",
            "author_intervals.json",
            "meaning_vs_action.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summary), flush=True)


def verify(out):
    """Literal token and SOURCE sums; do not launch another model observation."""
    R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    cases, readings = observations(out)
    cfg = R.read(out / "manifest.json")["config"]
    parent = ROOT / cfg["parent"]
    saved = S.selected(
        out / "actions.npz", tuple(C.decompose({r: np.zeros(1) for r in C.ROUTES}))
    )
    cold = R.read(out / "cold_output.json")["rows"]
    action_error, cold_error, head_error, counts = 0.0, 0.0, 0.0, 0
    for case, family, cr in zip(cases, readings, cold, strict=True):
        raw = {}
        for route, value in family.items():
            ll = value["chosen"] - value["partition"]
            raw[route] = np.asarray(
                [
                    sum(
                        ll[0, 2 * q + 1, t]
                        - ll[0, 2 * q + 2, t]
                        - ll[1, 2 * q + 1, t]
                        + ll[1, 2 * q + 2, t]
                        for t in range(ll.shape[-1])
                    )
                    for q in range(5)
                ]
            )
            counts += 5
        literal = C.decompose(raw)
        for term, value in literal.items():
            action_error = max(
                action_error, float(np.max(abs(value - saved[term][case["unit_id"]])))
            )
            cold_error = max(cold_error, float(np.max(abs(value - cr["terms"][term]))))
        with np.load(
            parent / f"captures/basis_{case['unit_id']:02d}.npz", allow_pickle=False
        ) as z:
            w, bias = z["covectors"].astype(float), z["bias"].astype(float)
        for mode in range(2):
            for route in cfg["new_routes"]:
                for start in range(0, 11, 4):
                    with np.load(
                        out / capture_name(case["unit_id"], mode, route, start),
                        allow_pickle=False,
                    ) as z:
                        replay = (
                            np.einsum(
                                "btd,td->bt",
                                z["states"].astype(float),
                                w,
                                optimize=False,
                            )
                            + bias
                        )
                        head_error = max(
                            head_error, float(np.max(abs(replay - z["chosen"])))
                        )
    source = S.selected(
        out / "source_observations.npz", ("fields", "choice", "coordinates")
    )
    contacts = R.read(out / "contacts.json")
    contact_error, contact_count, split_error = 0.0, 0, 0.0
    lengths = np.asarray([len(c["target_ids"]) for c in cases])
    for term, action in saved.items():
        for scale, y in (("sum", action), ("per_token", action / lengths[:, None])):
            for part, x in (
                ("full", source["fields"]),
                ("choice", source["choice"]),
                ("within", source["fields"] - source["choice"]),
                ("GRAIN15", source["coordinates"]),
            ):
                for direction in ("pooled", 0, 1):
                    xx, yy = x.reshape(2, 24, -1), y.reshape(2, 24, 5)
                    xx, yy = (
                        (xx.mean(0), yy.mean(0))
                        if direction == "pooled"
                        else (xx[direction], yy[direction])
                    )
                    for p in range(xx.shape[1]):
                        for q in range(5):
                            direct = (
                                sum(
                                    (xx[u, p] - xx[:, p].mean())
                                    * (yy[u, q] - yy[:, q].mean())
                                    for u in range(24)
                                )
                                / 24
                            )
                            contact_error = max(
                                contact_error,
                                abs(
                                    direct
                                    - contacts[f"{term}/{part}/{scale}/{direction}"][p][
                                        q
                                    ]
                                ),
                            )
                            contact_count += 1
            full, choice, within = [
                np.asarray(contacts[f"{term}/{p}/{scale}/pooled"])
                for p in ("full", "choice", "within")
            ]
            split_error = max(split_error, float(np.max(abs(full - choice - within))))
    if (
        max(action_error, cold_error, contact_error, split_error) > 1e-10
        or head_error > 0.001
    ):
        raise ArithmeticError("Declared arithmetic/head fidelity failed")
    result = {
        "PASS": True,
        "literal_route_target_accounts": counts,
        "literal_contacts": contact_count,
        "action_error": action_error,
        "cold_error": cold_error,
        "contact_error": contact_error,
        "head_error": head_error,
        "choice_within_error": split_error,
        "new_model_calls": 0,
        "same_agent_not_blind": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result), flush=True)


def main():
    """Expose independent immutable stages; check cannot resume a collector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(
            args.out,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("ALL_SEALS_PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
