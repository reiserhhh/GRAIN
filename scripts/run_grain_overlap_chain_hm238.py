#!/usr/bin/env python3
"""One compatible overlapping-evidence construction, without prediction search."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_overlap_chain as M
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_path_edit_hm237 as P
from scripts import run_grain_source_meaning_bridge_hm233 as S

CONFIG = "configs/grain_overlap_chain_hm238.json"
DEFAULT = ROOT / "results/grain_math_response_v1/overlap_chain_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_OVERLAP_CHAIN_PLAN_HM238.md",
    "grain_research/construction/math_overlap_chain.py",
    "scripts/run_grain_overlap_chain_hm238.py",
    "tests/test_grain_overlap_chain.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_source_meaning_bridge_hm233.py",
    "scripts/run_grain_path_edit_hm237.py",
)
STAGES = (
    "compiled_seal.json",
    "cold_seal.json",
    "analyzed_seal.json",
    "verification_seal.json",
)


def prepare(out):
    """Freeze existing SOURCE and position-qualified application identities."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    R.check(parent, P.STAGES)
    source = S.selected(parent / "source_inputs.npz", ("source_paths", "targets"))
    records = [
        json.loads(s) for s in (parent / "cold_inputs.jsonl").read_text().splitlines()
    ]
    quality = R.read(parent / "input_quality.json")
    if (
        source["source_paths"].shape != (386, 2, 18)
        or source["targets"].shape != (2, 386, 5)
        or len(records) != 264
    ):
        raise ValueError("Changed complete inherited SOURCE/application scope")
    inputs = []
    for r in records:
        for arm in ("full", "support", "counter"):
            removed = set() if arm == "full" else set(r["edits"][arm])
            events = [
                {
                    "event_id": e["event_id"],
                    "position": e["position"] % 18,
                    "cell": e["cell"],
                }
                for e in r["events"]
                if e["event_id"] not in removed
            ]
            case = {
                "case_key": r["author_key"] + "/" + arm,
                "author_key": r["author_key"],
                "window_size": 18,
                "events": events,
            }
            M.observed_vector(case, 8, 18)
            inputs.append(case)
    inputs.append(
        {
            "case_key": "empty/control",
            "author_key": "control",
            "window_size": 18,
            "events": [],
        }
    )
    if len(inputs) != cfg["cold_cases"]:
        raise ValueError("Incomplete fixed input families")
    out.mkdir(parents=True, exist_ok=False)
    files = [
        parent / s
        for s in (
            "manifest.json",
            "source_inputs.npz",
            "cold_inputs.jsonl",
            "input_quality.json",
            *P.STAGES,
        )
    ]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "previous_turn": "NO_PROGRESS_EXECUTION:direction review only;no new scientific state",
            "original_scope_complete": False,
        },
    )
    R.archive(out / "source_inputs.npz", **source)
    with (out / "cold_inputs.jsonl").open("x") as f:
        for r in inputs:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    R.write(
        out / "input_quality.json",
        {
            "inherited": quality,
            "histories": 264,
            "distinct_application_authors": 140,
            "cases": len(inputs),
            "native_phase_positions": [6, 17],
            "deleted_states_absent_from_each_arm": True,
            "new_neural_API_private_training_application_labels": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("source_inputs.npz", "cold_inputs.jsonl", "input_quality.json"),
    )
    print("PREPARED", len(inputs), flush=True)


def compile_source(out):
    """Export only SOURCE aggregates and document the law-completion assumption."""
    cfg = R.check(out)["config"]
    source = S.selected(out / "source_inputs.npz", ("source_paths", "targets"))
    start = time.perf_counter()
    arrays, accounts = {}, []
    for i, y in enumerate(source["targets"]):
        b = M.compile_source(source["source_paths"], y, cfg["states"])
        arrays.update({f"{k}_{i}": v for k, v in b.items()})
        actual_cliques = M.clique_marginals(b["initial"], b["transitions"])
        product_cliques = M.clique_marginals(b["product_initial"], b["product_factors"])
        empty = np.full(18, -1, int)
        raw_initial = np.broadcast_to(
            b["prior"][:, None, None], b["initial"].shape
        ).copy()
        raw_profile, _ = M.propagate(raw_initial, b["product_factors"], empty)
        accounts.append(
            {
                "law": cfg["laws"][i],
                "separator_consistency_error": float(b["consistency_error"]),
                "chain_clique_max_error": float(
                    np.max(abs(actual_cliques - b["cliques"]))
                ),
                "product_clique_max_difference": float(
                    np.max(abs(product_cliques - b["cliques"]))
                ),
                "product_clique_mean_TV": float(
                    np.sum(
                        abs(product_cliques - b["cliques"]), axis=(1, 2, 3, 4)
                    ).mean()
                    / 2
                ),
                "raw_product_empty_prior_TV": float(
                    np.sum(abs(raw_profile - b["prior"])) / 2
                ),
                "raw_product_empty_target_changes": (
                    (raw_profile - b["prior"]) @ b["profiles"]
                ).tolist(),
                **M.source_entropy_accounts(source["source_paths"], y, b),
            }
        )
    seconds = time.perf_counter() - start
    if seconds > cfg["max_cpu_seconds"]:
        raise TimeoutError("Finite compile cap exceeded;no retry")
    R.archive(out / "cold_bundle.npz", **arrays)
    R.write(
        out / "source_accounts.json",
        {
            "accounts": accounts,
            "CPU_wall_seconds": seconds,
            "clique_match_is_by_construction": True,
            "Markov_completion_not_identified_human_law": True,
        },
    )
    R.seal(out, "compiled_seal.json", ("cold_bundle.npz", "source_accounts.json"))
    print(json.dumps({"SOURCE_COMPILED": seconds, "accounts": accounts}), flush=True)


def bundles(path):
    """Load only aggregate law tables; no source authors or application outcomes."""
    with np.load(path, allow_pickle=False) as z:
        return [
            {
                k: z[f"{k}_{i}"]
                for k in (
                    "profiles",
                    "prior",
                    "initial",
                    "transitions",
                    "product_initial",
                    "product_factors",
                    "cliques",
                )
            }
            for i in (0, 1)
        ]


def cold_runtime(bundle, inputs, output):
    """Execute whole-path masked inference without model weights/text/labels."""
    laws = bundles(bundle)
    with inputs.open() as f, output.open("x") as dest:
        for line in f:
            record = json.loads(line)
            dest.write(
                json.dumps(
                    {
                        "case_key": record["case_key"],
                        "laws": [M.score(b, record) for b in laws],
                    },
                    allow_nan=False,
                )
                + "\n"
            )


def cold(out):
    """Run the aggregate-bundle consumer in an actual separate process, once."""
    cfg = R.check(out, STAGES[:1])["config"]
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "cold-runtime",
        "--bundle",
        str(out / "cold_bundle.npz"),
        "--inputs",
        str(out / "cold_inputs.jsonl"),
        "--output",
        str(out / "cold_outputs.jsonl"),
    ]
    start = time.perf_counter()
    p = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=cfg["max_cpu_seconds"],
        check=False,
    )
    R.write(
        out / "cold_receipt.json",
        {
            "command": command,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "CPU_wall_seconds": time.perf_counter() - start,
            "SOURCE_authors_or_labels_read": False,
            "new_model_calls": 0,
        },
    )
    p.check_returncode()
    R.seal(out, "cold_seal.json", ("cold_outputs.jsonl", "cold_receipt.json"))
    print("COLD_COMPLETE", flush=True)


def analyze(out):
    """Keep every law/cohort/arm/target and all unsupported histories."""
    cfg = R.check(out, STAGES[:2])["config"]
    rows = [
        json.loads(s) for s in (out / "cold_outputs.jsonl").read_text().splitlines()
    ]
    if len(rows) != cfg["cold_cases"]:
        raise ValueError("Missing cold cases")
    cases = {r["case_key"]: r for r in rows}
    readings, support, actions = [], [], []
    for i, name in enumerate(cfg["laws"]):
        for cohort in ("held", "native"):
            subset = [r for r in rows if r["case_key"].startswith(cohort + "/")]
            for arm in ("full", "support", "counter"):
                armrows = [r for r in subset if r["case_key"].endswith("/" + arm)]
                for mode in ("chain", "normalized_product"):
                    known = [r for r in armrows if r["laws"][i][mode]["supported"]]
                    support.append(
                        {
                            "law": name,
                            "cohort": cohort,
                            "arm": arm,
                            "mode": mode,
                            "cases": len(armrows),
                            "supported": len(known),
                        }
                    )
            for r in subset:
                value = r["laws"][i]
                for mode in ("chain", "normalized_product"):
                    scores = value[mode]
                    for q, target in enumerate(cfg["targets"]):
                        readings.append(
                            {
                                "case_key": r["case_key"],
                                "law": name,
                                "mode": mode,
                                "target": target,
                                "supported": scores["supported"],
                                "reading": scores["targets"][q]
                                if scores["supported"]
                                else None,
                                "positive_profiles": int(
                                    np.count_nonzero(scores["profile"])
                                )
                                if scores["supported"]
                                else None,
                            }
                        )
            histories = sorted({r["case_key"].rsplit("/", 1)[0] for r in subset})
            for h in histories:
                a = [
                    cases[h + "/" + arm]["laws"][i]
                    for arm in ("full", "support", "counter")
                ]
                for mode in ("chain", "normalized_product"):
                    complete = all(v[mode]["supported"] for v in a)
                    for q, target in enumerate(cfg["targets"]):
                        diff = (
                            (a[2][mode]["targets"][q] - a[1][mode]["targets"][q])
                            if complete
                            else None
                        )
                        actions.append(
                            {
                                "history": h,
                                "cohort": cohort,
                                "law": name,
                                "mode": mode,
                                "target": target,
                                "complete": complete,
                                "counter_minus_support": diff,
                            }
                        )
    R.table(out / "support.csv", support)
    R.table(out / "readings.csv", readings)
    R.table(out / "actions.csv", actions)
    paired = []
    for i, name in enumerate(cfg["laws"]):
        for cohort in ("held", "native"):
            for arm in ("full", "support", "counter"):
                both = [
                    r
                    for r in rows
                    if r["case_key"].startswith(cohort + "/")
                    and r["case_key"].endswith("/" + arm)
                    and all(
                        r["laws"][i][m]["supported"]
                        for m in ("chain", "normalized_product")
                    )
                ]
                diffs = np.array(
                    [
                        np.array(r["laws"][i]["normalized_product"]["targets"])
                        - r["laws"][i]["chain"]["targets"]
                        for r in both
                    ]
                ).reshape(-1, 5)
                for q, target in enumerate(cfg["targets"]):
                    paired.append(
                        {
                            "law": name,
                            "cohort": cohort,
                            "arm": arm,
                            "target": target,
                            "paired": len(both),
                            "signed_mean_product_minus_chain": float(diffs[:, q].mean())
                            if len(both)
                            else None,
                            "RMS_difference": float(np.sqrt(np.mean(diffs[:, q] ** 2)))
                            if len(both)
                            else None,
                        }
                    )
    R.table(out / "paired_laws.csv", paired)
    R.write(
        out / "summary.json",
        {
            "support": support,
            "paired_laws": paired,
            "empty_controls": cases["empty/control"],
            "source": R.read(out / "source_accounts.json"),
            "accuracy_promotion": False,
            "native_contacts_computed": False,
            "application_truth_read": False,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "support.csv",
            "readings.csv",
            "actions.csv",
            "paired_laws.csv",
            "summary.json",
        ),
    )
    print(json.dumps({"support": support}), flush=True)


def literal_masses(initial, factors, observation):
    """Independent unscaled contraction using explicit next-cell slices."""
    _g, k, _ = initial.shape
    masks = np.ones((len(observation), k))
    for t, cell in enumerate(observation):
        if cell >= 0:
            masks[t] = np.eye(k)[cell]
    state = initial * masks[0][None, :, None] * masks[1][None, None, :]
    for t, factor in enumerate(factors):
        following = np.zeros_like(state)
        for c in range(k):
            following[:, :, c] = (state * factor[:, :, :, c]).sum(1) * masks[t + 2, c]
        state = following
    return state.sum((1, 2))


def verify(out):
    """Audit actual source cliques and every cold output with separate arithmetic."""
    cfg = R.check(out, STAGES[:3])["config"]
    source = S.selected(out / "source_inputs.npz", ("source_paths", "targets"))
    laws = bundles(out / "cold_bundle.npz")
    records = [
        json.loads(s) for s in (out / "cold_inputs.jsonl").read_text().splitlines()
    ]
    outputs = [
        json.loads(s) for s in (out / "cold_outputs.jsonl").read_text().splitlines()
    ]
    source_error, posterior_error, log_error, count, missing = 0.0, 0.0, 0.0, 0, 0
    for i, b in enumerate(laws):
        literal = np.zeros_like(b["cliques"])
        for u, author in enumerate(source["source_paths"]):
            code = int(source["targets"][i, u] @ (1 << np.arange(5)))
            for path in author:
                for t in range(16):
                    literal[t, code, *path[t : t + 3]] += 1
        literal /= 772
        source_error = max(source_error, float(np.max(abs(literal - b["cliques"]))))
        for record, result in zip(records, outputs, strict=True):
            if record["case_key"] != result["case_key"]:
                raise ValueError("Changed cold identities")
            obs = np.full(18, -1, int)
            for e in record["events"]:
                obs[e["position"]] = e["cell"]
            for mode, initial, factors in (
                ("chain", b["initial"], b["transitions"]),
                ("normalized_product", b["product_initial"], b["product_factors"]),
            ):
                mass = literal_masses(initial, factors, obs)
                value = result["laws"][i][mode]
                if mass.sum() == 0:
                    missing += 1
                    if value["supported"] or value["targets"] is not None:
                        raise ValueError("Unsupported path promoted into a score")
                    continue
                if not value["supported"]:
                    raise ValueError("Supported path lost")
                posterior = mass / mass.sum()
                posterior_error = max(
                    posterior_error,
                    float(np.max(abs(posterior - value["profile"]))),
                    float(np.max(abs(posterior @ b["profiles"] - value["targets"]))),
                )
                log_error = max(
                    log_error, abs(float(np.log(mass.sum())) - value["log_evidence"])
                )
                count += 1
    errors = {
        "source_clique_error": source_error,
        "posterior_error": posterior_error,
        "log_mass_error": log_error,
    }
    if max(errors.values()) > cfg["audit_tolerance"]:
        raise ValueError("Finite same-agent alternate audit failed")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            **errors,
            "supported_law_case_mode_accounts": count,
            "missing_accounts_retained": missing,
            "same_agent_not_blind_replication": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(
        json.dumps({"PASS": True, **errors, "supported": count, "missing": missing}),
        flush=True,
    )


def main():
    """Run exclusive stages; read-only check never re-executes science."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "compile",
            "cold",
            "analyze",
            "verify",
            "check",
            "cold-runtime",
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    for name in ("bundle", "inputs", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.stage == "cold-runtime":
        cold_runtime(args.bundle, args.inputs, args.output)
    elif args.stage == "check":
        R.check(args.out, STAGES)
        print("ALL_SEALS_PASS")
    else:
        {
            "prepare": prepare,
            "compile": compile_source,
            "cold": cold,
            "analyze": analyze,
            "verify": verify,
        }[args.stage](args.out)


if __name__ == "__main__":
    main()
