#!/usr/bin/env python3
"""One matched adjacent-evidence reader experiment, never an accuracy grid."""

import argparse
import csv
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
from grain_research.construction import math_behavior_predictive as B
from grain_research.construction import math_late_condition as L
from grain_research.construction import math_same_regime_reader as M
from grain_research.construction import math_source_event_coupling as P
from scripts import run_grain_conditional_likelihood_hm152 as N
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts.run_grain_source_event_coupling_hm273 import field_summary, load

CONFIG = "configs/grain_same_regime_reader_hm274.json"
DEFAULT = ROOT / "results/grain_math_response_v1/same_regime_reader_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_SAME_REGIME_READER_PLAN_HM274.md",
    "grain_research/construction/math_same_regime_reader.py",
    "grain_research/construction/math_source_event_coupling.py",
    "grain_research/construction/math_behavior_predictive.py",
    "grain_research/construction/math_late_condition.py",
    "scripts/run_grain_same_regime_reader_hm274.py",
    "scripts/score_grain_same_regime_reader_cold.py",
    "tests/test_grain_same_regime_reader.py",
)


def prepare(out):
    """Bind fixed adjacent PUBLIC evidence, full tokens and immutable source units."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No replacement of frozen/failed work")
    cfg = R.read(ROOT / CONFIG)
    source, parent, address = [
        ROOT / cfg[k + "_parent"] for k in ("source", "bank", "address")
    ]
    R.check(source, ("computed_seal.json", "verification_seal.json"))
    pm = R.check(parent, ("collected_seal.json", "verification_seal.json"))
    am = R.check(address, ("collected_seal.json", "verification_seal.json"))
    material = ROOT / am["config"]["material"]
    mm = R.read(material / "material_manifest.json")
    if mm["meps_used"] or mm["authors"] != 624:
        raise ValueError("Only the authenticated PUBLIC624 material is authorized")
    for f in ("private_material.json", "private_arrays.npz"):
        if R.digest(material / f) != mm["files"][f]:
            raise ValueError("Changed public-cache identity")
    panel = load(source / "qualified_panel_pairs.npz")
    indices = panel["panel_indices"]
    if len(indices) != 24:
        raise ValueError("All24 original panel authors required")
    expression = (
        "[$idx[] as $i | .bundles[$i] | {author_id,source_group,views:"
        "[.views[0:2][] | [.[5],.[6]] | [.[] | "
        "{author_id,body,comment_id,subreddit,timestamp}]]}]"
    )
    records = json.loads(
        subprocess.check_output(
            [
                "jq",
                "--argjson",
                "idx",
                json.dumps(indices.tolist()),
                expression,
                str(material / "private_material.json"),
            ],
            text=True,
        )
    )
    tok = AutoTokenizer.from_pretrained(str(J.MODEL), local_files_only=True)
    bank = R.read(parent / "bank.json")
    end = tok.convert_tokens_to_ids("<|im_end|>")
    if len(bank) != 50 or any(
        tok.encode(b["completion"], add_special_tokens=False) + [end] != b["target_ids"]
        for b in bank
    ):
        raise ValueError("Complete unchanged public behavior bank required")
    bundle = R.read(ROOT / am["config"]["anonymous_bundle"])
    with np.load(material / "private_arrays.npz", allow_pickle=False) as z:
        author_ids = z["author_ids"][indices]
        numeric = z["comments"][indices, :24].astype(float)
    numeric /= np.linalg.norm(numeric, axis=-1, keepdims=True)
    selected_edges = np.array([22 * u + 11 * w + 5 for u in range(24) for w in (0, 1)])
    pairs = panel["primary"][selected_edges]
    cases, bindings, max_length = [], [], 0
    separator = tok.encode("\n\n", add_special_tokens=False)
    for u, row in enumerate(records):
        if row["author_id"] != author_ids[u] or row["source_group"] != "big5_only":
            raise ValueError("Changed public author binding")
        for w in (0, 1):
            events = row["views"][w]
            if len(events) != 2 or events[0]["comment_id"] == events[1]["comment_id"]:
                raise ValueError("Two distinct full adjacent events required")
            texts = []
            for k, e in enumerate(events):
                slot = 12 * w + 5 + k
                if (
                    e["author_id"] != row["author_id"]
                    or float(e["timestamp"]) != panel["times"][u, slot]
                    or int(e["subreddit"] in bundle["context_rule"]["top"])
                    != panel["conditions"][u, slot]
                ):
                    raise ValueError("Changed adjacent event condition/time/author")
                np.testing.assert_allclose(
                    numeric[u, slot], pairs[2 * u + w, k], atol=1e-12, rtol=0
                )
                texts.append(tok.encode(e["body"], add_special_tokens=False))
            prefixes, suffix = [], None
            for tokens in (
                texts[0] + separator + texts[1],
                texts[1] + separator + texts[0],
                texts[0],
                texts[1],
            ):
                prefix, tail = B.observation_parts(tok, tokens)
                if suffix is not None and suffix != tail:
                    raise ValueError("The neutral suffix changed")
                suffix = tail
                length = (
                    len(prefix)
                    + len(tail)
                    + max(len(b["target_ids"]) for b in bank)
                    - 1
                )
                max_length = max(max_length, length)
                if length > cfg["max_input_tokens"]:
                    raise ValueError(
                        "Fixed complete-token cap exceeded;no pair replacement/clipping"
                    )
                prefixes.append(prefix)
            unit = 2 * u + w
            cases.append(
                {
                    "unit_id": unit,
                    "author": u,
                    "window": w,
                    "prefixes": prefixes,
                    "suffix": suffix,
                }
            )
            bindings.append(
                {
                    "unit_id": unit,
                    "author": u,
                    "window": w,
                    "panel_edge": int(selected_edges[unit]),
                    "comment_ids": [e["comment_id"] for e in events],
                    "token_lengths": [len(t) for t in texts],
                    "times": [float(e["timestamp"]) for e in events],
                }
            )
    files = [
        source / p
        for p in (
            "manifest.json",
            "prepared_seal.json",
            "computed_seal.json",
            "verification_seal.json",
            "source_pair_bank.npz",
            "qualified_panel_pairs.npz",
            "conditional_profiles.npz",
        )
    ]
    files += [
        parent / "manifest.json",
        parent / "bank.json",
        address / "manifest.json",
        material / "material_manifest.json",
        material / "private_material.json",
        material / "private_arrays.npz",
        J.MODEL / "tokenizer.json",
        J.MODEL / "config.json",
    ]
    out.mkdir(parents=True)
    (out / "captures").mkdir()
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "model_weights_sha256": pm["model_weights_sha256"],
            "original_scope_complete": False,
        },
    )
    R.write(out / "cases.json", cases)
    R.write(out / "bank.json", bank)
    R.write(out / "event_bindings.json", bindings)
    R.archive(
        out / "numeric_pairs.npz",
        points=pairs,
        panel_edges=selected_edges,
        windows=np.tile([0, 1], 24),
    )
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "adjacent_pairs": 48,
            "prefix_reads": 192,
            "complete_candidate_sequences": 9600,
            "maximum_complete_tokens": max_length,
            "all_tokens_retained": True,
            "source_disjoint_panel": True,
            "application_labels_read": False,
            "meps_or_other_private_material_read": False,
            "position_restricted_historical_panel_not_iid_pairs": True,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "cases.json",
            "bank.json",
            "event_bindings.json",
            "numeric_pairs.npz",
            "input_quality.json",
        ),
    )
    H.status(out, {"state": "PREPARED", "done_batches": 0, "model_seconds": 0})
    print(json.dumps(R.read(out / "input_quality.json")), flush=True)


def collect(out):
    """Observe every fixed context once; save failures and never renew the cap."""
    import torch

    manifest = R.check(out)
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No retry of a scientific collection")
    cfg, cases, bank = (
        manifest["config"],
        R.read(out / "cases.json"),
        R.read(out / "bank.json"),
    )
    lengths = np.array([len(b["target_ids"]) for b in bank])
    chosen, partition = np.zeros((48, 4, 50)), np.zeros((48, 4, 50))
    done, seconds, pending, files = 0, 0.0, None, []
    errors = {"head": 0.0, "native": 0.0, "prefix_mutation": 0.0}
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))

    def mark(state, error=None):
        """Maintain the actual live operation and nonrenewable resource account."""
        H.status(
            out,
            {
                "state": state,
                "pid": os.getpid(),
                "done_batches": done,
                "model_seconds": seconds,
                "pending": pending,
                "error": error,
                "instrument_errors": errors,
            },
        )

    def timed(fn):
        """Charge even failed model calls; adequate precision is not optimized."""
        nonlocal seconds
        if seconds >= cfg["max_model_seconds"]:
            raise TimeoutError("Fixed900-model-second cap exhausted")
        start = time.monotonic()
        try:
            value = fn()
            torch.mps.synchronize()
            return value
        finally:
            seconds += time.monotonic() - start

    try:
        mark("RUNNING")
        observer = J.offline_model(manifest)
        for case in cases:
            u = case["unit_id"]
            for c, prefix in enumerate(case["prefixes"]):
                pending = {"unit": u, "context": c, "operation": "encode"}
                mark("RUNNING")
                cache = timed(partial(L.encode_context, observer, prefix))
                before = L.cache_arrays(cache)
                all_rows = []
                for start in range(0, 50, cfg["batch_size"]):
                    targets = [
                        b["target_ids"] for b in bank[start : start + cfg["batch_size"]]
                    ]
                    pending = {
                        "unit": u,
                        "context": c,
                        "start": start,
                        "operation": "complete_candidate",
                    }
                    mark("RUNNING")
                    rows = timed(
                        partial(
                            B.observe_candidates,
                            observer,
                            cache,
                            case["suffix"],
                            targets,
                        )
                    )
                    all_rows.extend(rows)
                    done += 1
                    for row in rows:
                        replay = (
                            np.sum(
                                row["states"].astype(float)
                                * row["covectors"].astype(float),
                                -1,
                            )
                            + row["bias"]
                        )
                        errors["head"] = max(
                            errors["head"], float(np.max(abs(replay - row["chosen"])))
                        )
                    if u == 0 and start == 0:
                        native = timed(
                            partial(
                                N.observe,
                                observer,
                                [(prefix + case["suffix"], t) for t in targets[:2]],
                            )
                        )
                        differences = np.concatenate(
                            [
                                row["chosen"]
                                - row["partition"]
                                - (
                                    native["chosen"][i, : len(t)]
                                    - native["partition"][i, : len(t)]
                                ).astype(float)
                                for i, (row, t) in enumerate(
                                    zip(rows[:2], targets[:2], strict=True)
                                )
                            ]
                        )
                        errors["native"] = max(
                            errors["native"], float(np.max(abs(differences)))
                        )
                        name = f"native_guard_{c}.npz"
                        R.archive(out / name, differences=differences)
                        files.append(name)
                errors["prefix_mutation"] = max(
                    errors["prefix_mutation"],
                    max(
                        float(np.max(abs(a - b)))
                        for a, b in zip(before, L.cache_arrays(cache), strict=True)
                    ),
                )
                if max(errors.values()) > cfg["native_tolerance"]:
                    raise ArithmeticError(
                        "Instrument error can affect this observation;no precision rescue"
                    )
                chosen[u, c] = [r["chosen"].sum() for r in all_rows]
                partition[u, c] = [r["partition"].sum() for r in all_rows]
                name = f"captures/{u:02d}_{c}.npz"
                R.archive(
                    out / name,
                    chosen_tokens=np.concatenate([r["chosen"] for r in all_rows]),
                    partition_tokens=np.concatenate([r["partition"] for r in all_rows]),
                    offsets=np.r_[0, lengths.cumsum()],
                )
                files.append(name)
            print(
                f"OBSERVED {u + 1}/48 pairs; {done} candidate batches; {seconds:.3f}s model",
                flush=True,
            )
        R.archive(
            out / "observed_sums.npz",
            chosen=chosen,
            partition=partition,
            lengths=lengths,
        )
        files.append("observed_sums.npz")
        R.seal(out, "collected_seal.json", files)
        pending = None
        mark("COLLECTED")
    except BaseException as exc:
        mark("PARTIAL_BUDGET" if isinstance(exc, TimeoutError) else "FAILED", str(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)


def analyze(out):
    """Compare source and reader on the same adjacent events without fitting."""
    manifest = R.check(out, ("collected_seal.json",))
    R.write(out / "analysis_started.json", {"state": "RUNNING"})
    began = time.perf_counter()
    cfg = manifest["config"]
    old = ROOT / cfg["source_parent"]
    source, panel = load(old / "source_pair_bank.npz"), load(out / "numeric_pairs.npz")
    sums, bank = load(out / "observed_sums.npz"), R.read(out / "bank.json")
    packet = {
        "version": "HM274-v1",
        "bank": bank,
        **{k: v.tolist() for k, v in sums.items()},
    }
    reading = M.cold(packet)
    R.write(out / "cold_packet.json", packet)
    R.write(out / "cold_readback.json", reading)
    points = np.stack([panel["points"], panel["points"][:, ::-1]], 1).reshape(
        96, 2, 1024
    )
    fields, rows, summary = {}, [], {"matched": {}, "source_recurrence": {}}
    closure = 0.0
    for space in ("E5", "GRAIN15"):
        x, z = points, source["prototypes"]
        if space == "GRAIN15":
            x, z = x @ source["projection"], z @ source["projection"]
        logs = P.event_logs(x, z, float(source["variance"]))
        for assignment, codes in (
            ("actual", source["codes"]),
            ("fixed_profile_relink", source["null_codes"]),
        ):
            group = f"{assignment}/{space}"
            for label, mask in (
                ("bank0", source["folds"] == 0),
                ("bank1", source["folds"] == 1),
                ("pooled", np.ones(386, bool)),
            ):
                result = P.profile_readings(
                    logs[:, mask], codes[mask], source["relink"][mask]
                )
                for k in ("profile", "targets", "log_odds", "log_laws"):
                    fields[f"{group}/{label}/{k}"] = result[k].reshape(48, 2, 4, -1)
                values = fields[f"{group}/{label}/targets"]
                summary["matched"][f"{group}/{label}"] = {}
                for component, action in P.components(values).items():
                    for scale in ("sum", "per_token"):
                        behavior = np.array(
                            reading["accounts"][scale]["balanced_key"]["values"]
                        )
                        for window, ix in (
                            ("0", panel["windows"] == 0),
                            ("1", panel["windows"] == 1),
                            ("pooled", np.ones(48, bool)),
                        ):
                            cs = M.contacts(action[ix], behavior[ix])
                            closure = max(
                                closure,
                                float(
                                    np.max(
                                        abs(
                                            cs["raw"]
                                            - cs["join"]
                                            - cs["order"]
                                            - cs["single"]
                                        )
                                    )
                                ),
                            )
                            summary["matched"][f"{group}/{label}"][
                                f"{component}/{scale}/{window}"
                            ] = {k: np.diag(v).tolist() for k, v in cs.items()}
                            for channel in ("join", "order"):
                                for a, target in enumerate(B.TARGETS):
                                    for b, reader in enumerate(B.TARGETS):
                                        rows.append(
                                            {
                                                "assignment": assignment,
                                                "space": space,
                                                "bank": label,
                                                "component": component,
                                                "scale": scale,
                                                "window": window,
                                                "channel": channel,
                                                "source_target": target,
                                                "reader_target": reader,
                                                "contact": float(cs[channel][a, b]),
                                                "pairs": int(ix.sum()),
                                            }
                                        )
            b0, b1 = [
                P.components(fields[f"{group}/{b}/targets"]) for b in ("bank0", "bank1")
            ]
            summary["source_recurrence"][group] = {
                c: {
                    k: field_summary(M.source_parts(b0[c])[k], M.source_parts(b1[c])[k])
                    for k in ("even", "odd")
                }
                for c in P.COMPONENTS
            }
    R.archive(out / "source_readings.npz", **fields)
    with (out / "all_contacts.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary.update(
        {
            "contact_closure_error": closure,
            "contact_rows": len(rows),
            "CPU_analysis_seconds": time.perf_counter() - began,
            "original_scope_complete": False,
            "no_prediction_accuracy_significance_or_new_theorem_claim": True,
        }
    )
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "cold_packet.json",
            "cold_readback.json",
            "source_readings.npz",
            "all_contacts.csv",
            "summary.json",
        ),
    )
    print(
        "ANALYZED same-regime joint/order contacts; no selected target or fitted transport",
        flush=True,
    )


def verify(out):
    """Independently resum observations and consequential signed contact products."""
    manifest = R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    sums, fields, panel = [
        load(out / p)
        for p in ("observed_sums.npz", "source_readings.npz", "numeric_pairs.npz")
    ]
    errors = {"token_resum": 0.0, "old_source_forward": 0.0, "literal_contacts": 0.0}
    for u in range(48):
        for c in range(4):
            cap = load(out / f"captures/{u:02d}_{c}.npz")
            for k in ("chosen", "partition"):
                direct = np.array(
                    [
                        sum(float(v) for v in cap[k + "_tokens"][a:b])
                        for a, b in zip(
                            cap["offsets"][:-1], cap["offsets"][1:], strict=True
                        )
                    ]
                )
                errors["token_resum"] = max(
                    errors["token_resum"], float(np.max(abs(direct - sums[k][u, c])))
                )
    old = load(ROOT / manifest["config"]["source_parent"] / "conditional_profiles.npz")
    for key, value in fields.items():
        prefix, field = key.rsplit("/", 1)
        old_prefix = prefix.replace(
            "fixed_profile_relink", "fixed_within_bank_profile_relink"
        )
        errors["old_source_forward"] = max(
            errors["old_source_forward"],
            float(
                np.max(
                    abs(
                        value[:, 0] - old[f"{old_prefix}/{field}"][panel["panel_edges"]]
                    )
                )
            ),
        )
    behavior = R.read(out / "cold_readback.json")["accounts"]
    with (out / "all_contacts.csv").open() as f:
        for row in csv.DictReader(f):
            key = f"{row['assignment']}/{row['space']}/{row['bank']}/targets"
            values = P.components(fields[key])[row["component"]]
            ix = (
                np.ones(48, bool)
                if row["window"] == "pooled"
                else panel["windows"] == int(row["window"])
            )
            a = M.source_parts(values)["even" if row["channel"] == "join" else "odd"]
            b = np.array(
                behavior[row["scale"]]["balanced_key"]["parts"][row["channel"]]
            )
            i, j = (
                B.TARGETS.index(row["source_target"]),
                B.TARGETS.index(row["reader_target"]),
            )
            literal = sum(
                float(x) * float(y) for x, y in zip(a[ix, i], b[ix, j], strict=True)
            ) / int(ix.sum())
            errors["literal_contacts"] = max(
                errors["literal_contacts"], abs(literal - float(row["contact"]))
            )
    cli = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(ROOT / "scripts/score_grain_same_regime_reader_cold.py"),
                str(out / "cold_packet.json"),
            ],
            text=True,
        )
    )
    if cli != R.read(out / "cold_readback.json") or max(errors.values()) > 1e-10:
        raise ArithmeticError("Frozen numerical readback failed")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "errors": errors,
            "cold_CLI_equal": True,
            "same_agent_not_blind": True,
            "original_scope_complete": False,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(R.read(out / "verification.json")), flush=True)


def main():
    """Run finite stages explicitly; none auto-retries or starts another packet."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage == "check":
        R.check(
            args.out,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM274 input/output seals PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
