#!/usr/bin/env python3
"""Freeze and test source-only actor actions without prediction/metric search."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_actor_action as M
from grain_research.construction import math_native_joint_batch as B
from grain_research.construction import math_native_joint_value as V
from grain_research.construction.math_construct_routing import query
from scripts import run_grain_action_expression_hm115 as P
from scripts import run_grain_author_evidence_merge_hm102 as H

SPEC = ROOT / "configs/grain_actor_action_hm116.json"
CODE = (
    "configs/grain_actor_action_hm116.json",
    "docs/GRAIN_ACTOR_ACTION_PLAN_HM116.md",
    "grain_research/construction/math_actor_action.py",
    "scripts/run_grain_actor_action_hm116.py",
    "tests/test_grain_math_actor_action.py",
    "grain_research/construction/math_action_expression.py",
    "grain_research/construction/math_conditional_chart.py",
    "grain_research/construction/math_native_joint_batch.py",
    "grain_research/construction/math_native_joint_value.py",
    "grain_research/construction/math_construct_routing.py",
    "grain_research/construction/math_joint_reading.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/prefix.py",
    "grain_research/vector_mechanisms/navigation.py",
    "scripts/run_grain_action_expression_hm115.py",
    "scripts/run_grain_author_evidence_merge_hm102.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
)
UNITS = ("score", "log_odds", "mass", "entropy")


def check(out):
    """Authenticate scientific code, source lineage and precommitted actions."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    return m


def prepare(out):
    """Construct actions from original V only and seal all new native prompts."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL, write_json

    if out.exists():
        raise FileExistsError("Preserve terminal, failed and partial experiments")
    spec = H.load(SPEC)
    src = ROOT / spec["source"]
    old = P.check(src)
    H.verify_hashes(src, H.load(src / "01/raw_seal.json"))
    H.verify_hashes(src, H.load(src / "02/output_seal.json"))
    if (
        H.load(src / "verification.json")["status"]
        != "VERIFIED_FINITE_HM115_NOT_GLOBAL"
    ):
        raise ValueError("Verified original-expression source required")
    if spec["source_variant_wording"] != [0, 0] or spec["arms"] != list(M.ARMS):
        raise ValueError("Source/action selection is forbidden")
    shape = P.raw_unit(src, 0)["v"].shape
    with np.load(src / "02/fields.npz", allow_pickle=False) as z:
        x = z["v"]
    actions = M.source_actions(x.reshape(*x.shape[:-1], *shape), spec["random_seed"])
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    system = ROOT / "configs/grain_math_passive_event_composition_v1.json"
    marker = "__HM116_BODY__"
    template = tok.apply_chat_template(
        [
            {"role": "system", "content": H.load(system)["system"]},
            {"role": "user", "content": marker},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix, closing = template.split(marker)
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    labels = [str(i) for i in range(10)]
    ids = [encode(s) for s in labels]
    if any(len(i) != 1 for i in ids) or len({i[0] for i in ids}) != 10:
        raise ValueError("Ten single-token candidate digits required")
    questions = [
        query(t).replace("the writer's", f"Person {p}'s")
        for p in ("A", "B")
        for t in spec["targets"]
    ]
    queries = {
        "questions": questions,
        "suffixes": [
            encode("\n\nEvidence ends here.\n\nQuestion: " + q + "\nAnswer:" + closing)
            for q in questions
        ],
        "alphabet": {
            "labels": labels,
            "values": (np.arange(10) / 9).tolist(),
            "token_ids": [i[0] for i in ids],
        },
    }
    rows = [
        r
        for r in H.load(src / "00/rows.json")
        if r["variant"] == spec["application_variant"]
        and r["wording"] in spec["application_wordings"]
    ]
    if len(rows) != 96 or any(
        r["input_ids"][: len(encode(prefix))] != encode(prefix) for r in rows
    ):
        raise ValueError("Complete new-instance support and inherited system required")
    design = []
    for arm in (*M.ARMS, "noop"):
        for row in rows:
            if (
                arm == "noop"
                and [row[k] for k in ("wording", "family", "condition")]
                != spec["noop_group"]
            ):
                continue
            design.append({"batch": len(design), "row": row["id"], "arm": arm})
    if len(design) != spec["max_batches"] or len(design) * 12 != spec["max_endpoints"]:
        raise ValueError("Full planned finite battery required")
    folder = out / "00"
    folder.mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    for name, value in (("rows", rows), ("design", design), ("queries", queries)):
        H.write(folder / f"{name}.json", value)
    np.savez_compressed(folder / "actions.npz", **actions)
    norms = [
        {
            "arm": arm,
            "family": f,
            "condition": 2 * c - 1,
            "head": h,
            "norm": float(np.linalg.norm(a[f, c, :, h])),
        }
        for arm, a in actions.items()
        for f in range(6)
        for c in range(2)
        for h in range(shape[1])
    ]
    H.table(folder / "control_norms.csv", norms)
    sources = [
        src / n
        for n in (
            "00/manifest.json",
            "00/rows.json",
            "01/raw_seal.json",
            "02/output_seal.json",
            "02/fields.npz",
            "verification.json",
        )
    ] + [system]
    H.write(
        folder / "manifest.json",
        {
            "spec": spec,
            "model": old["model"],
            "code": {n: H.sha(ROOT / n) for n in CODE},
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in sources},
            "inputs": H.seals(out, folder),
            "physical_shape": list(shape),
            "model_blocks": H.load(MODEL / "config.json")["num_hidden_layers"],
            "source_only": True,
            "application/native_scores_used_to_construct_actions": False,
            "original_scope_complete": False,
        },
    )
    write_json(
        out / "status.json",
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "model_seconds": 0.0,
            "pending": None,
        },
    )
    print(
        "PREPARED HM116",
        len(design),
        "batches",
        len(design) * 12,
        "endpoints",
        flush=True,
    )


def raw_unit(out, index):
    """Read only a fully committed native batch, never replay an uncertain call."""
    stem = out / f"01/units/b{index:03d}"
    meta = H.load(stem.with_suffix(".json"))
    if H.sha(stem.with_suffix(".npz")) != meta["sha256"]:
        raise ValueError("Native capture changed")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def collect(out):
    """Run once under a cumulative finite cap, charging failed/load work too."""
    from grain_research.vector_mechanisms.navigation import MODEL, write_json
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    m, status = check(out), H.load(out / "status.json")
    if status["state"] not in ("PREPARED", "RUNNING") or status["pending"] is not None:
        raise RuntimeError("Terminal or uncertain experiment cannot restart")
    H.verify_hashes(MODEL, m["model"])
    rows = {r["id"]: r for r in H.load(out / "00/rows.json")}
    design, queries = H.load(out / "00/design.json"), H.load(out / "00/queries.json")
    with np.load(out / "00/actions.npz", allow_pickle=False) as z:
        actions = {k: z[k] for k in z.files}
    try:
        with collector_guard(out):
            start = time.monotonic()
            try:
                observer = load_observer()
            finally:
                status["model_seconds"] += time.monotonic() - start
                write_json(out / "status.json", status)
            H.write(out / "model_manifest.json", observer.manifest())
            for item in design[status["completed"] :]:
                if status["model_seconds"] >= m["spec"]["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    raise TimeoutError("Finite model cap exhausted; no renewal")
                row, arm = rows[item["row"]], item["arm"]
                stem = out / f"01/units/b{item['batch']:03d}"
                if (
                    stem.with_suffix(".npz").exists()
                    or stem.with_suffix(".json").exists()
                ):
                    raise RuntimeError("Uncertain existing call must not be retried")
                delta = None if arm == "baseline" else np.zeros(m["physical_shape"])
                if arm in actions:
                    delta = (
                        -row["polarity"]
                        * row["owner"]
                        * actions[arm][row["family"], (row["condition"] + 1) // 2]
                    )
                status.update(
                    state="RUNNING", pending=item, attempts=status["attempts"] + 1
                )
                write_json(out / "status.json", status)
                start = time.monotonic()
                try:
                    readings, arrays = B.native_batch(
                        observer,
                        row,
                        queries["suffixes"],
                        queries["alphabet"],
                        delta,
                        m["spec"]["layer"],
                    )
                finally:
                    seconds = time.monotonic() - start
                    status["model_seconds"] += seconds
                    write_json(out / "status.json", status)
                logits = np.array([r["candidate_logits"] for r in readings])
                norm = np.array([r["full_vocab_log_normalizer"] for r in readings])
                arrays.update(
                    logits=logits,
                    normalizer=norm,
                    readings=np.array(
                        [
                            V.native_units(z, n)
                            for z, n in zip(logits, norm, strict=True)
                        ]
                    ),
                    requested_delta=np.zeros(m["physical_shape"])
                    if delta is None
                    else delta,
                )
                with stem.with_suffix(".npz").open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                H.write(
                    stem.with_suffix(".json"),
                    {
                        **item,
                        "sha256": H.sha(stem.with_suffix(".npz")),
                        "seconds": seconds,
                        "endpoints": 12,
                    },
                )
                status.update(completed=status["completed"] + 1, pending=None)
                write_json(out / "status.json", status)
                if status["completed"] % 24 == 0:
                    print("HM116", status, flush=True)
            status["state"] = (
                "COLLECTED"
                if status["model_seconds"] <= m["spec"]["max_model_seconds"]
                else "BUDGET_STOP"
            )
            write_json(out / "status.json", status)
            if status["state"] != "COLLECTED":
                raise TimeoutError("Finite cap crossed; retain results")
            H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    except BaseException as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "status.json", status)
        raise
    print("COLLECTED HM116", status, flush=True)


def assemble(out):
    """Check actual masks/edits and populate all native endpoints without fits."""
    m = check(out)
    rows = {r["id"]: r for r in H.load(out / "00/rows.json")}
    design = H.load(out / "00/design.json")
    baselines = {i["row"]: i["batch"] for i in design if i["arm"] == "baseline"}
    native = np.full((5, 2, 6, 2, 2, 2, 2, 6, 4), np.nan)
    errors = {k: 0.0 for k in ("edit", "key", "before", "noop", "readout", "requested")}
    with np.load(out / "00/actions.npz", allow_pickle=False) as z:
        actions = {k: z[k] for k in z.files}
    short_errors = {"k": 0.0, "v": 0.0}
    src = ROOT / m["spec"]["source"]
    for item in design:
        raw, base, row = (
            raw_unit(out, item["batch"]),
            raw_unit(out, baselines[item["row"]]),
            rows[item["row"]],
        )
        if raw["readings"].shape != (12, 4) or not np.array_equal(
            raw["masked_blocks"], np.tile(np.arange(m["model_blocks"]), (12, 1))
        ):
            raise ValueError("Incomplete native endpoint/layer trace")
        for name in (
            "support_hash",
            "execution_support_hash",
            "masked_blocks",
            "execution_length",
            "endpoint",
        ):
            if not np.array_equal(raw[name], base[name]):
                raise ValueError("Intervention changed actual execution support")
        errors["key"] = max(errors["key"], float(np.max(abs(raw["k"] - base["k"]))))
        errors["before"] = max(
            errors["before"], float(np.max(abs(raw["before"] - base["before"])))
        )
        errors["edit"] = max(
            errors["edit"],
            float(np.max(abs(raw["after"] - raw["before"] - raw["requested_delta"]))),
        )
        requested = np.zeros(m["physical_shape"])
        if item["arm"] in actions:
            requested = (
                -row["polarity"]
                * row["owner"]
                * actions[item["arm"]][row["family"], (row["condition"] + 1) // 2]
            )
        errors["requested"] = max(
            errors["requested"], float(np.max(abs(requested - raw["requested_delta"])))
        )
        expected = np.array(
            [
                V.native_units(z, n)
                for z, n in zip(raw["logits"], raw["normalizer"], strict=True)
            ]
        )
        errors["readout"] = max(
            errors["readout"], float(np.max(abs(expected - raw["readings"])))
        )
        if item["arm"] == "noop":
            errors["noop"] = max(
                errors["noop"], float(np.max(abs(raw["readings"] - base["readings"])))
            )
            continue
        if item["arm"] == "baseline":
            short = P.raw_unit(src, row["id"])
            for name, old in (("k", short["k"]), ("v", short["v"])):
                short_errors[name] = max(
                    short_errors[name],
                    float(np.max(abs(raw["k" if name == "k" else "before"] - old))),
                )
        index = (
            M.ARMS.index(item["arm"]),
            row["wording"],
            row["family"],
            (row["condition"] + 1) // 2,
            (row["polarity"] + 1) // 2,
            (row["owner"] + 1) // 2,
        )
        native[index] = raw["readings"].reshape(2, 6, 4)
    if not np.isfinite(native).all() or errors["edit"] > m["spec"]["edit_tolerance"]:
        raise ValueError("Incomplete support or unrealized edit")
    if (
        max(errors[k] for k in ("key", "before", "noop"))
        > m["spec"]["same_shape_tolerance"]
        or max(errors[k] for k in ("readout", "requested"))
        > m["spec"]["arithmetic_tolerance"]
    ):
        raise ValueError("Same-shape isolation/no-op/readout gate failed")
    return native, {**errors, "short_prefix_descriptive_not_gate": short_errors}


def result_tables(native, accounts, spec):
    """Return every planned signed target/effect/author comparison, not winners."""
    target, author, summary = [], [], []
    for a, arm in enumerate(M.ARMS):
        for w in range(2):
            for f, family in enumerate(spec["targets"]):
                own = M.author_difference(native[a, w, f], spec["author_amplitude"])
                for c, cs in enumerate((-1, 1)):
                    for q, name in enumerate(spec["targets"]):
                        for u, unit in enumerate(UNITS):
                            target.append(
                                {
                                    "arm": arm,
                                    "wording": w,
                                    "family": family,
                                    "condition": cs,
                                    "query": name,
                                    "unit": unit,
                                    "actor_differential_bo": float(
                                        accounts["actor_difference"][a, w, f, 6, q, u]
                                        + cs
                                        * accounts["actor_difference"][a, w, f, 7, q, u]
                                    ),
                                    "actor_common_bo": float(
                                        accounts["actor_common"][a, w, f, 6, q, u]
                                        + cs
                                        * accounts["actor_common"][a, w, f, 7, q, u]
                                    ),
                                }
                            )
                for p, person in enumerate(("A", "B")):
                    for q, name in enumerate(spec["targets"]):
                        for u, unit in enumerate(UNITS):
                            author.append(
                                {
                                    "arm": arm,
                                    "wording": w,
                                    "family": family,
                                    "person": person,
                                    "query": name,
                                    "unit": unit,
                                    "positive_law_difference": float(own[p, q, u]),
                                }
                            )
            for u, unit in enumerate(UNITS):
                x = accounts["actor_difference"][a, w, :, 6, :, u]
                g = accounts["actor_common"][a, w, :, 6, :, u]
                baseline = (
                    accounts["native"][0, w, :, 6, 0, :, u]
                    - accounts["native"][0, w, :, 6, 1, :, u]
                ) / 2
                den = float(np.sum(baseline**2))
                diag = float(np.trace(x) / 6)
                off = float(x[~np.eye(6, dtype=bool)].mean())
                summary.append(
                    {
                        "arm": arm,
                        "wording": w,
                        "unit": unit,
                        "actor_difference_RMS": float(np.sqrt(np.mean(x**2))),
                        "actor_common_RMS": float(np.sqrt(np.mean(g**2))),
                        "diagonal_mean": diag,
                        "off_target_mean": off,
                        "diagonal_minus_off": diag - off,
                        "baseline_actor_RMS": float(np.sqrt(np.mean(baseline**2))),
                        "baseline_signed_alignment": float(np.sum(x * baseline) / den)
                        if den
                        else None,
                    }
                )
    return target, author, summary


def analyze(out):
    """Separate actor-common from selective effects and positive-law integrals."""
    m = check(out)
    if H.load(out / "status.json")["state"] != "COLLECTED":
        raise ValueError("Full finite support required; no partial promotion")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    native, errors = assemble(out)
    accounts = M.effect_accounts(native)
    folder = out / "02"
    folder.mkdir()
    np.savez_compressed(folder / "accounts.npz", raw=native, **accounts)
    target, author, summaries = result_tables(native, accounts, m["spec"])
    H.table(folder / "target_effects.csv", target)
    H.table(folder / "author_accounts.csv", author)
    H.table(folder / "action_summaries.csv", summaries)
    H.write(
        folder / "summary.json",
        {
            "packet": "HM116",
            "status": H.load(out / "status.json"),
            "checks": errors,
            "summaries": summaries,
            "target_rows": len(target),
            "author_rows": len(author),
            "source_only": True,
            "new_fits": 0,
            "human_personality_accuracy": False,
            "neural_runtime_loaded_in_analysis": any(
                n in sys.modules for n in ("torch", "transformers")
            ),
            "original_scope_complete": False,
        },
    )
    H.write(folder / "output_seal.json", H.seals(out, folder))
    print(
        "ANALYZED HM116",
        errors,
        "all target/author rows",
        len(target),
        len(author),
        flush=True,
    )


def verify(out):
    """Replay raw units and saved tables with independent signed-cell integrals."""
    m = check(out)
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    native, checks = assemble(out)
    with np.load(out / "02/accounts.npz", allow_pickle=False) as z:
        saved = {k: z[k] for k in z.files}
    errors = {
        "raw": float(np.max(abs(native - saved["raw"]))),
        "character": 0.0,
        "all_characters": 0.0,
        "author": 0.0,
        "target_csv": 0.0,
        "author_csv": 0.0,
    }
    for a in range(5):
        for w in range(2):
            for f in range(6):
                for mask in range(8):
                    direct_char = sum(
                        np.prod(
                            [
                                s
                                for bit, s in enumerate(
                                    (2 * c - 1, 2 * b - 1, 2 * o - 1)
                                )
                                if mask & (1 << bit)
                            ]
                        )
                        * native[a, w, f, c, b, o]
                        / 8
                        for c in range(2)
                        for b in range(2)
                        for o in range(2)
                    )
                    errors["all_characters"] = max(
                        errors["all_characters"],
                        float(
                            np.max(abs(direct_char - saved["native"][a, w, f, mask]))
                        ),
                    )
                for c, cs in enumerate((-1, 1)):
                    direct = sum(
                        (2 * b - 1)
                        * (2 * o - 1)
                        * (native[0, w, f, c, b, o] - native[a, w, f, c, b, o])
                        / 4
                        for b in range(2)
                        for o in range(2)
                    )
                    effect = (
                        saved["effects"][a, w, f, 6] + cs * saved["effects"][a, w, f, 7]
                    )
                    errors["character"] = max(
                        errors["character"], float(np.max(abs(direct - effect)))
                    )
                direct_author = sum(
                    m["spec"]["author_amplitude"]
                    * (2 * b - 1)
                    * (2 * o - 1)
                    * native[a, w, f, c, b, o]
                    / 24
                    for c in range(2)
                    for b in range(2)
                    for o in range(2)
                )
                compiled = (2 * m["spec"]["author_amplitude"] / 6) * saved["native"][
                    a, w, f, 6
                ]
                errors["author"] = max(
                    errors["author"], float(np.max(abs(direct_author - compiled)))
                )
    target, author, summaries = result_tables(
        native, M.effect_accounts(native), m["spec"]
    )
    for name, expected, key in (
        ("target_effects", target, "target_csv"),
        ("author_accounts", author, "author_csv"),
        ("action_summaries", summaries, "summary_csv"),
    ):
        with (out / f"02/{name}.csv").open(newline="") as stream:
            actual = list(csv.DictReader(stream))
        if len(actual) != len(expected):
            raise ValueError("Missing signed output rows")
        errors[key] = 0.0
        for row, ref in zip(actual, expected, strict=True):
            for k, v in ref.items():
                if isinstance(v, float):
                    errors[key] = max(errors[key], abs(float(row[k]) - v))
                elif row[k] != ("" if v is None else str(v)):
                    raise ValueError("Output address mismatch")
    if (
        max(errors.values()) > m["spec"]["arithmetic_tolerance"]
        or H.load(out / "02/summary.json")["neural_runtime_loaded_in_analysis"]
    ):
        raise ValueError("Numerical replay failed or analysis loaded model")
    H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM116_NOT_GLOBAL",
            "errors": errors,
            "checks": checks,
            "batches": m["spec"]["max_batches"],
            "endpoints": m["spec"]["max_endpoints"],
            "same_agent_not_third_party": True,
            "original_scope_complete": False,
        },
    )
    print("VERIFIED HM116", errors, flush=True)


def main():
    """Expose finite stages; no implicit queue, parameter search or restart."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / H.load(SPEC)["output"])
    args = parser.parse_args()
    globals()[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
