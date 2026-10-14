#!/usr/bin/env python3
"""One finite GRAIN/native complement experiment, not a predictor search."""

from __future__ import annotations

import argparse
import collections
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_grain_native_remainder as M
from grain_research.construction import math_native_joint_batch as B
from grain_research.construction import math_native_joint_value as V
from grain_research.construction.math_construct_routing import query
from grain_research.vector_mechanisms.navigation import write_json
from scripts import run_grain_action_expression_hm115 as P
from scripts import run_grain_author_evidence_merge_hm102 as H

SPEC = ROOT / "configs/grain_native_remainder_hm124.json"
CODE = (
    "configs/grain_native_remainder_hm124.json",
    "docs/GRAIN_NATIVE_REMAINDER_PLAN_HM124.md",
    "grain_research/construction/math_grain_native_remainder.py",
    "scripts/run_grain_native_remainder_hm124.py",
    "scripts/readback_grain_native_remainder_hm124.py",
    "tests/test_grain_native_remainder.py",
    "grain_research/construction/math_native_joint_batch.py",
    "grain_research/construction/math_native_joint_value.py",
    "grain_research/construction/math_construct_routing.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
    "scripts/run_grain_author_evidence_merge_hm102.py",
)


def save(path, **arrays):
    """Write a new numeric unit without replacing existing evidence."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def check(out):
    """Authenticate the protocol, source inputs and cold application material."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out):
    """Seal source geometry and all new text before any native application call."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL

    if out.exists():
        raise FileExistsError("Do not replace earlier or failed experiments")
    spec = H.load(SPEC)
    src, mark = ROOT / spec["source"], ROOT / spec["marks_source"]
    source_manifest = P.check(src)
    H.verify_hashes(src, H.load(src / "02/output_seal.json"))
    old_rows = H.load(src / "00/rows.json")
    source_rows = [r for r in old_rows if r["variant"] == 0]
    if [r["id"] for r in source_rows] != list(range(96)):
        raise ValueError("Source array/row ordering changed")
    with np.load(src / "02/fields.npz", allow_pickle=False) as z:
        values = z["v"][0].reshape(96, -1)
    with np.load(mark / "00/marks.npz", allow_pickle=False) as z:
        coordinates = z["slice_scores"][:96].astype(float)
    decomposition = M.source_decomposition(coordinates, values, source_rows)
    shape = tuple(spec["physical_shape"])
    if np.prod(shape) != values.shape[1] or spec["arms"] != list(M.ARMS):
        raise ValueError("Physical source units or fixed arms changed")
    actions = {n: decomposition[n].reshape(6, 2, *shape) for n in M.ARMS[1:4]}
    actions["random_matched"] = M.random_control(
        decomposition["remainder"], shape, spec["seed"]
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    system = ROOT / "configs/grain_math_passive_event_composition_v1.json"
    marker = "__HM124_BODY__"
    template = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": H.load(system)["system"]},
            {"role": "user", "content": marker},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix, closing = template.split(marker)
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    current = encode(spec["current"])
    if (
        len(current) != shape[0]
        or spec["current"]
        != H.load(ROOT / "configs/grain_action_expression_hm115.json")["current"]
    ):
        raise ValueError("Current carrier tokens must match source")
    rows = []
    for w, f, b, o in itertools.product(range(2), range(6), (-1, 1), (-1, 1)):
        actor, other = ("A", "B") if o == 1 else ("B", "A")
        text = (
            spec["conditions"][w]
            + "\nDuring this activity, Person "
            + actor
            + " "
            + spec["behaviours"][w][f][(b + 1) // 2]
            + ".\nPerson "
            + other
            + " noted the time on a plain record sheet.\n"
        )
        rows.append(
            dict(
                id=len(rows),
                wording=w,
                family=f,
                polarity=b,
                owner=o,
                condition=-1,
                text=text,
                body_ids=encode(text),
            )
        )
    width, prefix_ids = max(len(r["body_ids"]) for r in rows), encode(prefix)
    bags = {}
    for row in rows:
        actual = len(row.pop("body_ids"))
        body = encode(row["text"])
        row["input_ids"] = (
            prefix_ids + body + [tokenizer.pad_token_id] * (width - actual) + current
        )
        row["current"] = list(
            range(len(prefix_ids) + width, len(prefix_ids) + width + len(current))
        )
        row["padding"] = list(range(len(prefix_ids) + actual, len(prefix_ids) + width))
        bag = collections.Counter(body)
        key = row["wording"], row["family"], row["polarity"]
        if key in bags and bags[key] != bag:
            raise ValueError("Actor exchange changed the visible token multiset")
        bags[key] = bag
    digits = [encode(str(i)) for i in range(10)]
    if any(len(d) != 1 for d in digits) or len({d[0] for d in digits}) != 10:
        raise ValueError("Native digit alphabet must consist of ten distinct tokens")
    questions = [
        query(t).replace("the writer's", f"Person {p}'s")
        for p in ("A", "B")
        for t in spec["targets"]
    ]
    queries = dict(
        questions=questions,
        suffixes=[
            encode("\n\nEvidence ends here.\n\nQuestion: " + q + "\nAnswer:" + closing)
            for q in questions
        ],
        alphabet=dict(
            labels=[str(i) for i in range(10)],
            values=(np.arange(10) / 9).tolist(),
            token_ids=[d[0] for d in digits],
        ),
    )
    design = [
        dict(batch=len(M.ARMS) * r["id"] + a, row=r["id"], arm=arm)
        for r in rows
        for a, arm in enumerate(M.ARMS)
    ]
    if len(rows) != spec["new_records"] or len(design) != spec["max_batches"]:
        raise ValueError("Incomplete finite design")
    for folder in ("00", "01/units", "02"):
        (out / folder).mkdir(parents=True)
    for name, data in (
        ("rows", rows),
        ("source_rows", source_rows),
        ("design", design),
        ("queries", queries),
    ):
        H.write(out / f"00/{name}.json", data)
    save(
        out / "00/decomposition.npz",
        coordinates=coordinates,
        values=values,
        **decomposition,
    )
    save(out / "00/actions.npz", **actions)
    names = [
        src / n
        for n in (
            "00/manifest.json",
            "00/rows.json",
            "02/fields.npz",
            "02/output_seal.json",
            "verification.json",
        )
    ]
    names += [mark / "00/marks.npz", mark / "00/manifest.json", system]
    H.write(
        out / "00/manifest.json",
        dict(
            spec=spec,
            model=source_manifest["model"],
            code={n: H.sha(ROOT / n) for n in CODE},
            source={str(p.relative_to(ROOT)): H.sha(p) for p in names},
            inputs=H.seals(out, out / "00"),
            source_uses_target_answers=False,
            application_uses_design_metadata_for_action=True,
            source_sample_projection_not_feature_erasure=True,
            new_application_material=True,
            original_scope_complete=False,
        ),
    )
    H.write(
        out / "status.json",
        dict(
            state="PREPARED", completed=0, attempts=0, pending=None, model_seconds=0.0
        ),
    )
    print(
        "HM124 PREPARED",
        len(rows),
        "new texts",
        len(design),
        "native batches",
        flush=True,
    )


def raw(out, index):
    """Return only an authenticated fully committed native batch."""
    stem = out / f"01/units/b{index:03d}"
    metadata = H.load(stem.with_suffix(".json"))
    if H.sha(stem.with_suffix(".npz")) != metadata["sha256"]:
        raise ValueError("Saved native batch changed")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
        return {n: z[n] for n in z.files}


def collect(out):
    """Run the registered finite batch with one cumulative, nonrenewable cap."""
    from grain_research.vector_mechanisms.navigation import MODEL
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    manifest, state = check(out), H.load(out / "status.json")
    if state["state"] != "PREPARED" or state["pending"] is not None:
        raise RuntimeError("No restarts of terminal or uncertain collections")
    H.verify_hashes(MODEL, manifest["model"])
    spec = manifest["spec"]
    rows, design, queries = [
        H.load(out / f"00/{n}.json") for n in ("rows", "design", "queries")
    ]
    with np.load(out / "00/actions.npz", allow_pickle=False) as z:
        actions = {n: z[n] for n in z.files}
    try:
        with collector_guard(out):
            begun = time.monotonic()
            try:
                observer = load_observer()
            finally:
                state["model_seconds"] += time.monotonic() - begun
                write_json(out / "status.json", state)
            H.write(out / "model_manifest.json", observer.manifest())
            for item in design:
                if state["model_seconds"] >= spec["max_model_seconds"]:
                    state["state"] = "BUDGET_STOP"
                    break
                row, arm = rows[item["row"]], item["arm"]
                delta = (
                    None
                    if arm == "baseline"
                    else -row["polarity"]
                    * row["owner"]
                    * actions[arm][row["family"], 0]
                )
                state.update(
                    state="RUNNING", pending=item, attempts=state["attempts"] + 1
                )
                write_json(out / "status.json", state)
                begun = time.monotonic()
                try:
                    readings, arrays = B.native_batch(
                        observer,
                        row,
                        queries["suffixes"],
                        queries["alphabet"],
                        delta,
                        spec["layer"],
                    )
                finally:
                    state["model_seconds"] += time.monotonic() - begun
                    write_json(out / "status.json", state)
                logits = np.array([r["candidate_logits"] for r in readings])
                normalizer = np.array(
                    [r["full_vocab_log_normalizer"] for r in readings]
                )
                arrays.update(
                    logits=logits,
                    normalizer=normalizer,
                    readings=np.array(
                        [
                            V.native_units(z, n)
                            for z, n in zip(logits, normalizer, strict=True)
                        ]
                    ),
                    requested_delta=np.zeros(spec["physical_shape"])
                    if delta is None
                    else delta,
                )
                stem = out / f"01/units/b{item['batch']:03d}"
                save(stem.with_suffix(".npz"), **arrays)
                H.write(
                    stem.with_suffix(".json"),
                    dict(**item, sha256=H.sha(stem.with_suffix(".npz"))),
                )
                state.update(completed=state["completed"] + 1, pending=None)
                write_json(out / "status.json", state)
                if state["completed"] % 20 == 0:
                    print("HM124", state, flush=True)
            if state["completed"] == len(design):
                state["state"] = (
                    "COLLECTED"
                    if state["model_seconds"] <= spec["max_model_seconds"]
                    else "BUDGET_STOP"
                )
            write_json(out / "status.json", state)
            H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    except BaseException as exc:
        state.update(state="FAILED", error=f"{type(exc).__name__}: {exc}")
        write_json(out / "status.json", state)
        raise
    print("HM124 TERMINAL", state, flush=True)


def analyze(out):
    """Compare actual native effects without fitting or promoting partial data."""
    spec, state = check(out)["spec"], H.load(out / "status.json")
    if (
        state["state"] not in ("COLLECTED", "BUDGET_STOP")
        or state["pending"] is not None
    ):
        raise ValueError("Analyze only a safely terminal collection")
    rows, design = H.load(out / "00/rows.json"), H.load(out / "00/design.json")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    if state["completed"] != len(design):
        H.write(
            out / "02/coverage.json",
            dict(
                completed_batches=state["completed"],
                planned=len(design),
                scientific_completion=False,
            ),
        )
        print("HM124 PARTIAL; no complete-field assertion", flush=True)
        return
    n = np.zeros((5, 2, 6, 2, 2, 2, 6, 4))
    errors = {k: 0.0 for k in ("edit", "key", "before", "replay", "action_sum")}
    for item in design:
        row, arm = rows[item["row"]], M.ARMS.index(item["arm"])
        x, base = raw(out, item["batch"]), raw(out, len(M.ARMS) * row["id"])
        for name in (
            "support_hash",
            "execution_support_hash",
            "masked_blocks",
            "execution_length",
            "endpoint",
        ):
            if not np.array_equal(x[name], base[name]):
                raise ValueError("Intervention changed native execution support")
        errors["key"] = max(errors["key"], float(np.max(abs(x["k"] - base["k"]))))
        errors["before"] = max(
            errors["before"], float(np.max(abs(x["before"] - base["before"])))
        )
        errors["edit"] = max(
            errors["edit"],
            float(np.max(abs(x["after"] - x["before"] - x["requested_delta"]))),
        )
        replay = np.array(
            [
                V.native_units(l, z)
                for l, z in zip(x["logits"], x["normalizer"], strict=True)
            ]
        )
        errors["replay"] = max(
            errors["replay"], float(np.max(abs(replay - x["readings"])))
        )
        n[
            arm,
            row["wording"],
            row["family"],
            (row["polarity"] + 1) // 2,
            (row["owner"] + 1) // 2,
        ] = x["readings"].reshape(2, 6, 4)
    with np.load(out / "00/decomposition.npz", allow_pickle=False) as z:
        orthogonality = float(np.max(abs(z["design"].T @ z["remainder_values"])))
        reconstruction = float(
            np.max(abs(z["represented"] + z["remainder_values"] - z["values"]))
        )
        errors["action_sum"] = float(
            np.max(abs(z["grain_part"] + z["remainder"] - z["whole"]))
        )
        source = dict(
            rank=int(z["rank"]),
            source_records=96,
            coordinate_columns=13,
            orthogonality_max=orthogonality,
            reconstruction_max=reconstruction,
            whole_norm=float(np.linalg.norm(z["whole"])),
            grain_part_norm=float(np.linalg.norm(z["grain_part"])),
            remainder_norm=float(np.linalg.norm(z["remainder"])),
        )
    if (
        max(errors["edit"], errors["key"], errors["before"]) > 5e-5
        or max(orthogonality, reconstruction, errors["action_sum"]) > 1e-8
    ):
        H.write(
            out / "02/failed_numeric_checks.json", dict(errors=errors, source=source)
        )
        raise ValueError("Native support/edit or source identity failed")
    fields = M.actor_focus_field(n)
    save(out / "02/native_fields.npz", native=n, **fields)
    summaries = M.response_summary(fields)
    H.table(out / "02/effects.csv", summaries)
    cells = [
        dict(
            arm=M.ARMS[a],
            wording=w,
            family=f,
            target=q,
            unit=u,
            baseline=float(fields["baseline"][w, f, q, u]),
            effect=float(fields["effects"][a, w, f, q, u]),
        )
        for a, w, f, q, u in itertools.product(
            range(5), range(2), range(6), range(6), range(4)
        )
    ]
    H.table(out / "02/all_target_effects.csv", cells)
    H.write(
        out / "02/summary.json",
        dict(
            status="FINITE_NATIVE_GRAIN_REMAINDER_ACCOUNT_COMPLETE_NOT_FULL",
            source=source,
            numeric_checks=errors,
            effects=summaries,
            interaction_rms_by_wording_and_unit=np.sqrt(
                np.mean(fields["interaction"] ** 2, axis=(1, 2))
            ).tolist(),
            native_digit_mass_range=[float(n[..., 2].min()), float(n[..., 2].max())],
            actual_new_records=48,
            actual_native_batches=240,
            actual_native_endpoints=2880,
            model_seconds=state["model_seconds"],
            original_scope_complete=False,
            source_split_guarantees_linear_sample_orthogonality_only=True,
            no_unknown_natural_text_personality_inference_claim=True,
        ),
    )
    H.write(out / "02/seal.json", H.seals(out, out / "02"))
    print("HM124 ANALYZED native effects, no prediction or accuracy fit", flush=True)


def main():
    """Expose separate sealed preparation, native collection and offline analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "collect", "analyze"))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/grain_native_remainder_r1",
    )
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze}[args.phase](
        args.out.resolve()
    )


if __name__ == "__main__":
    main()
