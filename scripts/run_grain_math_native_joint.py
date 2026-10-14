#!/usr/bin/env python3
"""HM10B1: actual narrow-query responses and categorical-link geometry."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_native_joint as N
from grain_research.vector_mechanisms.navigation import (
    MODEL,
    digest,
    load_functions,
    read_json,
)
from grain_research.vector_mechanisms.navigation import (
    write_json as atomic_json,
)
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_narrow_functions import reference_view
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_native_joint_v1.json"
PLAN = ROOT / "docs/GRAIN_NATIVE_JOINT_RESPONSE_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/native_joint_r3"
CODE = (
    "scripts/run_grain_math_native_joint.py",
    "grain_research/construction/math_native_joint.py",
    "grain_research/vector_mechanisms/full_state_edit.py",
    "grain_research/vector_mechanisms/functional_profiles.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_narrow_functions.py",
)


def write_json(path, value):
    """Create packet subdirectories before the shared atomic JSON writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def frozen_files(out):
    """Validate this packet's code, input addresses and source bytes on resume."""
    manifest = read_json(out / "00/manifest.json")
    for name, expected in manifest["inputs"].items():
        if digest(out / name) != expected:
            raise ValueError(f"changed input: {name}")
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"changed source: {name}")
    if digest(MODEL / "model.safetensors") != manifest["model_sha256"]:
        raise ValueError("observer model changed")
    return manifest


def prepare(out):
    """Reconcile all 25 old functions; freeze three readers before new forwards."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("use a fresh packet; existing prepared packets resume")
    spec = read_json(SPEC)
    source, reference = ROOT / spec["source_packet"], ROOT / spec["reference"]
    hashes = read_json(source / "04/artifact_hashes.json")
    for name, expected in hashes.items():
        if digest(source / name) != expected:
            raise ValueError("HM10A changed")
    definitions = read_json(reference / "queries.json")
    old = read_json(reference / "manifest.json")
    refmanifest = read_json(reference / "reference_manifest.json")
    if (
        len(definitions) != spec["query_count_reconciled"]
        or digest(reference / "reference.npz") != refmanifest["sha256"]
    ):
        raise ValueError("narrow function registry or coefficients changed")
    if old["meps_reference_authors"] != 0 or old["human_outcome_fit"]:
        raise ValueError("this packet does not permit private outcome construction")
    if digest(MODEL / "model.safetensors") != old["model_sha256"]:
        raise ValueError("fixed observer changed")
    documents = [
        json.loads(line)
        for line in (source / "00/documents.jsonl").read_text().splitlines()
    ]
    by_text = {}
    for r in documents:
        by_text.setdefault(r["text"], len(by_text))
        r["text_id"] = by_text[r["text"]]
    if (
        len(documents) != spec["document_records"]
        or len(by_text) != spec["unique_texts"]
    ):
        raise ValueError("unexpected reused document coverage")
    functions = load_functions(reference / "reference.npz", list(definitions))
    ledger = []
    with np.load(reference / "reference.npz", allow_pickle=False) as arrays:
        for q, definition in definitions.items():
            fn = functions[q]
            ledger.append(
                {
                    "query": q,
                    "family": definition.get("family", definition.get("items_basis")),
                    "kind": definition["kind"],
                    "question": definition["question"],
                    "selected": q in spec["queries"],
                    "question_tokens": len(old["question_tokens"][q]),
                    "hidden_dimension": len(fn.q_ref),
                    "source_authors": 2,
                    "anchor": float(arrays[f"{q}__b"]),
                    "reference_unit": "block16 question-token P and final Q, normalized conditional digit expectation",
                    "private_text_or_outcome_read": False,
                    "new_external_replication": False,
                    "content_support": "expressed anxiety"
                    if q == "adapted_na8"
                    else "adjacent dejection content, not a declared low/high label"
                    if q == "adapted_dejected5"
                    else "no direct AI trust evidence"
                    if q == "pre_trust_positive"
                    else "not tested in this packet",
                }
            )
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    design = []
    for text, tid in by_text.items():
        for q in spec["queries"]:
            _, _, encoded, mask = reference_view(
                tokenizer, [text], definitions[q]["question"]
            )
            if (
                np.asarray(encoded["input_ids"])[mask].tolist()
                != old["question_tokens"][q]
            ):
                raise ValueError("query coordinates differ from frozen source")
            design.append(
                {
                    "unit": len(design),
                    "text_id": tid,
                    "query": q,
                    "input_ids": encoded["input_ids"],
                    "mask": mask.tolist(),
                }
            )
    if len(design) != spec["forward_calls"]:
        raise ValueError("new observation count changed")
    (out / "01/units").mkdir(parents=True)
    write_json(out / "00/documents.json", documents)
    write_json(out / "00/design.json", design)
    write_json(out / "00/query_ledger.json", ledger)
    write_csv(out / "00/query_ledger.csv", ledger)
    sources = {
        name: digest(ROOT / name)
        for name in (
            *CODE,
            str(SPEC.relative_to(ROOT)),
            str(PLAN.relative_to(ROOT)),
            str((source / "04/artifact_hashes.json").relative_to(ROOT)),
            str((source / "00/documents.jsonl").relative_to(ROOT)),
            str((reference / "queries.json").relative_to(ROOT)),
            str((reference / "manifest.json").relative_to(ROOT)),
            str((reference / "reference_manifest.json").relative_to(ROOT)),
            str((reference / "reference.npz").relative_to(ROOT)),
        )
    }
    for name in CODE:
        target = out / "00/code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "packet": spec["packet"],
            "spec": spec,
            "model_sha256": old["model_sha256"],
            "alphabet": old["alphabet"],
            "question_tokens": {q: old["question_tokens"][q] for q in spec["queries"]},
            "inputs": {
                name: digest(out / name)
                for name in (
                    "00/documents.json",
                    "00/design.json",
                    "00/query_ledger.json",
                    "00/query_ledger.csv",
                )
            },
            "source_files": sources,
            "hm10a_hashes_checked": len(hashes),
            "frozen_before_new_calls": True,
            "new_texts": 0,
            "new_observer_experiment_on_old_texts": True,
            "new_neural_training": False,
            "api_calls": 0,
            "private_meps_reads": 0,
            "individual_parameters_fitted": False,
        },
    )
    write_json(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "units": 0, "seconds": 0},
    )
    print(
        f"HM10B1 prepared: 25 reconciled functions, {len(design)} actual new forwards",
        flush=True,
    )


def collect(out, seconds):
    """Capture one atomic native/projection unit per unique text and query."""
    import torch

    from grain_research.vector_mechanisms.full_state_edit import capture_boundary
    from grain_research.vector_mechanisms.trace import DecoderObserver

    m = frozen_files(out)
    previous = read_json(out / "run_status.json")
    if previous["status"] == "COMPLETE":
        print(previous, flush=True)
        return
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    reference = ROOT / m["spec"]["reference"]
    functions = load_functions(reference / "reference.npz", m["spec"]["queries"])
    start, complete = time.monotonic(), 0
    alphabet = m["alphabet"]
    head = (
        observer.model.lm_head.weight[alphabet["token_ids"]]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    if (out / "01/head.npz").exists():
        with np.load(out / "01/head.npz", allow_pickle=False) as a:
            if not np.array_equal(a["head"], head):
                raise ValueError("saved output head changed")
    else:
        np.savez_compressed(out / "01/head.npz", head=head)
        write_json(
            out / "01/head.json", {"sha256": digest(out / "01/head.npz"), "bias": False}
        )
    write_json(out / "01/observer.json", observer.manifest())

    def status(label):
        s = {
            "status": label,
            "units": complete,
            "calls": previous["calls"] + observer.calls,
            "seconds": previous["seconds"] + time.monotonic() - start,
        }
        write_json(out / "run_status.json", s)
        return s

    try:
        for d in read_json(out / "00/design.json"):
            path = out / "01/units" / f"unit_{d['unit']:04d}.json"
            if path.exists():
                saved = read_json(path)
                if (
                    saved["unit"] != d["unit"]
                    or saved["query"] != d["query"]
                    or saved["array_sha256"] != digest(path.with_suffix(".npz"))
                ):
                    raise ValueError("saved unit changed")
            else:
                if time.monotonic() - start >= seconds:
                    print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
                    return
                final = {}

                def norm_hook(_module, _args, output, captured=final):
                    captured["state"] = output[0, -1].detach().float().cpu().numpy()

                handle = observer.model.model.norm.register_forward_hook(norm_hook)
                try:
                    state, _, native = capture_boundary(
                        observer, d["input_ids"], alphabet
                    )
                finally:
                    handle.remove()
                p = (
                    state[0, np.asarray(d["mask"], bool)]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
                q = state[0, -1].float().cpu().numpy().astype(float)
                fn = functions[d["query"]]
                parts = fn.parts(p, q)
                delta = p - fn.p_ref
                mean = delta.mean(axis=0)
                config_terms = np.sum(fn.config_coef * (delta - mean), axis=1)
                direct = (
                    fn.anchor
                    + np.sum((fn.config_coef + fn.mean_coef / len(p)) * delta)
                    + fn.g_final @ (q - fn.q_ref)
                )
                np.savez_compressed(
                    path.with_suffix(".npz"),
                    mean_delta=mean,
                    config_token_terms=config_terms,
                    Q=q,
                    final_normalized=final["state"],
                    logits=np.asarray(native["candidate_logits"]),
                    probabilities=np.asarray(native["conditional_probabilities"]),
                )
                write_json(
                    path,
                    {
                        "unit": d["unit"],
                        "text_id": d["text_id"],
                        "query": d["query"],
                        "native": native,
                        "affine": parts,
                        "affine_direct_error": abs(float(direct) - parts["score"]),
                        "array_sha256": digest(path.with_suffix(".npz")),
                        "stored_P": "mean delta and exact target-weighted centered token terms, not full P",
                    },
                )
                del state, p, q
            complete += 1
            if complete % 128 == 0:
                print(status("CAPTURING"), flush=True)
        print(status("COMPLETE"), flush=True)
    except Exception as exc:
        status("ERROR")
        write_json(
            out / "error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise


def evaluate(out):
    """Compute interactions, source-only completions and all declared controls."""
    m = frozen_files(out)
    if read_json(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("native observations incomplete")
    values = m["alphabet"]["values"]
    docs = read_json(out / "00/documents.json")
    collected = {}
    max_head_error = 0.0
    with np.load(out / "01/head.npz", allow_pickle=False) as a:
        head = a["head"].astype(float)
    for d in read_json(out / "00/design.json"):
        path = out / "01/units" / f"unit_{d['unit']:04d}.json"
        unit = read_json(path)
        if unit["array_sha256"] != digest(path.with_suffix(".npz")):
            raise ValueError("native unit changed")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
            logits = a["logits"].copy()
            max_head_error = max(
                max_head_error,
                float(np.max(abs(head @ a["final_normalized"].astype(float) - logits))),
            )
        collected[d["text_id"], d["query"]] = (unit, logits)
    scalar_rows, cubes = [], defaultdict(dict)
    for doc in docs:
        for query in m["spec"]["queries"]:
            meta, logits = collected[doc["text_id"], query]
            base = {
                k: doc[k]
                for k in (
                    "material_id",
                    "cube_id",
                    "family",
                    "role",
                    "frame",
                    "actor",
                    "mask",
                    "text_id",
                )
            }
            scalar_rows.append(
                base
                | {
                    "query": query,
                    "native": meta["native"]["conditional_score"],
                    "candidate_mass": meta["native"]["candidate_mass"],
                    "affine": meta["affine"]["score"],
                    **{
                        k: meta["affine"][k] for k in ("mean", "configuration", "final")
                    },
                }
            )
            cubes[doc["cube_id"], query][doc["mask"]] = (doc, meta, logits)
    effects, content_pairs = [], []
    max_closure = max_integral = max_kl = 0.0
    for (cube_id, query), cells in cubes.items():
        if set(cells) != set(range(8)):
            raise ValueError("incomplete three-coordinate cube")
        for incidence in (0, 1):
            sub = [cells[mask + 4 * incidence] for mask in range(4)]
            logits = np.array([t[2] for t in sub])
            affine = np.array([t[1]["affine"]["score"] for t in sub])
            d = N.interaction(logits, values)
            geometry = N.categorical_geometry(logits[3], d["additive_logits"])
            integral = N.response_integral(
                d["additive_logits"], d["logit_interaction"], values
            )
            max_closure = max(
                max_closure,
                abs(d["total"] - d["link_only"] - d["representation_action"]),
            )
            max_integral = max(max_integral, abs(integral - d["representation_action"]))
            max_kl = max(max_kl, abs(geometry["kl"] - geometry["bregman"]))
            first = sub[0][0]
            effects.append(
                {
                    "cube_id": cube_id,
                    "query": query,
                    **{k: first[k] for k in ("family", "role", "frame", "actor")},
                    "incidence": incidence,
                    **d,
                    **geometry,
                    "gradient_integral": integral,
                    "affine_interaction": float(
                        affine[3] - affine[1] - affine[2] + affine[0]
                    ),
                    "actual": float(N.score(logits[3], values)),
                    "scalar_additive": float(
                        N.score(logits[1], values)
                        + N.score(logits[2], values)
                        - N.score(logits[0], values)
                    ),
                }
            )
        for mask in range(4):
            left, right = cells[mask], cells[mask + 4]
            content_pairs.append(
                {
                    "cube_id": cube_id,
                    "query": query,
                    **{k: left[0][k] for k in ("family", "role", "frame", "actor")},
                    "mask": mask,
                    "identical_raw": left[0]["text"] == right[0]["text"],
                    "native_delta": right[1]["native"]["conditional_score"]
                    - left[1]["native"]["conditional_score"],
                    "affine_delta": right[1]["affine"]["score"]
                    - left[1]["affine"]["score"],
                }
            )
    predictions, summary_rows, ci_rows = [], [], []
    for qi, query in enumerate(m["spec"]["queries"]):
        selected = [r for r in effects if r["query"] == query]
        fit = N.fit_joint_source(selected)
        app_units = []
        for r in selected:
            if r["role"] == "source":
                continue
            key = (r["frame"], r["actor"], r["incidence"])
            wrong = ((r["frame"] + 1) % 4, r["actor"], 1 - r["incidence"])
            z = r["additive_logits"]
            arms = {
                "scalar_additive": r["scalar_additive"],
                "link_only": r["link_score"],
                "source_global_logit": float(N.score(z + fit["global_logits"], values)),
                "source_conditional_logit": float(
                    N.score(z + fit["conditional"][key]["logits"], values)
                ),
                "wrong_key_logit": float(
                    N.score(z + fit["conditional"][wrong]["logits"], values)
                ),
                "source_conditional_scalar": r["scalar_additive"]
                + fit["conditional"][key]["scalar"],
            }
            for arm, predicted in arms.items():
                predictions.append(
                    {
                        k: r[k]
                        for k in (
                            "cube_id",
                            "query",
                            "family",
                            "role",
                            "frame",
                            "actor",
                            "incidence",
                        )
                    }
                    | {
                        "arm": arm,
                        "prediction": predicted,
                        "actual": r["actual"],
                        "squared_error": (predicted - r["actual"]) ** 2,
                    }
                )
            if r["role"] == "application":
                app_units.append(
                    {
                        "family": r["family"],
                        "role": "application",
                        "key": key,
                        "baseline": z,
                        "actual": r["actual"],
                        "values": values,
                    }
                )
        ci = N.family_interval(
            selected,
            app_units,
            seed=m["spec"]["seed"] + qi,
            draws=m["spec"]["bootstrap_draws"],
        )
        ci_rows.append(
            {
                "query": query,
                "arm": "source_conditional_logit",
                "baseline": "link_only",
                "source_families": fit["source_families"],
                "application_families": len({r["family"] for r in app_units}),
                "draws": m["spec"]["bootstrap_draws"],
                "ci_low": ci[0],
                "ci_high": ci[1],
            }
        )
        for role in ("validation", "application"):
            for arm in m["spec"]["arms"]:
                rows = [
                    r
                    for r in predictions
                    if r["query"] == query and r["role"] == role and r["arm"] == arm
                ]
                baseline = [
                    r
                    for r in predictions
                    if r["query"] == query
                    and r["role"] == role
                    and r["arm"] == "link_only"
                ]
                summary_rows.append(
                    {
                        "query": query,
                        "role": role,
                        "arm": arm,
                        "units": len(rows),
                        "independent_families": len({r["family"] for r in rows}),
                        "rmse": float(
                            np.sqrt(np.mean([r["squared_error"] for r in rows]))
                        ),
                        "mse_gain_over_link": float(
                            np.mean([r["squared_error"] for r in baseline])
                            - np.mean([r["squared_error"] for r in rows])
                        ),
                    }
                )
    mechanism = []
    for query in m["spec"]["queries"]:
        rows = [r for r in effects if r["query"] == query]
        pairs = [
            r for r in content_pairs if r["query"] == query and not r["identical_raw"]
        ]
        native = np.array([r["native"] for r in scalar_rows if r["query"] == query])
        affine = np.array([r["affine"] for r in scalar_rows if r["query"] == query])
        mechanism.append(
            {
                "query": query,
                "total_rms": float(np.sqrt(np.mean([r["total"] ** 2 for r in rows]))),
                "link_rms": float(
                    np.sqrt(np.mean([r["link_only"] ** 2 for r in rows]))
                ),
                "representation_rms": float(
                    np.sqrt(np.mean([r["representation_action"] ** 2 for r in rows]))
                ),
                "logit_interaction_norm_mean": float(
                    np.mean([np.linalg.norm(r["logit_interaction"]) for r in rows])
                ),
                "mean_kl": float(np.mean([r["kl"] for r in rows])),
                "same_word_incidence_delta_rms": float(
                    np.sqrt(np.mean([r["native_delta"] ** 2 for r in pairs]))
                ),
                "affine_native_level_r": float(np.corrcoef(native, affine)[0, 1]),
                "affine_native_level_rmse": float(
                    np.sqrt(np.mean((native - affine) ** 2))
                ),
                "native_range": [float(native.min()), float(native.max())],
                "candidate_mass_min": min(
                    r["candidate_mass"] for r in scalar_rows if r["query"] == query
                ),
            }
        )
    outputs = {
        "02/readouts.csv": scalar_rows,
        "02/interactions.csv": [
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in r.items()}
            for r in effects
        ],
        "02/content_incidence_pairs.csv": content_pairs,
        "03/predictions.csv": predictions,
        "03/completion_summary.csv": summary_rows,
        "03/source_application_uncertainty.csv": ci_rows,
        "03/mechanism_summary.csv": mechanism,
    }
    for name, rows in outputs.items():
        write_csv(out / name, rows)
    summary = {
        "packet": "HM-10B1",
        "status": "FINITE_NATIVE_RESPONSE_COMPLETE",
        "documents": len(docs),
        "unique_native_units": len(collected),
        "queries_reconciled": 25,
        "queries_observed": 3,
        "effects": len(effects),
        "incidence_pairs": len(content_pairs),
        "identical_raw_null_pairs": sum(r["identical_raw"] for r in content_pairs),
        "max_head_reconstruction_error": max_head_error,
        "max_interaction_closure_error": max_closure,
        "max_integral_error": max_integral,
        "max_kl_bregman_error": max_kl,
        "mechanisms": mechanism,
        "application_completion": [
            r for r in summary_rows if r["role"] == "application"
        ],
        "source_application_uncertainty": ci_rows,
        "new_raw_texts": 0,
        "individual_response_fit": False,
        "query_truth_is_human_measurement": False,
        "absent_trust_evidence_is_not_a_zero_personality_score": True,
        "outputs": {name: digest(out / name) for name in outputs},
        "observer_calls": read_json(out / "run_status.json")["calls"],
    }
    write_json(out / "03/summary.json", summary)
    artifacts = {
        str(p.relative_to(out)): digest(p)
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name != "artifact_hashes.json"
    }
    write_json(out / "03/artifact_hashes.json", artifacts)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "mechanisms": mechanism,
                "uncertainty": ci_rows,
            }
        ),
        flush=True,
    )


def main():
    """Expose independent freeze, bounded resumable collect and cold evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "collect", "evaluate"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    if (
        not args.out.resolve().is_relative_to(
            (ROOT / "results/grain_math_response_v1").resolve()
        )
        or not 0 < args.seconds <= 1800
    ):
        parser.error("use a bounded new local research packet")
    with collector_guard(args.out):
        {
            "prepare": lambda: prepare(args.out),
            "collect": lambda: collect(args.out, args.seconds),
            "evaluate": lambda: evaluate(args.out),
        }[args.mode]()


if __name__ == "__main__":
    main()
