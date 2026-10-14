#!/usr/bin/env python3
"""One source construction and 384 new readings; no accuracy or rank search."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import local_measure as L
from grain_research.common_factors import WordFactorReference
from grain_research.construction import math_cold_joint_author as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SOURCE = ROOT / "results/grain_math_response_v1/role_assignment_dealiasing_r1"
OUT = ROOT / "results/grain_math_response_v1/cold_joint_author_r1"
CODE = (
    "docs/GRAIN_COLD_JOINT_AUTHOR_OBSERVATION_PLAN_HM81.md",
    "grain_research/construction/math_cold_joint_author.py",
    "grain_research/local_measure.py",
    "grain_research/common_factors.py",
    "scripts/run_grain_cold_joint_author_hm81.py",
    "scripts/score_grain_cold_joint_author.py",
    "tests/test_grain_math_cold_joint_author.py",
)
SCENES = (
    "During an evening project at a community hall",
    "While preparing a small exhibition in a shared room",
    "During a weekend activity at a neighbourhood centre",
    "While working on a group task in a public workshop",
    "During a morning session at a volunteer office",
    "While arranging a local event in a meeting room",
    "During an afternoon project at a library",
    "While finishing a shared task in a training room",
)


def fresh_materials(material):
    """Make sparse new author histories; never construct an application cube."""
    records = []
    for author, event in itertools.product(range(8), range(8)):
        focal = M.TARGETS[(event + author % 2) % 6]
        companion = M.TARGETS[(event + author % 2 + 1 + author % 3) % 6]
        positive = (event // 2 + author) % 2
        other_positive = positive if author % 4 < 2 else 1 - positive
        family = (event + author // 2) % 2
        actor = "A" if (event + author // 4) % 2 else "B"
        condition = "ordinary" if event % 2 == 0 else "deadline"
        pressure = (
            "the participants had plenty of time."
            if condition == "ordinary"
            else "an unexpected deadline left very little time."
        )
        body = f"{SCENES[event]}, {pressure}\n"
        body += f"The record for episode {event + 1} describes actual actions. "
        body += f"Person A {material['behaviours'][focal][family][positive]}. "
        body += f"In the same episode, Person {actor} {material['behaviours'][companion][family][other_positive]}. "
        body += "This is the end of the account; no other actions are reported."
        records.append(
            {
                "author_key": f"synthetic_{author:02d}",
                "event_id": f"episode_{event:02d}",
                "condition": condition,
                "qualification": "focal_report",
                "body": body,
            }
        )
    if len({r["body"] for r in records}) != 64:
        raise ValueError("64 unique sparse bodies required")
    return records


def prepare(out):
    """Freeze public source references and new materials before application reads."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("preserve all existing runs")
    verification = read_json(SOURCE / "verification.json")
    if verification["status"] != "VERIFIED_FINITE_HM79_NOT_GLOBAL":
        raise ValueError("verified HM79 source required")
    source_seal = read_json(SOURCE / "01/raw_seal.json")
    parent = ROOT / read_json(SOURCE / "00/manifest.json")["spec"]["parent"]
    prior_rows = {r["unit"]: r for r in read_json(parent / "00/rows.json")}
    prior_seal = read_json(parent / "01/raw_seal.json")
    # Only read score records and bodies from the declared source family.
    rows = [r for r in read_json(SOURCE / "00/rows.json") if r["family"] == 0]
    docs, scores, seen, inputs = {}, {}, set(), {}
    for row in rows:
        reused = row["old_unit"] is not None
        base, capture_unit = (
            (parent, row["old_unit"]) if reused else (SOURCE, row["new_unit"])
        )
        path = base / f"01/units/unit_{capture_unit:04d}.json"
        name = str(path.relative_to(base))
        sha = digest(path)
        seal = prior_seal if reused else source_seal
        expected = seal.get("files", seal).get(name)
        if sha != expected:
            raise ValueError("source score seal drift:" + name)
        meta = read_json(path)
        addressed = prior_rows[capture_unit] if reused else row
        if reused and any(
            row[k] != addressed[k]
            for k in ("family", "focal", "companion", "query", "body", "input_ids")
        ):
            raise ValueError("reused source observation alias mismatch")
        if any(meta[k] != addressed[k] for k in ("unit", "doc_id", "family", "query")):
            raise ValueError("source observation identity mismatch")
        doc, q = row["doc_id"], M.TARGETS.index(row["query"])
        if (doc, q) in seen:
            raise ValueError("duplicate source query")
        seen.add((doc, q))
        docs[doc] = row["body"]
        scores.setdefault(doc, np.zeros(6))[q] = meta["native"]["conditional_score"]
        inputs[str(path.relative_to(ROOT))] = sha
    if len(docs) != 480 or len(seen) != 2880:
        raise ValueError("complete source family required")
    texts = [docs[d] for d in sorted(docs)]
    feature = WordFactorReference(method="pca", n_factors=4, max_features=200)
    with threadpool_limits(1):
        feature.fit(texts)
    lexical = L.export_reference(
        feature,
        texts,
        domain="HM81_public_source",
        source_ids=[str(d) for d in sorted(docs)],
    )
    grain = L.score_documents(lexical, texts)["z"]
    observer_id = digest(SOURCE / "01/observer.json")
    bundle = M.source_bundle(
        grain,
        [scores[d] for d in sorted(docs)],
        lexical,
        observer_id,
        {
            "source": str(SOURCE.relative_to(ROOT)),
            "source_family": 0,
            "fits": {
                "lexical_PCA_varimax": 1,
                "moment_calibration": 1,
                "teacher_or_label_regression": 0,
            },
        },
    )
    material_path = ROOT / "configs/grain_math_passive_event_composition_v1.json"
    material = read_json(material_path)
    application = fresh_materials(material)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    alphabet = read_json(SOURCE / "00/manifest.json")["alphabet"]
    if [tokenizer.encode(s, add_special_tokens=False) for s in alphabet["labels"]] != [
        [t] for t in alphabet["token_ids"]
    ]:
        raise ValueError("answer alphabet drift")
    design = []
    for episode in application:
        for query in M.TARGETS:
            text, ids, part, _, _ = render(
                episode["body"], query, "A", 1, material["system"], tokenizer
            )
            design.append(
                {
                    **episode,
                    "query": query,
                    "unit": len(design),
                    "input_ids": ids,
                    "partition": part,
                    "rendered": text,
                }
            )
    (out / "00/code").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    code = {}
    for name in CODE:
        dest = out / "00/code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        code[name] = digest(dest)
    write(out / "00/bundle.json", bundle)
    write(out / "00/design.json", design)
    write(out / "00/application.json", application)
    inputs.update(
        {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                material_path,
                SOURCE / "01/observer.json",
                SOURCE / "01/raw_seal.json",
                SOURCE / "00/rows.json",
                parent / "00/rows.json",
                parent / "01/raw_seal.json",
                SOURCE / "verification.json",
                MODEL / "config.json",
            )
        }
    )
    # An absolute model path is allowed as provenance, never an application input.
    write(
        out / "00/manifest.json",
        {
            "packet": "HM81",
            "code": code,
            "source_files": inputs,
            "assets": {
                f"00/{n}.json": digest(out / f"00/{n}.json")
                for n in ("bundle", "design", "application")
            },
            "alphabet": alphabet,
            "observer_id": observer_id,
            "source_bodies_readings": [480, 2880],
            "new_bodies_readings": [64, 384],
            "max_model_seconds": 240,
            "source_result_after_construction": True,
            "application_counterfactual_table": False,
            "historical_SU7_or_full_token_slice_model": False,
            "source_public_only": True,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "original_scope_complete": False,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "calls": 0,
            "attempted": 0,
            "model_seconds": 0.0,
            "in_flight": None,
        },
    )
    print(
        "HM81 PREPARED: source480/2880; cold64/384; one lexical fit; no teacher fit",
        flush=True,
    )


def check(out):
    """Require unchanged source/configuration/code; no application fitting."""
    manifest = read_json(out / "00/manifest.json")
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("code drift:" + name)
    for field, root in (("source_files", ROOT), ("assets", out)):
        for name, sha in manifest[field].items():
            if digest(root / name) != sha:
                raise ValueError("bound file drift:" + name)
    return manifest


def collect(out):
    """Read new bodies once, preserving attempts and a finite resource ceiling."""
    manifest, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["in_flight"] is not None or status["status"] == "STOPPED":
        raise RuntimeError("uncertain/stopped call requires inspection, not retry")
    started, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual = observer.manifest()
        old = read_json(SOURCE / "01/observer.json")
        for key in (
            "model",
            "revision",
            "dtype",
            "device",
            "layers",
            "hidden_size",
            "max_tokens",
            "attention",
            "use_cache",
            "training",
            "template_sha256",
        ):
            if key in old and actual.get(key) != old[key]:
                raise ValueError("observer drift:" + key)
        write(out / "01/observer.json", actual)
        for row in read_json(out / "00/design.json"):
            dest = out / f"01/units/unit_{row['unit']:04d}.json"
            if dest.exists():
                continue
            if (
                status["attempted"] >= 384
                or previous + time.monotonic() - started >= 240
            ):
                raise TimeoutError("HM81 finite model budget")
            status.update(
                status="COLLECTING",
                in_flight=row["unit"],
                attempted=status["attempted"] + 1,
            )
            write(out / "run_status.json", status)
            native, _ = observe_full(observer, row, manifest["alphabet"], 16)
            write(
                dest,
                {k: row[k] for k in ("unit", "author_key", "event_id", "query")}
                | {"native": native},
            )
            status.update(
                calls=status["calls"] + 1,
                in_flight=None,
                model_seconds=previous + time.monotonic() - started,
            )
            write(out / "run_status.json", status)
            if status["calls"] % 48 == 0:
                print(
                    f"HM81 {status['calls']}/384; {status['model_seconds']:.1f}s",
                    flush=True,
                )
        if status["calls"] != 384:
            raise ValueError("incomplete actual call count")
        status["status"] = "COLLECTED"
        files = {
            str(p.relative_to(out)): digest(p)
            for p in (out / "01/units").glob("unit_*.json")
            if not p.name.startswith("._")
        }
        write(out / "01/seal.json", {"files": files})
    except Exception as error:
        status.update(status="STOPPED", error=repr(error))
        raise
    finally:
        write(out / "run_status.json", status)


def analyze(out):
    """Produce event-only cold inputs, author explanations and alternate readback."""
    manifest = check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("new observations are not complete")
    for name, sha in read_json(out / "01/seal.json")["files"].items():
        if digest(out / name) != sha:
            raise ValueError("actual reading drift")
    destination = out / "03"
    destination.mkdir(exist_ok=False)
    bundle = read_json(out / "00/bundle.json")
    episodes = read_json(out / "00/application.json")
    grain = L.score_documents(bundle["lexical"], [e["body"] for e in episodes])["z"]
    records = {
        e["author_key"]: {"author_key": e["author_key"], "events": []} for e in episodes
    }
    query = np.empty((64, 6))
    decoder_errors = []
    for i, event in enumerate(episodes):
        for j in range(6):
            meta = read_json(out / f"01/units/unit_{6 * i + j:04d}.json")
            if (meta["author_key"], meta["event_id"], meta["query"]) != (
                event["author_key"],
                event["event_id"],
                M.TARGETS[j],
            ):
                raise ValueError("same-unit query address mismatch")
            native = meta["native"]
            logits = np.asarray(native["candidate_logits"])
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            decoder_errors.append(
                abs(
                    probabilities @ np.asarray(manifest["alphabet"]["values"])
                    - native["conditional_score"]
                )
            )
            query[i, j] = native["conditional_score"]
        records[event["author_key"]]["events"].append(
            {k: event[k] for k in ("event_id", "condition", "qualification")}
            | {
                "observer_id": manifest["observer_id"],
                "grain": grain[i].tolist(),
                "query": query[i].tolist(),
            }
        )
    cold_input = destination / "observations.jsonl"
    cold_input.write_text("".join(json.dumps(r) + "\n" for r in records.values()))
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_cold_joint_author.py"),
            "--bundle",
            str(out / "00/bundle.json"),
            "--sha256",
            digest(out / "00/bundle.json"),
            "--input",
            str(cold_input),
            "--output",
            str(destination / "author_profiles.jsonl"),
            "--audit",
            str(destination / "cold_file_audit.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    write(
        destination / "cold_process.json",
        {
            "exit_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        },
    )
    outputs = [
        json.loads(line)
        for line in (destination / "author_profiles.jsonl").read_text().splitlines()
    ]
    errors, covariance, shifted, rows = [], [], [], []
    for profile in outputs:
        record = records[profile["author_key"]]
        single = M.cold_score(bundle, record)
        errors.append(
            float(np.max(abs(np.asarray(profile["profile"]) - single["profile"])))
        )
        g = (
            np.asarray([e["grain"] for e in record["events"]]) - bundle["grain_center"]
        ) / bundle["grain_scale"]
        q = (
            np.asarray([e["query"] for e in record["events"]]) - bundle["query_center"]
        ) / bundle["query_scale"]
        explicit = []
        for gg, qq in zip(g, q, strict=True):
            explicit.append(
                [
                    *gg,
                    *qq,
                    *[a * b for a in gg for b in qq],
                    *[qq[j] * qq[k] for j in range(6) for k in range(j + 1, 6)],
                ]
            )
        alternate = (np.mean(explicit, 0) - bundle["feature_center"]) / bundle[
            "feature_scale"
        ]
        errors.append(float(np.max(abs(alternate - profile["profile"]))))
        contributions = np.sum([e["contribution"] for e in profile["contributions"]], 0)
        errors.append(float(np.max(abs(contributions - profile["profile"]))))
        conditional = sum(
            c["share"] * np.asarray(c["profile"])
            for c in profile["conditional_profiles"].values()
        )
        errors.append(float(np.max(abs(conditional - profile["profile"]))))
        cov = (g - g.mean(0)).T @ (q - q.mean(0)) / len(g)
        errors.append(float(np.max(abs(cov - profile["grain_query_covariance"]))))
        scrambled = json.loads(json.dumps(record))
        for i, event in enumerate(scrambled["events"]):
            event["query"] = record["events"][(i + 1) % len(g)]["query"]
        ps = M.cold_score(bundle, scrambled)["profile"]
        difference = np.asarray(ps) - profile["profile"]
        errors.append(float(np.max(abs(difference[:10]))))
        covariance.append(float(np.sqrt(np.mean(cov**2))))
        shifted.append(float(np.sqrt(np.mean(difference[10:34] ** 2))))
        rows.append(
            {
                "author": profile["author_key"],
                "events": len(g),
                "covariance_RMS": covariance[-1],
                "shifted_joint_RMS": shifted[-1],
                "profile_RMS": float(np.sqrt(np.mean(alternate**2))),
            }
        )
    distances = []
    for a, b in itertools.combinations(outputs, 2):
        difference = np.asarray(a["profile"]) - b["profile"]
        distances.append(
            {
                "a": a["author_key"],
                "b": b["author_key"],
                "marginal_mean_squared": float(np.mean(difference[:10] ** 2)),
                "paired_product_squared": float(np.mean(difference[10:] ** 2)),
                "finite_kernel_mean_squared": float(np.mean(difference**2)),
            }
        )
    if max(errors + decoder_errors) > 1e-10:
        raise ArithmeticError("cold observation arithmetic drift")
    write_csv(destination / "author_summary.csv", rows)
    write_csv(destination / "pair_distances.csv", distances)
    summary = {
        "status": "VERIFIED_FINITE_COLD_INTERFACE_HM81_NOT_PSYCHOLOGICAL_CALIBRATION",
        "new_calls": 384,
        "source_score_reuses": 2880,
        "cold_authors_episodes": [8, 64],
        "new_teacher_fits": 0,
        "source_lexical_fits": 1,
        "source_moment_calibrations": 1,
        "same_unit_covariance_RMS": covariance,
        "shift_pairing_joint_RMS": shifted,
        "arithmetic_max_error": max(errors),
        "decoder_max_error": max(decoder_errors),
        "source_score_access_by_cold_process": False,
        "no_application_inverse_counterfactual_table_or_truth": True,
        "not_historical_SU7_full_token_slice_or_new_math": True,
        "no_accuracy_p_value_or_personality_validity_claim": True,
        "original_scope_complete": False,
    }
    write(destination / "summary.json", summary)
    write(
        destination / "seal.json",
        {
            "files": {
                str(p.relative_to(out)): digest(p)
                for p in destination.iterdir()
                if p.is_file() and not p.name.startswith("._") and p.name != "seal.json"
            }
        },
    )
    print(json.dumps(summary), flush=True)


def main():
    """Run finite stages explicitly; no automatic continuation or old queue."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == "collect":
        with collector_guard():
            collect(args.out)
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
