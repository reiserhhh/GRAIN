#!/usr/bin/env python3
"""Compile fixed author-contrast observations; run new histories, not accuracy grids."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from grain_research.construction import math_author_contrast as M
from grain_research.construction import math_cold_joint_author as J

OUT = ROOT / "results/grain_math_response_v1/author_contrast_observability_r1"
HM81 = ROOT / "results/grain_math_response_v1/cold_joint_author_r1"
HM79 = ROOT / "results/grain_math_response_v1/role_assignment_dealiasing_r1"
CODE = (
    "docs/GRAIN_AUTHOR_CONTRAST_OBSERVABILITY_PLAN_HM82.md",
    "grain_research/construction/math_author_contrast.py",
    "scripts/run_grain_author_contrast_hm82.py",
    "scripts/verify_grain_author_contrast_hm82.py",
    "tests/test_grain_math_author_contrast.py",
)
TARGETS = ("condition_choice", "focal_polarity", "conditional_polarity", "role_joint")


def sha(path):
    """Hash a declared input or frozen output, never credentials or private text."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    """Read a declared JSON artifact."""
    return json.loads(Path(path).read_text())


def write(path, value):
    """Write generated scientific artifacts, refusing replacement."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def table(path, rows):
    """Save complete fixed-schema numerical tables without dropping failed arms."""
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seal(root, paths):
    """Authenticate all explicitly completed artifacts."""
    return {str(p.relative_to(root)): sha(p) for p in paths}


def source_inputs():
    """Replay HM81's authenticated source aliases and existing lexical observer."""
    bundle = load(HM81 / "00/bundle.json")
    J.verify_bundle(bundle)
    frozen = load(HM81 / "00/manifest.json")
    inputs = dict(frozen["source_files"])
    bundle_path = HM81 / "00/bundle.json"
    if sha(bundle_path) != frozen["assets"]["00/bundle.json"]:
        raise ValueError("HM81 source bundle seal drift")
    inputs[str(bundle_path.relative_to(ROOT))] = sha(bundle_path)
    inputs[str((HM81 / "00/manifest.json").relative_to(ROOT))] = sha(
        HM81 / "00/manifest.json"
    )
    for name in (
        "grain_research/construction/math_cold_joint_author.py",
        "grain_research/local_measure.py",
    ):
        if name in frozen["code"] and sha(ROOT / name) != frozen["code"][name]:
            raise ValueError("inherited source function drift: " + name)
        inputs[name] = sha(ROOT / name)
    for name, expected in inputs.items():
        if sha(ROOT / name) != expected:
            raise ValueError("HM81 input drift: " + name)
    parent = ROOT / load(HM79 / "00/manifest.json")["spec"]["parent"]
    prior = {r["unit"]: r for r in load(parent / "00/rows.json")}
    docs, score, coordinates, seen = {}, {}, {}, set()
    for row in load(HM79 / "00/rows.json"):
        if row["family"] != 0:
            continue
        reused = row["old_unit"] is not None
        base, unit = (parent, row["old_unit"]) if reused else (HM79, row["new_unit"])
        path = base / f"01/units/unit_{unit:04d}.json"
        if str(path.relative_to(ROOT)) not in inputs:
            raise ValueError("undeclared source score")
        meta = load(path)
        addressed = prior[unit] if reused else row
        if any(meta[k] != addressed[k] for k in ("unit", "doc_id", "family", "query")):
            raise ValueError("source score alias mismatch")
        if reused and any(
            row[k] != addressed[k] for k in ("body", "query", "input_ids")
        ):
            raise ValueError("reused source material mismatch")
        doc, q = row["doc_id"], J.TARGETS.index(row["query"])
        if (doc, q) in seen:
            raise ValueError("duplicate source reading")
        seen.add((doc, q))
        docs[doc], coordinates[doc] = row["body"], row["coordinates"]
        score.setdefault(doc, np.zeros(6))[q] = meta["native"]["conditional_score"]
    if len(docs) != 480 or len(seen) != 2880:
        raise ValueError("full declared source family required")
    ids = sorted(docs)
    grain = L.score_documents(bundle["lexical"], [docs[d] for d in ids])["z"]
    raw_q = np.array([score[d] for d in ids])
    joint, g, q = J.observed_features(bundle, grain, raw_q)
    return bundle, inputs, np.array([coordinates[d] for d in ids]), joint, g, q, raw_q


def prepare(out):
    """Freeze all observation functions before sampling any application author law."""
    if out.exists():
        raise FileExistsError("preserve every existing run")
    bundle, inputs, design, joint, g, q, raw_q = source_inputs()
    x, c, actor, _ = design.T
    targets = np.column_stack((c, x, c * x, x * c * actor))
    p = np.full(len(g), 1 / len(g))
    if (
        np.max(abs(p @ targets)) > 1e-12
        or np.max(abs((targets.T * p) @ targets - np.eye(4))) > 1e-12
    ):
        raise ValueError("registered balanced source target design required")
    marginal = np.column_stack((g, q))
    cube = raw_q**3
    cube_q = (cube - cube.mean(0)) / cube.std(0, ddof=1)
    inverse_q = np.cbrt(cube)
    inverse_joint, _, _ = J.observed_features(
        bundle, grain_from_g(bundle, g), inverse_q
    )
    arms = {
        "G4": g,
        "Q6": q,
        "marginal10": marginal,
        "joint49": joint,
        "conditional_marginal21": np.column_stack((marginal, c, c[:, None] * marginal)),
        "conditional_joint99": np.column_stack((joint, c, c[:, None] * joint)),
        "recoded_joint49": J.lift(g, cube_q),
        "inverse_recoded_joint49": inverse_joint,
    }
    out.mkdir(parents=True)
    (out / "source_code").mkdir()
    code = {}
    for name in CODE:
        path = out / "source_code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
        code[name] = sha(path)
    inputs[str((HM81 / "00/bundle.json").relative_to(ROOT))] = sha(
        HM81 / "00/bundle.json"
    )
    arrays = {"reference": p, "targets": targets, "conditions": c, "design": design}
    compiled, diagnostics = {}, []
    for arm, b in arms.items():
        projection = M.source_projection(b, p)
        values = M.project(targets, p, projection)
        compiled[arm] = values.tolist()
        arrays[arm + "_features"] = b
        arrays[arm + "_compiled"] = values
        for j, target in enumerate(TARGETS):
            diagnostics.append(
                {
                    "arm": arm,
                    "target": target,
                    "rank": projection["rank"],
                    "tolerance": projection["tolerance"],
                    "visible_norm2": M.norm2(values[:, j], p),
                    "lost_norm2": M.norm2(targets[:, j] - values[:, j], p),
                    "target_unit": "declared finite material contrast;not a personality factor",
                }
            )
        arrays[arm + "_singular"] = projection["singular"]
    np.savez_compressed(out / "source.npz", **arrays)
    write(
        out / "bundle.json",
        {"targets": TARGETS, "compiled": compiled, "registered_events": 480},
    )
    table(out / "source_target_visibility.csv", diagnostics)
    write(
        out / "source_freeze.json",
        {
            "code": code,
            "inputs": inputs,
            "files": seal(
                out,
                [
                    out / "source.npz",
                    out / "bundle.json",
                    out / "source_target_visibility.csv",
                ],
            ),
            "application_not_generated": True,
            "new_neural_calls": 0,
            "reference_is_declared_uniform_vocabulary_not_population": True,
        },
    )


def grain_from_g(bundle, g):
    """Undo HM81's known source unit solely for the exact recoding control."""
    return g * bundle["grain_scale"] + bundle["grain_center"]


def checked_source(out):
    """Refuse code/source changes after the plan's freeze."""
    frozen = load(out / "source_freeze.json")
    for name, expected in frozen["code"].items():
        if sha(ROOT / name) != expected or sha(out / "source_code" / name) != expected:
            raise ValueError("frozen code drift: " + name)
    for name, expected in frozen["inputs"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("source input drift: " + name)
    for name, expected in frozen["files"].items():
        if sha(out / name) != expected:
            raise ValueError("compiled source drift: " + name)


def cold(out, expected_sha):
    """Execute only compiled functions and sampled event addresses in a fresh process."""
    names = ("bundle.json", "observations.jsonl", "scores.jsonl", "cold_audit.json")
    allowed = {os.path.realpath(out / n) for n in names}
    opened = set()

    def guard(event, values):
        if event == "open" and not isinstance(values[0], int):
            path = os.path.realpath(os.fsdecode(values[0]))
            opened.add(path)
            if path not in allowed and not path.startswith("/dev/"):
                raise PermissionError("cold scoring refused " + path)

    sys.addaudithook(guard)
    if sha(out / "bundle.json") != expected_sha:
        raise ValueError("bundle authentication failed")
    bundle = load(out / "bundle.json")
    with (out / "scores.jsonl").open("x") as stream:
        for line in (out / "observations.jsonl").read_text().splitlines():
            record = json.loads(line)
            if set(record) != {"author_key", "events"}:
                raise ValueError("application truth forbidden in cold inputs")
            scores = {
                a: M.cold_score(values, record["events"])
                for a, values in bundle["compiled"].items()
            }
            stream.write(
                json.dumps(
                    {"author_key": record["author_key"], "arms": scores},
                    allow_nan=False,
                )
                + "\n"
            )
    write(
        out / "cold_audit.json",
        {
            "opened": sorted(opened | {os.path.realpath(out / "cold_audit.json")}),
            "allowed": sorted(allowed),
            "truth_or_source_npz_access": False,
        },
    )


def channels():
    """Return fixed four-state channels; none is estimated or selected."""
    states = [(c, x) for c in (-1, 1) for x in (-1, 1)]
    flip = np.zeros((4, 4))
    for i, (c, x) in enumerate(states):
        flip[i, i], flip[i, states.index((c, -x))] = 0.8, 0.2
    return {
        "identity": np.eye(4),
        "readdress": np.eye(4)[[3, 1, 0, 2]],
        "forget_condition": np.eye(2)[[0, 1, 0, 1]],
        "forget_content": np.eye(2)[[0, 0, 1, 1]],
        "flip_content_02": flip,
    }


def run(out):
    """Generate all bounded new histories, cold score, and retain every result."""
    checked_source(out)
    start = time.perf_counter()
    data = np.load(out / "source.npz", allow_pickle=False)
    p, targets, c = data["reference"], data["targets"], data["conditions"]
    worlds = (
        "choice_only",
        "level_only",
        "conditional_only",
        "role_joint_only",
        "mixed",
    )
    authors, observations = [], []
    for seed in (8201, 8202, 8203):
        rng = np.random.default_rng(seed)
        for j, world in enumerate(worlds):
            for pair in range(4):
                amplitude = float(rng.uniform(0.15, 0.35))
                direction = targets[:, j] if j < 4 else targets.mean(1)
                for sign in (-1, 1):
                    key = f"s{seed}_{world}_p{pair}_{sign:+d}"
                    law = p * (1 + sign * amplitude * direction)
                    M.probability(law)
                    events = rng.choice(len(p), 128, p=law).tolist()
                    authors.append(
                        {
                            "author_key": key,
                            "seed": seed,
                            "world": world,
                            "pair": pair,
                            "sign": sign,
                            "amplitude": amplitude,
                            "probability": law.tolist(),
                        }
                    )
                    observations.append({"author_key": key, "events": events})
    write(out / "truth_terminal_only.json", authors)
    with (out / "observations.jsonl").open("x") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in observations)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "cold",
        "--out",
        str(out),
        "--sha256",
        sha(out / "bundle.json"),
    ]
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    write(
        out / "cold_process.json",
        {
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    if result.returncode:
        raise RuntimeError("cold scoring failed; preserve this run")
    scores = {
        v["author_key"]: v["arms"]
        for v in (
            json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()
        )
    }
    bundle = load(out / "bundle.json")
    score_rows, accounts, errors = [], [], []
    for arm in bundle["compiled"]:
        b = data[arm + "_features"]
        projection = M.source_projection(b, p)
        tv = np.asarray(bundle["compiled"][arm])
        for author in authors:
            law = np.asarray(author["probability"])
            truth, exact = law @ targets, law @ tv
            measured = scores[author["author_key"]][arm]["scores"]
            for j, target in enumerate(TARGETS):
                score_rows.append(
                    {
                        "author_key": author["author_key"],
                        "world": author["world"],
                        "arm": arm,
                        "target": target,
                        "true_declared_target": truth[j],
                        "observable_expectation": exact[j],
                        "sampled_score": measured[j],
                        "omitted_target": truth[j] - exact[j],
                        "sampling_error": measured[j] - exact[j],
                        "n": 128,
                    }
                )
        for a, other in zip(authors[::2], authors[1::2], strict=True):
            f = (np.array(other["probability"]) - a["probability"]) / p
            account = M.target_account(f, targets, p, projection)
            choice = M.condition_account(f, p, c)
            errors.extend(
                [
                    account["norm_closure_error"],
                    account["target_closure_error"],
                    abs(
                        choice["total_norm2"]
                        - choice["choice_norm2"]
                        - choice["within_norm2"]
                    ),
                    abs(choice["cross"]),
                ]
            )
            for j, target in enumerate(TARGETS):
                accounts.append(
                    {
                        "seed": a["seed"],
                        "world": a["world"],
                        "pair": a["pair"],
                        "arm": arm,
                        "target": target,
                        "total_norm2": account["total_norm2"],
                        "visible_norm2": account["visible_norm2"],
                        "lost_norm2": account["lost_norm2"],
                        "choice_norm2": choice["choice_norm2"],
                        "conditional_norm2": choice["within_norm2"],
                        "true_contrast": account["truth"][j],
                        "visible_contrast": account["visible_target"][j],
                        "lost_contrast": account["lost_target"][j],
                    }
                )
    table(out / "author_scores.csv", score_rows)
    table(out / "contrast_accounts.csv", accounts)
    c4, x4 = np.array([-1, -1, 1, 1]), np.array([-1, 1, -1, 1])
    channel_rows = []
    for name, k in channels().items():
        for target, f in (("choice", c4), ("level", x4), ("response", c4 * x4)):
            a = M.channel_account(f, np.full(4, 0.25), k)
            errors.append(a["closure_error"])
            channel_rows.append(
                {
                    "channel": name,
                    "target": target,
                    "input_norm2": a["total_norm2"],
                    "visible_norm2": a["visible_norm2"],
                    "lost_norm2": a["lost_norm2"],
                    "closure_error": a["closure_error"],
                }
            )
    table(out / "channel_controls.csv", channel_rows)
    write(
        out / "summary.json",
        {
            "status": "FINITE_HM82_EXECUTED_REQUIRES_ALTERNATE_VERIFICATION",
            "source_bodies_actual_readings": [480, 2880],
            "new_simulated_author_laws_records": [
                len(authors),
                sum(len(r["events"]) for r in observations),
            ],
            "source_target_rows": 32,
            "score_rows": len(score_rows),
            "contrast_rows": len(accounts),
            "new_model_calls_or_teacher_fits": 0,
            "max_closure_error": max(errors),
            "seconds": time.perf_counter() - start,
            "condition_is_reported_companion_polarity_not_external_environment": True,
            "not_natural_author_or_psychological_validity": True,
        },
    )
    checked_source(out)
    write(
        out / "output_seal.json",
        seal(
            out,
            [p for p in out.iterdir() if p.is_file() and not p.name.startswith("._")],
        ),
    )


def main():
    """Run one bounded packet; never overwrite/restart a completed artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "cold"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.phase == "prepare":
            prepare(args.out)
        elif args.phase == "run":
            run(args.out)
        else:
            cold(args.out, args.sha256)


if __name__ == "__main__":
    main()
