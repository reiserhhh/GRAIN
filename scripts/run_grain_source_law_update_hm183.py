#!/usr/bin/env python3
"""Separate source author selection from conditional evidence; no score search."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_profile_reader as P
from grain_research.construction import math_source_law_update as M
from grain_research.construction import math_vector_lift as V
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_source_law_update_hm183.json"
DEFAULT = ROOT / "results/grain_math_response_v1/source_sampling_update_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_SOURCE_LAW_UPDATE_PLAN_HM183.md",
    "grain_research/construction/math_source_law_update.py",
    "scripts/run_grain_source_law_update_hm183.py",
    "scripts/score_grain_source_law_update_cold.py",
    "tests/test_grain_source_law_update.py",
)


def cosine(a, b):
    """Retain zero directions as undefined, not as perfect or failed agreement."""
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else None


def load(path):
    """Load registered numeric arrays, never raw records or application targets."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prepare(out):
    """Seal original actions and reader sources before outcomes of this comparison."""
    cfg, code, inputs = R.read(ROOT / CONFIG), {}, {}
    for key, stages in (
        (
            "vector_parent",
            ("compiled_seal.json", "scored_seal.json", "verification_seal.json"),
        ),
        (
            "reader_parent",
            (
                "compiled_seal.json",
                "scored_seal.json",
                "analysis_seal.json",
                "verification_seal.json",
            ),
        ),
    ):
        par = ROOT / cfg[key]
        prior = R.check(par, stages)
        code.update(prior["code"])
        for name in ("manifest.json", "prepared_seal.json", *stages):
            paths = (name,) if name == "manifest.json" else (name, *R.read(par / name))
            for p in paths:
                rel = str((par / p).relative_to(ROOT))
                inputs[rel] = R.digest(ROOT / rel)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": code | {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": inputs,
            "original_scope_complete": False,
        },
    )
    R.write(
        out / "prepared.json",
        {
            "source_authors": 386,
            "views": 2,
            "actions": 6,
            "laws": list(M.LAWS),
            "cold_frames": 240,
            "new_model_or_application_labels": 0,
        },
    )
    R.seal(out, "prepared_seal.json", ("prepared.json",))


def construct(out):
    """Compile every target under unchanged actions, with both sampling levels kept."""
    cfg, began = R.check(out)["config"], time.perf_counter()
    vp, rp = ROOT / cfg["vector_parent"], ROOT / cfg["reader_parent"]
    z = R.prepared(rp)
    old, footprint = load(rp / "source_fields.npz"), load(vp / "source_footprints.npz")
    y = z["targets"]
    if y.shape != (386, 5) or not np.array_equal(z["source"], R.prepared(vp)["source"]):
        raise ValueError("Source author identities changed")
    source_fields, all_laws, mass_rows = [], [], []
    errors = {
        "profile": 0.0,
        "probability_account": 0.0,
        "baseline_reader": 0.0,
        "marginal_change_identity": 0.0,
        "reader_account": 0.0,
    }
    profiles = y @ (2 ** np.arange(5))
    original = np.full((386, 64), 1 / (386 * 64))
    for v in range(2):
        d = np.moveaxis(footprint["weights"][v], 2, 0)
        d = np.concatenate((d, d.sum(0, keepdims=True)))
        laws = np.stack([M.four_laws(original, original - q)[0] for q in d])
        for k in range(6):
            _, ii = M.four_laws(original, laws[k, 3])
            errors["probability_account"] = max(
                errors["probability_account"],
                float(
                    np.max(
                        np.abs(
                            laws[k, 3]
                            - original
                            - (laws[k, 1] - original)
                            - (laws[k, 2] - original)
                            - ii
                        )
                    )
                ),
            )
            for j in range(4):
                mass = laws[k, j].sum(1)
                errors["profile"] = max(
                    errors["profile"],
                    float(
                        np.max(
                            np.abs(
                                np.bincount(profiles, weights=mass, minlength=32)
                                - np.bincount(
                                    profiles,
                                    weights=np.full(386, 1 / 386),
                                    minlength=32,
                                )
                            )
                        )
                    ),
                )
            author_ratio = laws[k, 3].sum(1) * 386
            original_a, original_b = M.disintegrate(original)
            new_a, new_b = M.disintegrate(laws[k, 3])
            mass_rows.append(
                {
                    "view": v,
                    "action": cfg["actions"][k],
                    "author_weight_ratio_min": float(author_ratio.min()),
                    "author_weight_ratio_max": float(author_ratio.max()),
                    "author_total_variation": float(
                        np.abs(new_a - original_a).sum() / 2
                    ),
                    "evidence_conditional_TV_mean": float(
                        (np.abs(new_b - original_b).sum(1) / 2).mean()
                    ),
                    "all_probabilities_min": float(laws[k].min()),
                }
            )
        effective = np.empty((6, 4, 386, 4096))
        for u in range(386):
            phi = V.tensor_features(z["source_coordinates"][u, v, V.OFFSETS])
            effective[:, :, u] = 386 * np.einsum("kjt,td->kjd", laws[:, :, u], phi)
        fields = np.stack([M.readers_from_features(effective[k], y) for k in range(6)])
        errors["baseline_reader"] = max(
            errors["baseline_reader"],
            float(np.max(np.abs(fields[:, 0] - old["fields"][v]))),
        )
        w = footprint["moments"][v]
        w = np.concatenate((w, w.sum(0, keepdims=True)))
        expected = M.contrast_change(w, y.mean(0))
        errors["marginal_change_identity"] = max(
            errors["marginal_change_identity"],
            float(np.max(np.abs(fields[:, 0, :, 0] - fields[:, 3, :, 0] - expected))),
        )
        changes = np.stack(
            (
                fields[:, 1] - fields[:, 0],
                fields[:, 2] - fields[:, 0],
                fields[:, 3] - fields[:, 1] - fields[:, 2] + fields[:, 0],
            ),
            axis=1,
        )
        errors["reader_account"] = max(
            errors["reader_account"],
            float(np.max(np.abs(changes.sum(1) - (fields[:, 3] - fields[:, 0])))),
        )
        source_fields.append(fields)
        all_laws.append(laws)
    f = np.stack(source_fields)
    R.archive(out / "reader_bundle.npz", fields=f, version=np.array(M.VERSION))
    R.archive(out / "source_laws.npz", laws=np.stack(all_laws))
    R.table(out / "sampling_levels.csv", mass_rows)
    rows = []
    for v, k, q, c in np.ndindex(2, 6, 5, 4):
        before, a, b, after = f[v, k, :, q, c]
        da, db, interaction = a - before, b - before, after - a - b + before
        total = after - before
        norm = np.linalg.norm(total)
        rows.append(
            {
                "view": v,
                "action": cfg["actions"][k],
                "target": V.TARGETS[q],
                "component": P.COMPONENTS[c],
                "before_after_cosine": cosine(before, after),
                "relative_update": float(norm / np.linalg.norm(before))
                if np.linalg.norm(before)
                else None,
                "author_projection": float(da @ total / norm**2) if norm else None,
                "evidence_projection": float(db @ total / norm**2) if norm else None,
                "interaction_projection": float(interaction @ total / norm**2)
                if norm
                else None,
                "signed_cross": float(
                    2 * (da @ db + da @ interaction + db @ interaction)
                ),
                "update_norm": float(norm),
            }
        )
    R.table(out / "all_reader_updates.csv", rows)
    gram = []
    for k, q, c in np.ndindex(6, 5, 4):
        before0, after0 = f[0, k, [0, 3], q, c]
        before1, after1 = f[1, k, [0, 3], q, c]
        gram.append(
            {
                "action": cfg["actions"][k],
                "target": V.TARGETS[q],
                "component": P.COMPONENTS[c],
                "original_cross_view_cosine": cosine(before0, before1),
                "updated_cross_view_cosine": cosine(after0, after1),
                "update_cross_view_cosine": cosine(after0 - before0, after1 - before1),
            }
        )
    R.table(out / "state_vs_update.csv", gram)
    R.write(
        out / "construction_receipt.json",
        {
            "errors": errors,
            "seconds": time.perf_counter() - began,
            "fields": 960,
            "all_actions_retained": True,
            "new_model_accuracy_or_fit": False,
        },
    )
    if max(errors.values()) > cfg["numerical_tolerance"]:
        raise ValueError("Fixed finite account failed; outputs retained, no refinement")
    R.seal(
        out,
        "compiled_seal.json",
        (
            "reader_bundle.npz",
            "source_laws.npz",
            "sampling_levels.csv",
            "all_reader_updates.csv",
            "state_vs_update.csv",
            "construction_receipt.json",
        ),
    )
    print(json.dumps(R.read(out / "construction_receipt.json")))


def score(out):
    """Run an independent process that cannot load source labels or native answers."""
    cfg = R.check(out, ("compiled_seal.json",))["config"]
    command = [
        sys.executable,
        str(ROOT / "scripts/score_grain_source_law_update_cold.py"),
        "--bundle",
        str(out / "reader_bundle.npz"),
        "--input",
        str(ROOT / cfg["reader_parent"] / "cold_input.jsonl"),
        "--output",
        str(out / "cold_output.jsonl"),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    if report["records"] != cfg["cold_records"]:
        raise ValueError("Cold scope changed")
    R.write(out / "cold_summary.json", report)
    R.seal(out, "scored_seal.json", ("cold_output.jsonl", "cold_summary.json"))
    print(result.stdout.strip())


def verify(out):
    """Check cold readings by independent role-Kronecker construction and full account."""
    cfg = R.check(out, ("compiled_seal.json", "scored_seal.json"))["config"]
    f = load(out / "reader_bundle.npz")["fields"]
    inputs = (ROOT / cfg["reader_parent"] / "cold_input.jsonl").read_text().splitlines()
    outputs = (out / "cold_output.jsonl").read_text().splitlines()
    error, count = 0.0, 0
    for iline, oline in zip(inputs, outputs, strict=True):
        row, observed = json.loads(iline), json.loads(oline)
        means = []
        for role in range(3):
            xx = [e["coordinates"] for e in row["events"] if e["slot"] // 4 == role]
            a = np.zeros(16)
            a[0] = len(xx) / 4
            if xx:
                a[1:] = np.sum(xx, axis=0) / 4
            means.append(a)
        phi = np.kron(np.kron(means[0], means[1]), means[2])
        direct = (f.reshape(-1, 4096) @ phi).reshape(f.shape[:-1])
        error = max(
            error, float(np.max(np.abs(direct - np.asarray(observed["retained"]))))
        )
        count += 1
    R.write(
        out / "verification.json",
        {
            "cold_frames": count,
            "Kronecker_error": error,
            "same_agent_not_blind": True,
            "old_source_moment_identity": R.read(out / "construction_receipt.json")[
                "errors"
            ],
        },
    )
    if error > cfg["numerical_tolerance"]:
        raise ValueError("Cold account failed; no tolerance change")
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(R.read(out / "verification.json")))


def main():
    """Execute one registered finite stage,never resume or tune another packet."""
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("prepare", "construct", "score", "verify"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    globals()[a.phase](a.out)


if __name__ == "__main__":
    main()
