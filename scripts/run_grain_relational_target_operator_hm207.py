#!/usr/bin/env python3
"""One source pairing mechanism experiment on existing addressed vector events."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_target_atlas as A
from grain_research.construction import math_relational_target_operator as M
from grain_research.construction import math_same_event_multitarget as N
from grain_research.construction import math_target_complement as P
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_same_event_multitarget_hm199 as H

CONFIG = "configs/grain_relational_target_operator_hm207.json"
DEFAULT = ROOT / "results/grain_math_response_v1/relational_target_operator_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_RELATIONAL_TARGET_OPERATOR_PLAN_HM207.md",
    "grain_research/construction/math_relational_target_operator.py",
    "scripts/run_grain_relational_target_operator_hm207.py",
    "tests/test_grain_relational_target_operator.py",
    "grain_research/construction/math_joint_target_atlas.py",
    "grain_research/construction/math_same_event_multitarget.py",
    "grain_research/construction/math_target_complement.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def arrays(path):
    """Read numeric arrays without pickle, raw text, or questionnaire access."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def number(x):
    """Keep undefined normalized accounts absent, not successful zero values."""
    return float(x) if np.isfinite(x) else None


def prepare(out):
    """Authenticate exact evidence addresses; only reuse the older author split."""
    cfg = R.read(ROOT / CONFIG)
    vp, sp, np_ = (
        ROOT / cfg[k] for k in ("vector_parent", "split_parent", "native_parent")
    )
    R.check(vp, ("compiled_seal.json", "scored_seal.json", "verification_seal.json"))
    R.check(sp, ("computed_seal.json",))
    H.check(np_)
    H.H.check_hashes(np_, R.read(np_ / "analyzed_seal.json"))
    H.H.check_hashes(np_, R.read(np_ / "verification_seal.json"))
    v, s, n = (
        arrays(vp / "observations.npz"),
        arrays(sp / "observations.npz"),
        arrays(np_ / "00/observations.npz"),
    )
    if not np.array_equal(v["source_indices"], s["source"]) or not np.array_equal(
        A.profile_codes(v["labels"]), s["codes"]
    ):
        raise ValueError("Source author/profile linkage changed")
    if set(v["source_indices"]) & set(v["panel_indices"]):
        raise ValueError("Source/panel author overlap")
    if not np.array_equal(v["panel_vectors"][:, 0], n["panel_vectors"]):
        raise ValueError("Native evidence is not the same cue-mask event family")
    if any(not np.array_equal(v[k], n[k]) for k in ("selected", "control")):
        raise ValueError("Different native edit addresses")
    if v["source_vectors"].shape != (386, 2, 12, 1024):
        raise ValueError("Changed declared source view support")
    files = [
        vp / "observations.npz",
        vp / "manifest.json",
        vp / "compiled_seal.json",
        sp / "observations.npz",
        sp / "manifest.json",
        np_ / "analysis.npz",
        np_ / "00/observations.npz",
        np_ / "00/manifest.json",
        np_ / "verification.json",
    ]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        source_means=v["source_vectors"].mean(2),
        codes=s["codes"],
        folds=s["folds"],
        source_indices=s["source"],
        panel_vectors=n["panel_vectors"],
        selected=n["selected"],
        control=n["control"],
        mapping=v["mapping"],
        author_ids=n["author_ids"],
    )
    quality = {
        "source_authors": 386,
        "panel_authors": 24,
        "source_events": 9264,
        "panel_events": 576,
        "fold_counts": np.bincount(s["folds"]).tolist(),
        "profile_counts_by_fold": [
            np.bincount(s["codes"][s["folds"] == i], minlength=32).tolist()
            for i in (0, 1)
        ],
        "source_windows_12_not_17_joined": True,
        "new_neural_API_private_application_labels": 0,
        "historical_encoder_overlap_and_N_guided_edits_retained": True,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(out, "prepared_seal.json", ("observations.npz", "input_quality.json"))
    print(json.dumps(quality))


def banks(out):
    """Load frozen compact dual operators, retaining all source terms."""
    z = arrays(out / "operator_bundle.npz")
    return [
        {k[3:]: v for k, v in z.items() if k.startswith(f"b{i}_")} for i in (0, 1)
    ], z["basis"]


def build(out):
    """Freeze all31 source operators before reading any native response contacts."""
    R.check(out)
    z, start = arrays(out / "observations.npz"), time.perf_counter()
    q, _ = P.fixed_basis(z["mapping"])
    bs = [
        M.compile_bank(z["source_means"][z["folds"] == i], z["codes"][z["folds"] == i])
        for i in (0, 1)
    ]
    R.archive(
        out / "operator_bundle.npz",
        basis=q,
        **{f"b{i}_{k}": v for i, bank in enumerate(bs) for k, v in bank.items()},
    )
    receipt = {
        "characters": 31,
        "components": list(M.COMPONENTS),
        "spaces": list(M.SPACES),
        "native_answers_read": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    R.write(out / "build_summary.json", receipt)
    R.seal(out, "built_seal.json", ("operator_bundle.npz", "build_summary.json"))
    print(json.dumps(receipt))


def panel_frames(z):
    """Construct five actual mean-vector arms without reencoding or fabricated text."""
    p = z["panel_vectors"]
    frames = np.empty((24, 2, 5, 1024))
    for u, view in np.ndindex(24, 2):
        s, c, all_ = z["selected"][u, view], z["control"][u, view], np.arange(12)
        groups = (all_, all_[~np.isin(all_, s)], all_[~np.isin(all_, c)], s, c)
        frames[u, view] = [p[u, view, ix].mean(0) for ix in groups]
    return frames


def analyze(out):
    """Separate cross-author source pairing from same-edit native response contacts."""
    cfg = R.check(out, ("built_seal.json",))["config"]
    start, z = time.perf_counter(), arrays(out / "observations.npz")
    bs, q = banks(out)
    dots = np.empty((3, 3, 5, 31))
    rows = []
    for i, j, space in np.ndindex(3, 3, 5):
        dots[i, j, space] = M.contraction(
            bs[0], bs[1], M.COMPONENTS[i], M.COMPONENTS[j], M.SPACES[space], q
        )
        for mode in range(31):
            rows.append(
                {
                    "component0": M.COMPONENTS[i],
                    "component1": M.COMPONENTS[j],
                    "space": M.SPACES[space],
                    "character": mode + 1,
                    "signed_cross_author_contraction": dots[i, j, space, mode],
                }
            )
    R.table(out / "source_operator_accounts.csv", rows)
    gram_a, gram_b = (
        bs[0]["pair_" + leg] @ bs[1]["pair_" + leg].T for leg in ("a", "b")
    )
    w, v = bs[0]["pair_w"], bs[1]["pair_w"]
    rng, refs = np.random.default_rng(cfg["reference_seed"]), []
    permutations = []
    for _ in range(cfg["pairing_references"]):
        _, p0 = M.repaire(bs[0], rng)
        _, p1 = M.repaire(bs[1], rng)
        refs.append(np.sum((w @ (gram_a * gram_b[np.ix_(p0, p1)])) * v, axis=1))
        permutations.append((p0, p1))
    refs = np.asarray(refs)
    primary = dots[2, 2, 0, M.TARGET_MODES].sum()
    null_primary = refs[:, M.TARGET_MODES].sum(1)
    upper = (1 + int(np.sum(null_primary >= primary))) / (1 + len(refs))
    R.table(
        out / "pairing_references.csv",
        [
            {"reference": k, "character": m + 1, "contraction": refs[k, m]}
            for k, m in np.ndindex(len(refs), 31)
        ],
    )
    frames = panel_frames(z)
    scores = np.empty((2, 3, 5, 2, 24, 2, 5, 31))
    stranger = np.roll(np.arange(24), -1)
    for bank, component, space, companion, view in np.ndindex(2, 3, 5, 2, 2):
        other = frames[np.arange(24) if companion == 0 else stranger, 1 - view, 0]
        own = frames[:, view]
        other = np.broadcast_to(other[:, None], own.shape)
        left, right = (own, other) if view == 0 else (other, own)
        scores[bank, component, space, companion, :, view] = M.read(
            bs[bank], M.COMPONENTS[component], M.SPACES[space], left, right, q
        )
    actions = np.stack(
        (scores[..., 1, :] - scores[..., 2, :], scores[..., 3, :] - scores[..., 4, :]),
        -2,
    )
    native = arrays(ROOT / cfg["native_parent"] / "analysis.npz")["native_actions"]
    contacts = []
    for bank, component, space, companion, operation, unit in np.ndindex(
        2, 3, 5, 2, 2, 3
    ):
        x = actions[bank, component, space, companion, :, :, operation][
            :, :, M.TARGET_MODES
        ].mean(1)
        y = native[:, :, operation, :, unit].mean(1)
        cov, corr = N.covariances(x, y), N.correlations(x, y)
        for r, target in np.ndindex(5, 5):
            contacts.append(
                {
                    "source_bank": bank,
                    "component": M.COMPONENTS[component],
                    "space": M.SPACES[space],
                    "companion": "own" if companion == 0 else "cyclic_stranger",
                    "operation": "deletion" if operation == 0 else "isolated",
                    "unit": ("score", "logodds", "geometric")[unit],
                    "source_target": A.TARGETS[r],
                    "reader_target": A.TARGETS[target],
                    "covariance": float(cov[r, target]),
                    "r": number(corr[r, target]),
                }
            )
    R.table(out / "native_contacts.csv", contacts)
    diag = [
        row
        for row in contacts
        if row["component"] == "pair"
        and row["space"] == "full"
        and row["unit"] == "score"
        and row["operation"] == "deletion"
        and row["source_target"] == row["reader_target"]
    ]
    R.archive(
        out / "analysis.npz",
        contractions=dots,
        references=refs,
        scores=scores,
        actions=actions,
        permutation0=np.array([p[0] for p in permutations]),
        permutation1=np.array([p[1] for p in permutations]),
    )
    summary = {
        "primary_pair_cross_author_sum5": float(primary),
        "primary_pairing_reference_upper_rank": upper,
        "pair_by_target": {
            name: float(dots[2, 2, 0, mode])
            for name, mode in zip(A.TARGETS, M.TARGET_MODES, strict=True)
        },
        "profile_cross_author_sum5": float(dots[1, 1, 0, M.TARGET_MODES].sum()),
        "joint_cross_author_sum5": float(dots[0, 0, 0, M.TARGET_MODES].sum()),
        "native_pair_score_diagonals": diag,
        "operator_addition_max_error": float(
            np.max(np.abs(scores[:, 0] - scores[:, 1] - scores[:, 2]))
        ),
        "space_addition_max_error": float(
            np.max(np.abs(scores[:, :, 0] - scores[:, :, 1:].sum(2)))
        ),
        "elapsed_seconds": time.perf_counter() - start,
        "psychological_prediction_accuracy": False,
        "new_neural_API_private_training": 0,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "analysis.npz",
            "source_operator_accounts.csv",
            "pairing_references.csv",
            "native_contacts.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def verify(out):
    """Use literal small coordinate tensors and event-formula replay once."""
    R.check(out, ("built_seal.json", "analyzed_seal.json"))
    start, z, result = (
        time.perf_counter(),
        arrays(out / "observations.npz"),
        arrays(out / "analysis.npz"),
    )
    bs, q = banks(out)
    error, checks = 0.0, 0
    for bank in bs:
        tiny = {
            k: v[..., [0, 17, 91, 200]]
            if k.endswith(("_a", "_b")) or k == "centers"
            else v
            for k, v in bank.items()
        }
        for component in M.COMPONENTS:
            aa, bb, w = M.sides(tiny, component, "full", np.eye(4))
            t = np.array([aa.T @ (row[:, None] * bb) for row in w])
            x, y = np.array([0.2, -0.1, 0.4, 0.05]), np.array([-0.2, 0.5, 0.1, -0.3])
            literal = np.einsum(
                "i,mij,j->m", x - tiny["centers"][0], t, y - tiny["centers"][1]
            )
            error = max(
                error,
                float(
                    np.max(
                        np.abs(
                            literal - M.read(tiny, component, "full", x, y, np.eye(4))
                        )
                    )
                ),
            )
            error = max(
                error,
                float(
                    np.max(
                        np.abs(
                            np.sum(t * t, axis=(1, 2))
                            - M.contraction(
                                tiny, tiny, component, component, "full", np.eye(4)
                            )
                        )
                    )
                ),
            )
            checks += 62
    expected = (
        result["contractions"][1, 1]
        + result["contractions"][1, 2]
        + result["contractions"][2, 1]
        + result["contractions"][2, 2]
    )
    error = max(error, float(np.max(np.abs(expected - result["contractions"][0, 0]))))
    frames = panel_frames(z)
    record = {
        "frame_key": "actual-cold-panel-0",
        "windows": [
            [
                {
                    "event_id": f"{z['author_ids'][0]}:{view}:{j}",
                    "vector": z["panel_vectors"][0, view, j].tolist(),
                }
                for j in range(12)
            ]
            for view in (0, 1)
        ],
    }
    cold = M.cold(bs[0], q, record)
    literal = np.stack(
        [
            M.read(bs[0], c, "full", frames[0, 0, 0], frames[0, 1, 0], q)
            for c in M.COMPONENTS
        ]
    )
    error = max(
        error, float(np.max(np.abs(np.asarray(cold["operator_readings"]) - literal)))
    )
    field_checks = 0
    for bank, component, space, view in np.ndindex(2, 3, 5, 2):
        other = frames[0, 1 - view, 0]
        field = M.field(
            bs[bank], M.COMPONENTS[component], M.SPACES[space], other, view, q
        )
        for operation, indices in enumerate(((1, 2), (3, 4))):
            delta = frames[0, view, indices[0]] - frames[0, view, indices[1]]
            expected_action = field @ delta
            observed = result["actions"][bank, component, space, 0, 0, view, operation]
            error = max(error, float(np.max(np.abs(expected_action - observed))))
            field_checks += 31
    if (
        max(
            error,
            R.read(out / "summary.json")["operator_addition_max_error"],
            R.read(out / "summary.json")["space_addition_max_error"],
        )
        > 1e-12
    ):
        raise ValueError("Finite operator accounting failed;no precision rescue")
    R.write(out / "cold_example.json", record)
    receipt = {
        "PASS": True,
        "alternate_max_error": error,
        "literal_coordinate_checks": checks,
        "finite_field_action_checks": field_checks,
        "same_agent_not_independent": True,
        "science_rerun": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verified_seal.json", ("cold_example.json", "verification.json"))
    print(json.dumps(receipt))


def main():
    """Run one explicit stage; existing outputs cannot be silently overwritten."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "build", "analyze", "verify", "cold")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "cold":
        R.check(
            args.out, ("built_seal.json", "analyzed_seal.json", "verified_seal.json")
        )
        bs, q = banks(args.out)
        result = M.cold(bs[0], q, R.read(args.out / "cold_example.json"))
        print(
            json.dumps(
                {
                    "frame_key": result["frame_key"],
                    "five_target_pair_readings": np.asarray(
                        result["operator_readings"]
                    )[2, M.TARGET_MODES].tolist(),
                    "input_native_answers_or_labels": False,
                }
            )
        )
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
