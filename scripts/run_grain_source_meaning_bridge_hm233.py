#!/usr/bin/env python3
"""Join frozen GRAIN source meanings to SAME-event crossed neural actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observed_family as F
from grain_research.construction import math_source_meaning_bridge as M
from grain_research.construction import math_trait_anchor as A
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_source_meaning_bridge_hm233.json"
DEFAULT = ROOT / "results/grain_math_response_v1/source_meaning_bridge_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_SOURCE_MEANING_BRIDGE_PLAN_HM233.md",
    "grain_research/construction/math_source_meaning_bridge.py",
    "grain_research/construction/math_observed_family.py",
    "grain_research/construction/math_trait_anchor.py",
    "scripts/run_grain_source_meaning_bridge_hm233.py",
    "scripts/run_grain_observed_family_hm170.py",
    "tests/test_grain_source_meaning_bridge.py",
)
PARTS = ("full", "choice", "within", "unique", "shared")


def selected(path, keys):
    """Read only authorized numeric keys, never questionnaire/application labels."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def parent_seal(parent, seal, names):
    """Authenticate selected completed derivatives without rerunning any parent."""
    hashes = R.read(parent / seal)
    for name in names:
        if name not in hashes or R.digest(parent / name) != hashes[name]:
            raise ValueError("Changed or unsealed parent derivative: " + name)


def check(out, stages=()):
    """Verify frozen inputs, code and completed output stages."""
    manifest = R.read(out / "manifest.json")
    for kind in ("code", "inputs"):
        for name, digest in manifest[kind].items():
            if R.digest(ROOT / name) != digest:
                raise ValueError("Changed scientific input/code: " + name)
    for seal in ("prepared_seal.json",) + tuple(stages):
        for name, digest in R.read(out / seal).items():
            if R.digest(out / name) != digest:
                raise ValueError("Changed sealed derivative: " + name)
    return manifest


def prepare(out):
    """Bind laws and exact events before any new contact or geometry outcome."""
    cfg = R.read(ROOT / CONFIG)
    evidence, action, meaning = [
        ROOT / cfg[k] for k in ("evidence_parent", "action_parent", "meaning_parent")
    ]
    parent_seal(evidence, "prepared_seal.json", ("observations.npz", "selections.json"))
    parent_seal(evidence, "analyzed_seal.json", ("analysis.npz",))
    parent_seal(action, "prepared_seal.json", ("bindings.json", "units.json"))
    parent_seal(action, "analyzed_seal.json", ("analysis.npz",))
    parent_seal(meaning, "prepared_seal.json", ("observations.npz",))
    parent_seal(meaning, "compiled_seal.json", tuple(cfg["laws"]))
    z = selected(
        evidence / "observations.npz",
        ("panel_indices", "source_indices", "vectors", "conditions"),
    )
    old = selected(
        meaning / "observations.npz",
        ("application_indices", "source_indices", "source_mass"),
    )
    if (
        not np.array_equal(z["source_indices"], old["source_indices"])
        or set(z["panel_indices"]) != set(old["application_indices"])
        or np.intersect1d(z["panel_indices"], z["source_indices"]).size
        or len(z["source_indices"]) != cfg["source_authors"]
        or z["vectors"].shape != (cfg["authors"], cfg["events_per_author"], 1024)
    ):
        raise ValueError("SOURCE/application identities or full event scope disagree")
    bundle = R.read(ROOT / cfg["anonymous_bundle"])
    directions = np.asarray(bundle["directions"], float)
    if directions.shape != (1024, 15) or not np.isfinite(directions).all():
        raise ValueError("Frozen GRAIN15 directions required")
    if (
        hashlib.sha256(np.ascontiguousarray(directions).tobytes()).hexdigest()
        != bundle["directions_sha256"]
    ):
        raise ValueError("Direction identity mismatch")
    norms = np.linalg.norm(z["vectors"], axis=-1)
    if not np.allclose(norms, 1, rtol=0, atol=2e-12):
        raise ValueError("Exactly inherited normalized comment vectors required")
    coordinates = z["vectors"] @ directions
    selections = R.read(evidence / "selections.json")
    bindings = R.read(action / "bindings.json")
    if len(selections) != 24 or len(bindings) != 24:
        raise ValueError("Complete SAME-event binding required")
    laws = [R.read(meaning / name) for name in cfg["laws"]]
    for law in laws:
        A.verify_law(law)
        if (
            law["source_authors"] != 386
            or np.max(
                abs(np.asarray(law["reference_mass"]) - old["source_mass"].mean(0))
            )
            > 2e-12
        ):
            raise ValueError("SOURCE reference must match excluded author bank")
    if laws[0]["coordinate_cutoffs"] != laws[1]["coordinate_cutoffs"]:
        raise ValueError("Null changes profile assignment, not observation geometry")
    cells = 4 * z["conditions"] + F.states(
        coordinates.reshape(-1, 15), laws[0]["coordinate_cutoffs"]
    ).reshape(24, 12)
    masks = np.ones((24, 3, 12), bool)
    masses = np.empty((24, 3, 8))
    condition_matched = 0
    for u, (sel, binding) in enumerate(zip(selections, bindings, strict=True)):
        if (
            sel["author"] != u
            or binding["author"] != u
            or sel["original_author_index"] != int(z["panel_indices"][u])
            or binding["original_author_index"] != int(z["panel_indices"][u])
            or sel["event_ids"] != binding["event_ids"]
            or len(set(binding["body_hashes"])) != 12
        ):
            raise ValueError(
                "Author order or actual publication event identity changed"
            )
        s, c = np.asarray(sel["support"], int), np.asarray(sel["counter"], int)
        if (
            s.shape != (4,)
            or c.shape != (4,)
            or len(set(np.r_[s, c])) != 8
            or not np.isin(np.r_[s, c], np.arange(12)).all()
        ):
            raise ValueError("Original disjoint four-event edits required")
        if not np.array_equal(
            np.bincount(z["conditions"][u, s], minlength=2),
            np.bincount(z["conditions"][u, c], minlength=2),
        ):
            raise ValueError("Historical condition matching changed")
        condition_matched += 1
        masks[u, 1, s], masks[u, 2, c] = False, False
        for arm in range(3):
            ix = np.flatnonzero(masks[u, arm])
            masses[u, arm] = M.event_mass(
                cells[u, ix], [sel["event_ids"][i] for i in ix]
            )
    units = R.read(action / "units.json")
    if len(units) != 64:
        raise ValueError("Full oriented comparison graph required")
    edges = []
    for i in range(32):
        fwd, rev = units[2 * i : 2 * i + 2]
        if (
            fwd["edge"] != i
            or rev["edge"] != i
            or fwd["order"] != 0
            or rev["order"] != 1
            or (fwd["left"], fwd["right"]) != (rev["right"], rev["left"])
        ):
            raise ValueError("Original crossed edge orientation changed")
        edges.append((fwd["left"], fwd["right"]))
    files = [cfg["anonymous_bundle"]]
    for parent, names in (
        (
            evidence,
            (
                "observations.npz",
                "selections.json",
                "analysis.npz",
                "prepared_seal.json",
                "analyzed_seal.json",
            ),
        ),
        (
            action,
            (
                "analysis.npz",
                "bindings.json",
                "units.json",
                "prepared_seal.json",
                "analyzed_seal.json",
            ),
        ),
        (
            meaning,
            (
                "observations.npz",
                "source_law.json",
                "permuted_law.json",
                "prepared_seal.json",
                "compiled_seal.json",
            ),
        ),
    ):
        files.extend(str((parent / n).relative_to(ROOT)) for n in names)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {p: R.digest(ROOT / p) for p in files},
            "cohort": "historical geometry,17 representation-overlap authors;target orientation excludes24;not fresh holdout",
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        cells=cells,
        masks=masks,
        masses=masses,
        coordinates=coordinates,
        conditions=z["conditions"],
        edges=np.asarray(edges, int),
        source_indices=z["source_indices"],
        application_indices=z["panel_indices"],
    )
    R.write(out / "source_fields.json", {"actual": laws[0], "permuted": laws[1]})
    R.write(out / "bindings.json", bindings)
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "unique_events": 288,
            "source_authors": 386,
            "edges": 32,
            "actual_source_overlap": 0,
            "historical_geometry_overlap": 17,
            "condition_matched_edits": condition_matched,
            "new_model_or_API_calls": 0,
            "application_label_reads": 0,
            "source_labels_reread": False,
            "source_profile_law_reused_not_refit": True,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "observations.npz",
            "source_fields.json",
            "bindings.json",
            "input_quality.json",
        ),
    )
    print(
        "HM233 prepared:SAME288events,386disjoint SOURCE,no new labels/model",
        flush=True,
    )


def score(out):
    """Freeze cold GRAIN scores BEFORE loading any native/action answer arrays."""
    check(out)
    p = selected(out / "observations.npz", ("masses",))["masses"]
    laws = R.read(out / "source_fields.json")
    arrays = {}
    for name, law in laws.items():
        f = M.fields(law["reference_mass"], law["probabilities"], law["prior"])
        scores = M.cold_scores(p.reshape(-1, 8), law["reference_mass"], f)
        for part in PARTS:
            arrays[f"{name}_{part}_field"] = f[part]
            arrays[f"{name}_{part}_score"] = scores[part].reshape(24, 3, 5)
    R.archive(out / "cold_scores.npz", **arrays)
    R.seal(out, "cold_seal.json", ("cold_scores.npz",))
    print(
        "HM233 cold scored:all fields/arms/targets retained,no model answers",
        flush=True,
    )


def analyze(out):
    """Inspect all signed connections, never choose a fitted predictor."""
    started = time.monotonic()
    cfg = check(out, ("cold_seal.json",))["config"]
    z = selected(out / "observations.npz", ("edges",))
    with np.load(out / "cold_scores.npz", allow_pickle=False) as archive:
        cold = {k: archive[k] for k in archive.files}
    action = selected(
        ROOT / cfg["action_parent"] / "analysis.npz",
        ("matrix", "interaction", "native", "native_increment"),
    )
    deletion = selected(
        ROOT / cfg["evidence_parent"] / "analysis.npz", ("raw", "specific")
    )
    actions = {
        "total": action["matrix"],
        "source_discrimination": action["matrix"]
        - action["matrix"].mean(2, keepdims=True),
        "reader_source_interaction": action["interaction"],
    }
    laws = R.read(out / "source_fields.json")
    rows, geom, direct, summary = [], {}, {}, {}
    for law_name, law in laws.items():
        r = np.asarray(law["reference_mass"])
        whole = cold[f"{law_name}_full_score"][:, 0]
        edge_whole = whole[z["edges"][:, 0]] - whole[z["edges"][:, 1]]
        geom[law_name] = {}
        for scope, values, weight in (
            ("source_fields", cold[f"{law_name}_full_field"], r),
            ("author_scores", whole - whole.mean(0), None),
            ("edge_scores", edge_whole, None),
        ):
            g = M.gram(values, weight)
            geom[law_name][scope] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in g.items()
            }
        for part in PARTS:
            scores = cold[f"{law_name}_{part}_score"]
            x = scores[:, 0][z["edges"][:, 0]] - scores[:, 0][z["edges"][:, 1]]
            for mode, values in actions.items():
                result = M.bridge(x, values)
                direct[f"{law_name}_{part}_{mode}"] = result["contact"]
                for t, q, s in np.ndindex(5, 5, 5):
                    rows.append(
                        {
                            "law": law_name,
                            "part": part,
                            "action": mode,
                            "meaning": cfg["targets"][t],
                            "reader": cfg["targets"][q],
                            "source": cfg["targets"][s],
                            "contact": float(result["contact"][t, q, s]),
                            "cosine": float(result["cosine"][t, q, s])
                            if result["defined"][t, q, s]
                            else None,
                        }
                    )
        full = cold[f"{law_name}_full_score"]
        change = full[:, 1] - full[:, 2]
        native_contact = edge_whole.T @ action["native"] / 32
        native_increment_contact = edge_whole.T @ action["native_increment"] / 32
        deletion_contact = change.T @ deletion["raw"] / 24
        direct[f"{law_name}_native"] = native_contact
        direct[f"{law_name}_native_increment"] = native_increment_contact
        direct[f"{law_name}_deletion"] = deletion_contact
        direct[f"{law_name}_deletion_scores"] = change
        summary[law_name] = {
            "source_field_rank": geom[law_name]["source_fields"]["rank"],
            "source_field_effective_rank": geom[law_name]["source_fields"][
                "effective_rank"
            ],
            "author_score_effective_rank": geom[law_name]["author_scores"][
                "effective_rank"
            ],
            "edge_score_effective_rank": geom[law_name]["edge_scores"][
                "effective_rank"
            ],
            "native_contact_diagonal": np.diag(native_contact).tolist(),
            "native_increment_contact_diagonal": np.diag(
                native_increment_contact
            ).tolist(),
            "deletion_contact_diagonal": np.diag(deletion_contact).tolist(),
            "deletion_choice_max": float(
                np.max(
                    abs(
                        cold[f"{law_name}_choice_score"][:, 1]
                        - cold[f"{law_name}_choice_score"][:, 2]
                    )
                )
            ),
            "source_discrimination_diagonal_contacts": [
                float(direct[f"{law_name}_full_source_discrimination"][q, q, q])
                for q in range(5)
            ],
            "unique_source_energy_ratio": (
                (r[:, None] * cold[f"{law_name}_unique_field"] ** 2).sum(0)
                / (r[:, None] * cold[f"{law_name}_full_field"] ** 2).sum(0)
            ).tolist(),
        }
    elapsed = time.monotonic() - started
    if elapsed > cfg["max_compute_seconds"]:
        raise RuntimeError("Finite CPU cap exceeded;no automatic retry")
    R.archive(out / "bridge_accounts.npz", **direct)
    R.table(out / "meaning_action_tensor.csv", rows)
    R.write(out / "geometry.json", geom)
    R.write(
        out / "summary.json",
        {
            "actual": summary["actual"],
            "permuted": summary["permuted"],
            "contact_rows": len(rows),
            "CPU_seconds": elapsed,
            "no_fit_or_accuracy_promotion": True,
            "one_old_permutation_not_pvalue": True,
            "edges_share_authors": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "bridge_accounts.npz",
            "meaning_action_tensor.csv",
            "geometry.json",
            "summary.json",
        ),
    )
    print(json.dumps(summary["actual"]), flush=True)


def verify(out):
    """Independently sum event contributions and every full tensor entry."""
    cfg = check(out, ("cold_seal.json", "analyzed_seal.json"))["config"]
    with np.load(out / "observations.npz", allow_pickle=False) as archive:
        z = {k: archive[k] for k in archive.files}
    with np.load(out / "cold_scores.npz", allow_pickle=False) as archive:
        cold = {k: archive[k] for k in archive.files}
    with np.load(out / "bridge_accounts.npz", allow_pickle=False) as archive:
        accounts = {k: archive[k] for k in archive.files}
    action = selected(
        ROOT / cfg["action_parent"] / "analysis.npz", ("matrix", "interaction")
    )
    modes = {
        "total": action["matrix"],
        "source_discrimination": action["matrix"]
        - action["matrix"].mean(2, keepdims=True),
        "reader_source_interaction": action["interaction"],
    }
    laws = R.read(out / "source_fields.json")
    event_error = additive_error = projection_error = tensor_error = 0.0
    checks = 0
    for name, law in laws.items():
        r = np.asarray(law["reference_mass"])
        for part in PARTS:
            f = cold[f"{name}_{part}_field"]
            score_array = cold[f"{name}_{part}_score"]
            for u, arm, q in np.ndindex(24, 3, 5):
                actual = np.mean(
                    [f[int(a), q] for a in z["cells"][u, z["masks"][u, arm]]]
                ) - sum(r[a] * f[a, q] for a in range(8))
                event_error = max(
                    event_error, abs(float(actual - score_array[u, arm, q]))
                )
            x = (
                score_array[:, 0][z["edges"][:, 0]]
                - score_array[:, 0][z["edges"][:, 1]]
            )
            for mode, t in modes.items():
                stored = accounts[f"{name}_{part}_{mode}"]
                for a, q, s in np.ndindex(5, 5, 5):
                    value = sum(float(x[e, a] * t[e, q, s]) for e in range(32)) / 32
                    tensor_error = max(tensor_error, abs(value - stored[a, q, s]))
                    checks += 1
        for component1, component2 in (("choice", "within"), ("unique", "shared")):
            additive_error = max(
                additive_error,
                float(
                    np.max(
                        abs(
                            cold[f"{name}_full_score"]
                            - cold[f"{name}_{component1}_score"]
                            - cold[f"{name}_{component2}_score"]
                        )
                    )
                ),
            )
        w, nu = cold[f"{name}_full_field"], cold[f"{name}_unique_field"]
        product = w.T @ (r[:, None] * nu)
        off = product - np.diag(np.diag(product))
        projection_error = max(projection_error, float(np.max(abs(off))))
    max_error = max(event_error, additive_error, projection_error, tensor_error)
    if max_error > 2e-12:
        raise ValueError("Alternate accounting failed;do not rescore or tune")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "event_sum_error": event_error,
            "component_sum_error": additive_error,
            "source_projection_error": projection_error,
            "tensor_loop_error": tensor_error,
            "tensor_entries_verified": checks,
            "same_agent_alternate_arithmetic_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("HM233 verified", max_error, checks, flush=True)


def main():
    """Run one immutable stage; never automatically repeat failed science."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "score", "analyze", "verify", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        check(
            args.out, ("cold_seal.json", "analyzed_seal.json", "verification_seal.json")
        )
        print("HM233 all seals PASS;original full scope remains OPEN")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
