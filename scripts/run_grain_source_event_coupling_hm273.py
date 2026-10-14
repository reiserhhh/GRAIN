#!/usr/bin/env python3
"""One source-observed paired-vector construction, not a precision search."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_event_anchor_field as A
from grain_research.construction import math_linked_join as J
from grain_research.construction import math_source_event_coupling as M
from grain_research.construction.math_relational_diagnostic_update import BITS
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_event_anchor_field_hm267 import selected
from scripts.score_grain_source_event_pair_cold import score

CONFIG = "configs/grain_source_event_coupling_hm273.json"
DEFAULT = ROOT / "results/grain_math_response_v1/source_event_coupling_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_SOURCE_EVENT_COUPLING_PLAN_HM273.md",
    "grain_research/construction/math_source_event_coupling.py",
    "grain_research/construction/math_event_anchor_field.py",
    "grain_research/construction/math_linked_join.py",
    "grain_research/construction/math_relational_diagnostic_update.py",
    "scripts/run_grain_source_event_coupling_hm273.py",
    "scripts/score_grain_source_event_pair_cold.py",
    "scripts/run_grain_event_anchor_field_hm267.py",
    "scripts/run_grain_observed_family_hm170.py",
    "tests/test_grain_source_event_coupling.py",
)


def load(path):
    """Read only this packet's own declared numerical artifact."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prepare(out):
    """Qualify original-slot chronology and actual native event addresses once."""
    if out.exists():
        raise FileExistsError("Do not replace frozen or failed science")
    cfg = R.read(ROOT / CONFIG)
    paths = {
        k: ROOT / cfg[k + "_parent"]
        for k in ("source", "chronology", "identity", "panel", "join", "reader")
    }
    stages = {
        "source": ("computed_seal.json", "verification_seal.json"),
        "chronology": ("compiled_seal.json", "analysis_seal.json"),
        "identity": ("computed_seal.json", "verification_seal.json"),
        "panel": (),
        "join": ("computed_seal.json", "verification_seal.json"),
        "reader": (
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        ),
    }
    for name, path in paths.items():
        R.check(path, stages[name])
    source = selected(
        paths["source"] / "source_bundle.npz",
        ("source_comments", "codes", "variance", "directions"),
    )
    indices = selected(
        paths["source"] / "observations.npz", ("source_indices", "panel_indices")
    )
    identity = selected(
        paths["identity"] / "observations.npz", ("source_indices", "codes", "folds")
    )
    chronology = selected(
        paths["chronology"] / "observations.npz",
        ("times", "conditions", "original_units"),
    )
    panel = selected(
        paths["panel"] / "observations.npz", ("panel_vectors", "author_ids")
    )
    joined = selected(paths["join"] / "observations.npz", ("points",))
    readers = load(paths["reader"] / "stage_actions.npz")
    bindings = R.read(paths["join"] / "event_bindings.json")
    if (
        source["source_comments"].shape != (386, 24, 1024)
        or panel["panel_vectors"].shape != (24, 2, 12, 1024)
        or joined["points"].shape != (24, 2, 2, 1024)
        or not np.array_equal(indices["source_indices"], identity["source_indices"])
        or not np.array_equal(source["codes"], identity["codes"])
        or np.intersect1d(indices["source_indices"], indices["panel_indices"]).size
        or not np.isclose(
            float(source["variance"]), cfg["variance"], rtol=0, atol=1e-16
        )
        or set(np.unique(identity["folds"])) != {0, 1}
        or len(readers) != 18
        or any(v.shape != (24, 2, 5) for v in readers.values())
    ):
        raise ValueError("Changed source, historical banks or native-reader scope")
    inverse = np.argsort(chronology["original_units"], axis=1)
    original = {
        k: np.take_along_axis(chronology[k], inverse, axis=1)[:, :24]
        for k in ("times", "conditions")
    }
    st = original["times"][indices["source_indices"]]
    sc = original["conditions"][indices["source_indices"]]
    pt = original["times"][indices["panel_indices"]]
    pc = original["conditions"][indices["panel_indices"]]
    left, right, relink = M.pair_indices(st, sc)
    pl, pr, _ = M.pair_indices(pt, pc)
    np.testing.assert_array_equal(left, pl)
    np.testing.assert_array_equal(right, pr)
    z = source["source_comments"]
    p = panel["panel_vectors"].reshape(24, 24, 1024)
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    if not np.isfinite(p).all() or np.max(abs(np.linalg.norm(z, axis=-1) - 1)) > 1e-12:
        raise ValueError("Invalid source/panel vector normalization")
    for u, row in enumerate(bindings):
        for w, event in enumerate(row["events"]):
            np.testing.assert_allclose(
                p[u, w * 12 + event["own_slot"]],
                joined["points"][u, w, 0],
                atol=1e-12,
                rtol=0,
            )
            np.testing.assert_allclose(
                p[row["donor_author"], w * 12 + event["foreign_slot"]],
                joined["points"][u, w, 1],
                atol=1e-12,
                rtol=0,
            )
    fold = identity["folds"]
    null_codes = source["codes"].copy()
    counts = {}
    for name, mask in (
        ("bank0", fold == 0),
        ("bank1", fold == 1),
        ("pooled", np.ones(386, bool)),
    ):
        count = np.bincount(source["codes"][mask], minlength=32)
        if count.min() < 1:
            raise ValueError("Every bank/profile needs observed source support")
        counts[name] = {"authors": int(mask.sum()), "profile_counts": count.tolist()}
    for bank in (0, 1):
        ix = np.flatnonzero(fold == bank)
        null_codes[ix] = np.roll(source["codes"][ix], 1)
    primary = np.stack((p[:, left], p[:, right]), axis=2).reshape(528, 2, 1024)
    secondary = J.corners(joined["points"]).reshape(192, 2, 1024)
    prototypes = np.stack((z[:, left], z[:, right]), axis=2)
    example = {
        "frame_key": "historical-panel-0-cached-edge-0",
        "sampling_regime": "next_cached_record_within_registered_window",
        "events": [
            {
                "event_id": f"numeric-cache:{int(indices['panel_indices'][0])}:{s}",
                "vector": p[0, s].tolist(),
                "time": float(pt[0, s]),
                "window": 0,
                "condition": int(pc[0, s]),
            }
            for s in (int(left[0]), int(right[0]))
        ],
    }
    inputs = [path / "manifest.json" for path in paths.values()]
    inputs += [path / "prepared_seal.json" for path in paths.values()]
    inputs += [path / s for name, path in paths.items() for s in stages[name]]
    inputs += [
        paths[k] / f
        for k, f in (
            ("source", "source_bundle.npz"),
            ("source", "observations.npz"),
            ("identity", "observations.npz"),
            ("chronology", "observations.npz"),
            ("panel", "observations.npz"),
            ("join", "observations.npz"),
            ("join", "event_bindings.json"),
            ("reader", "stage_actions.npz"),
        )
    ]
    out.mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "source_pair_bank.npz",
        prototypes=prototypes,
        codes=source["codes"],
        null_codes=null_codes,
        folds=fold,
        relink=relink,
        variance=source["variance"],
        projection=A.projection(source["directions"]),
        source_indices=indices["source_indices"],
        times=st,
        conditions=sc,
        left=left,
        right=right,
    )
    R.archive(
        out / "qualified_panel_pairs.npz",
        primary=primary,
        secondary=secondary,
        windows=np.tile(left // 12, 24),
        authors=np.repeat(np.arange(24), 22),
        panel_indices=indices["panel_indices"],
        times=pt,
        conditions=pc,
    )
    R.archive(out / "reader_actions.npz", **readers)
    R.write(out / "example_pair.json", example)
    R.write(
        out / "input_quality.json",
        {
            "source_authors": 386,
            "observed_source_pairs": 8492,
            "source_pairs_per_author": 22,
            "application_authors": 24,
            "application_adjacent_pairs": 528,
            "strict_source_and_panel_chronology": True,
            "source_bank_support": counts,
            "right_relink_mobile_pairs": int(np.count_nonzero(relink != np.arange(22))),
            "right_relink_total_pairs": int(relink.size),
            "overlapping_pairs_dependent": True,
            "new_neural_private_API_training_calls": 0,
            "event_ids_are_opaque_numeric_cache_addresses_not_natural_text_identity_proof": True,
            "secondary_regime": "old_nonadjacent_cross_window_dyads_not_source_adjacency",
            "no_application_labels_or_answers_read": True,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "source_pair_bank.npz",
            "qualified_panel_pairs.npz",
            "reader_actions.npz",
            "example_pair.json",
            "input_quality.json",
        ),
    )
    print(
        "PREPARED HM273 386 source authors / 528 qualified adjacent panel pairs",
        flush=True,
    )


def field_summary(a, b):
    """Describe signed bank recurrence; neither correlation significance nor accuracy."""
    cross = np.mean(a * b, axis=0)
    ra, rb = np.sqrt(np.mean(a * a, axis=0)), np.sqrt(np.mean(b * b, axis=0))
    return {
        "cross_bank_product": cross.tolist(),
        "cosine": [
            float(c / (x * y)) if x * y > 0 else None
            for c, x, y in zip(cross, ra, rb, strict=True)
        ],
        "same_sign_pair_fraction": np.mean(a * b > 0, axis=0).tolist(),
        "bank_rms": [ra.tolist(), rb.tolist()],
    }


def compute(out):
    """Run all fixed laws and assignments once; preserve signed cross terms."""
    cfg = R.check(out)["config"]
    R.write(
        out / "compute_started.json",
        {"state": "RUNNING", "seconds_cap": cfg["compute_seconds_cap"]},
    )
    began = time.perf_counter()
    try:
        bank, points, readers = [
            load(out / p)
            for p in (
                "source_pair_bank.npz",
                "qualified_panel_pairs.npz",
                "reader_actions.npz",
            )
        ]
        queries = np.concatenate((points["primary"], points["secondary"]))
        arrays, kernels, recurrence_rows, contact_rows, descriptions, matched = (
            {},
            {},
            [],
            [],
            {},
            {},
        )
        for space in cfg["spaces"]:
            q = bank["projection"]
            x, z = (
                (queries @ q, bank["prototypes"] @ q)
                if space == "GRAIN15"
                else (queries, bank["prototypes"])
            )
            logs = M.event_logs(x, z, float(bank["variance"]))
            kernels[space] = logs
            for assignment in cfg["assignments"]:
                codes = bank["codes"] if assignment == "actual" else bank["null_codes"]
                group = f"{assignment}/{space}"
                descriptions[group], matched[group] = {}, {}
                for name, mask in (
                    ("bank0", bank["folds"] == 0),
                    ("bank1", bank["folds"] == 1),
                    ("pooled", np.ones(386, bool)),
                ):
                    reading = M.profile_readings(
                        logs[:, mask], codes[mask], bank["relink"][mask]
                    )
                    key = f"{group}/{name}"
                    for field in ("profile", "targets", "log_odds", "log_laws"):
                        arrays[key + "/" + field] = reading[field]
                f0, f1 = [
                    M.components(arrays[f"{group}/{name}/targets"][:528])
                    for name in ("bank0", "bank1")
                ]
                for window, mask in (
                    ("0", points["windows"] == 0),
                    ("1", points["windows"] == 1),
                    ("pooled", np.ones(528, bool)),
                ):
                    descriptions[group][window] = {
                        c: field_summary(f0[c][mask], f1[c][mask]) for c in M.COMPONENTS
                    }
                    for c0, a in f0.items():
                        for c1, b in f1.items():
                            contact = a[mask].T @ b[mask] / mask.sum()
                            for i, target0 in enumerate(A.TARGETS):
                                for j, target1 in enumerate(A.TARGETS):
                                    recurrence_rows.append(
                                        {
                                            "assignment": assignment,
                                            "space": space,
                                            "window": window,
                                            "bank0_component": c0,
                                            "bank1_component": c1,
                                            "bank0_target": target0,
                                            "bank1_target": target1,
                                            "cross_product": float(contact[i, j]),
                                            "dependent_pairs": int(mask.sum()),
                                        }
                                    )
                values = arrays[group + "/pooled/targets"][528:].reshape(24, 2, 4, 4, 5)
                mixed = (
                    values[:, :, 0]
                    - values[:, :, 1]
                    - values[:, :, 2]
                    + values[:, :, 3]
                )
                actions = {
                    name: mixed[:, :, i] for i, name in enumerate(M.LAWS)
                } | M.components(mixed)
                for component, action in actions.items():
                    arrays[group + "/mixed/" + component] = action
                    matched[group][component] = {}
                    for observer, native in readers.items():
                        matched[group][component][observer] = []
                        for order in (0, 1):
                            contact = action[:, order].T @ native[:, order] / 24
                            matched[group][component][observer].append(
                                np.diag(contact).tolist()
                            )
                            for i, target in enumerate(A.TARGETS):
                                for j, behavior in enumerate(A.TARGETS):
                                    contact_rows.append(
                                        {
                                            "assignment": assignment,
                                            "space": space,
                                            "component": component,
                                            "order": order,
                                            "observer": observer,
                                            "source_target": target,
                                            "behavior_target": behavior,
                                            "contact": float(contact[i, j]),
                                            "regime": "cross_regime_not_calibrated_adjacent_pair_posterior",
                                        }
                                    )
                if time.perf_counter() - began > cfg["compute_seconds_cap"]:
                    raise TimeoutError("No cap expansion or parameter rescue")
        R.archive(out / "conditional_profiles.npz", **arrays)
        R.archive(out / "log_kernel_accounts.npz", **kernels)
        R.table(out / "signed_recurrence.csv", recurrence_rows)
        R.table(out / "all_cross_regime_contacts.csv", contact_rows)
        elapsed = time.perf_counter() - began
        if elapsed > cfg["compute_seconds_cap"]:
            raise TimeoutError("Finite compute cap exceeded including artifact writes")
        R.write(
            out / "summary.json",
            {
                "recurrence": descriptions,
                "cross_regime_matched_contacts": matched,
                "input_quality": R.read(out / "input_quality.json"),
                "scientific_CPU_seconds": elapsed,
                "recurrence_rows": len(recurrence_rows),
                "cross_regime_contact_rows": len(contact_rows),
                "not_accuracy_r_AUC_causal_response_or_psychological_fraction": True,
                "no_fit_grid_new_model_or_application_labels": True,
                "original_scope_complete": False,
            },
        )
        R.write(
            out / "runtime.json",
            {
                "state": "COMPLETE",
                "scientific_CPU_seconds": elapsed,
                "new_model_calls": 0,
            },
        )
        R.seal(
            out,
            "computed_seal.json",
            (
                "conditional_profiles.npz",
                "log_kernel_accounts.npz",
                "signed_recurrence.csv",
                "all_cross_regime_contacts.csv",
                "summary.json",
                "runtime.json",
            ),
        )
        print(
            json.dumps(descriptions["actual/E5"]["pooled"], allow_nan=False), flush=True
        )
    except Exception as exc:
        R.write(
            out / "runtime.json",
            {
                "state": "FAILED",
                "error": repr(exc),
                "scientific_CPU_seconds": time.perf_counter() - began,
            },
        )
        raise


def literal_profile(logs, codes, relink):
    """Alternate direct positive-pair arithmetic, without the log-law helper."""
    weights = np.exp(logs)
    a, b = weights[:, :, 0], weights[:, :, 1]
    means_a, means_b = a.mean(-1), b.mean(-1)
    f = (a * b).mean(-1)
    j = means_a * means_b
    rr = (a * np.take_along_axis(b, relink[None], -1)).mean(-1)
    laws = []
    for code in range(32):
        ix = codes == code
        laws.append(
            np.stack(
                (
                    f[:, ix].mean(-1),
                    j[:, ix].mean(-1),
                    means_a[:, ix].mean(-1) * means_b[:, ix].mean(-1),
                    rr[:, ix].mean(-1),
                ),
                -1,
            )
            * ix.mean()
        )
    weighted = np.stack(laws, -1)
    return weighted / weighted.sum(-1, keepdims=True)


def verify(out):
    """Recompute all finite laws/contacts and cold CLI, with direct kernel spots."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    bank, points, readers, arrays, kernels = [
        load(out / p)
        for p in (
            "source_pair_bank.npz",
            "qualified_panel_pairs.npz",
            "reader_actions.npz",
            "conditional_profiles.npz",
            "log_kernel_accounts.npz",
        )
    ]
    error, checks, direct_error = 0.0, 0, 0.0
    queries = np.concatenate((points["primary"], points["secondary"]))
    for space in cfg["spaces"]:
        q = bank["projection"]
        x, z = (
            (queries @ q, bank["prototypes"] @ q)
            if space == "GRAIN15"
            else (queries, bank["prototypes"])
        )
        for v in (0, 1):
            for qi, ui, ti in (
                (0, 0, 0),
                (527, 385, 21),
                (528, 203, 11),
                (719, 182, 10),
            ):
                value = -np.sum((x[qi, v] - z[ui, ti, v]) ** 2) / (
                    2 * float(bank["variance"])
                )
                direct_error = max(
                    direct_error, abs(float(value - kernels[space][qi, ui, v, ti]))
                )
        for assignment in cfg["assignments"]:
            codes = bank["codes"] if assignment == "actual" else bank["null_codes"]
            for name, mask in (
                ("bank0", bank["folds"] == 0),
                ("bank1", bank["folds"] == 1),
                ("pooled", np.ones(386, bool)),
            ):
                profiles = literal_profile(
                    kernels[space][:, mask], codes[mask], bank["relink"][mask]
                )
                key = f"{assignment}/{space}/{name}"
                error = max(
                    error,
                    float(np.max(abs(profiles - arrays[key + "/profile"]))),
                    float(np.max(abs(profiles @ BITS - arrays[key + "/targets"]))),
                )
                checks += profiles.size
    recurrence_error, contact_error, telescope_error = 0.0, 0.0, 0.0
    with (out / "signed_recurrence.csv").open() as f:
        for row in csv.DictReader(f):
            key = row["assignment"] + "/" + row["space"]
            a, b = [
                M.components(arrays[f"{key}/{n}/targets"][:528])
                for n in ("bank0", "bank1")
            ]
            mask = (
                np.ones(528, bool)
                if row["window"] == "pooled"
                else points["windows"] == int(row["window"])
            )
            i, j = (
                A.TARGETS.index(row["bank0_target"]),
                A.TARGETS.index(row["bank1_target"]),
            )
            value = sum(
                float(x * y)
                for x, y in zip(
                    a[row["bank0_component"]][mask, i],
                    b[row["bank1_component"]][mask, j],
                    strict=True,
                )
            ) / int(mask.sum())
            recurrence_error = max(
                recurrence_error, abs(value - float(row["cross_product"]))
            )
    with (out / "all_cross_regime_contacts.csv").open() as f:
        for row in csv.DictReader(f):
            key = f"{row['assignment']}/{row['space']}/mixed/{row['component']}"
            order = int(row["order"])
            i, j = (
                A.TARGETS.index(row["source_target"]),
                A.TARGETS.index(row["behavior_target"]),
            )
            value = (
                sum(
                    float(x * y)
                    for x, y in zip(
                        arrays[key][:, order, i],
                        readers[row["observer"]][:, order, j],
                        strict=True,
                    )
                )
                / 24
            )
            contact_error = max(contact_error, abs(value - float(row["contact"])))
    for key in (
        f"{a}/{s}/pooled/targets" for a in cfg["assignments"] for s in cfg["spaces"]
    ):
        c = M.components(arrays[key])
        telescope_error = max(
            telescope_error,
            float(
                np.max(
                    abs(
                        c["pair_component"]
                        + c["author_component"]
                        - c["total_component"]
                    )
                )
            ),
        )
    expected = score(bank, R.read(out / "example_pair.json"))
    cli = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_source_event_pair_cold.py"),
            "--bundle",
            str(out / "source_pair_bank.npz"),
            "--record",
            str(out / "example_pair.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if json.loads(cli.stdout) != expected:
        raise ValueError("Standalone cold CLI changed numerical meanings")
    cold_error = 0.0
    for space in cfg["spaces"]:
        for i, law in enumerate(M.LAWS):
            cold_error = max(
                cold_error,
                float(
                    np.max(
                        abs(
                            np.array(expected["spaces"][space]["laws"][law]["targets"])
                            - arrays[f"actual/{space}/pooled/targets"][0, i]
                        )
                    )
                ),
            )
    if (
        max(
            error,
            direct_error,
            recurrence_error,
            contact_error,
            telescope_error,
            cold_error,
        )
        > 1e-11
    ):
        raise ValueError(
            "Finite law identity failed; do not rerun the frozen experiment"
        )
    R.write(out / "cold_readback.json", expected)
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "alternate_profile_values": checks,
            "alternate_profile_and_target_error": error,
            "direct_full_vector_kernel_spots": 16,
            "direct_kernel_log_error": direct_error,
            "literal_recurrence_error": recurrence_error,
            "literal_contact_error": contact_error,
            "normalized_telescope_error": telescope_error,
            "cold_CLI_and_python_equal": True,
            "cold_primary_readback_error": cold_error,
            "audit_same_agent_not_blind": True,
            "all_negative_and_null_cells_retained": True,
            "original_scope_complete": False,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json", "cold_readback.json"))
    print(json.dumps(R.read(out / "verification.json")), flush=True)


def main():
    """Expose immutable prepare, one compute, read-only verification and seal check."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage == "check":
        R.check(args.out, ("computed_seal.json", "verification_seal.json"))
        print("HM273 registered input/output seals PASS")
    else:
        {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](args.out)


if __name__ == "__main__":
    main()
