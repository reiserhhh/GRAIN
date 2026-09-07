#!/usr/bin/env python3
"""Independent material, intervention and finite-composition replay for P1N."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    """Hash a saved source or output artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result, parent, model_dir):
    """Verify all evidence axes and reconstruct response interactions without model calls."""
    manifest = json.loads((result / "manifest.json").read_text())
    status = json.loads((result / "run_status.json").read_text())
    if status["status"] != "COMPLETE" or status["calls"] != 644:
        raise ValueError("Incomplete P1N")
    for directory in (result, parent):
        for name, digest in json.loads(
            (directory / "output_hashes.json").read_text()
        ).items():
            if sha(directory / name) != digest:
                raise ValueError("Artifact changed")
    if (
        manifest["parent_identity"]
        != json.loads((parent / "manifest.json").read_text())["identity"]
    ):
        raise ValueError("Wrong parent")
    with zipfile.ZipFile(result / "source_snapshot.zip") as z:
        for name, digest in manifest["code"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError("Source snapshot changed")
    data = json.loads((result / "materials.json").read_text())
    old = {
        r["case_id"]: r
        for r in json.loads((parent / "materials.json").read_text())["rows"]
    }
    layouts = json.loads((result / "token_layouts.json").read_text())
    donors = json.loads((result / "donors.json").read_text())
    base, effects = [
        pd.read_csv(result / (name + ".csv"), keep_default_na=False)
        for name in ("baselines", "effects")
    ]
    if (
        len(base) != 80
        or len(effects) != 560
        or base.case_id.duplicated().any()
        or effects.duplicated(["case_id", "patch", "intervention"]).any()
    ):
        raise ValueError("Incomplete/distinct score grid")
    if set(base.case_id) != set(effects.case_id) or set(base.case_id) != {
        r["case_id"] for r in data["rows"]
    }:
        raise ValueError("Case population differs")
    index = base.set_index("case_id")
    material_lookup = {r["case_id"]: r for r in data["rows"]}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    prefixes, states = {}, {}
    prefix_error = 0.0
    for r in data["rows"]:
        p = old[r["parent_case"]]
        if p["record_order"] != "original" or p["criterion"] == "venue":
            raise ValueError("Unplanned parent cell")
        expected = (
            p["evidence"]
            if r["room_sign"] == 1
            else p["evidence"].replace("The room was open.", "The room was closed.")
        )
        if (
            r["evidence"] != expected
            or r["user"] != expected + p["user"][len(p["evidence"]) :]
        ):
            raise ValueError("Changed non-room evidence/question")
        for k in (
            "level",
            "group_id",
            "entity",
            "criterion",
            "role_sign",
            "target_kind",
        ):
            if r[k] != p[k] or r[k] != index.loc[r["case_id"], k]:
                raise ValueError("Changed role/material metadata")
        truth = r["room_sign"] if r["target_kind"] == "object" else r["role_sign"]
        if (
            truth != r["expected_sign"]
            or truth != index.loc[r["case_id"], "expected_sign"]
        ):
            raise ValueError("Wrong expected truth")
        text = tok.apply_chat_template(
            [
                {"role": "system", "content": data["system"]},
                {"role": "user", "content": r["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        layout = layouts[r["case_id"]]
        start = text.index(r["user"])
        a, b = r["anchor_span"]
        n = [
            j
            for j, (lo, hi) in enumerate(enc["offset_mapping"])
            if start + a <= lo < hi <= start + b
        ]
        if (
            enc["input_ids"] != layout["ids"]
            or n != layout["anchor"]
            or len(n) != 1
            or r["user"][a:b].strip() != r["anchor_text"]
        ):
            raise ValueError("Token/anchor mismatch")
        value = np.load(result / "activations" / (r["case_id"] + ".npz"))["block_16"][n]
        states[r["case_id"]] = value
        key = (r["group_id"], r["level"], r["entity"], r["room_sign"])
        previous = prefixes.setdefault(key, (layout["ids"][: n[-1] + 1], value))
        if previous[0] != layout["ids"][: n[-1] + 1]:
            raise ValueError("Noncommon prefix")
        prefix_error = max(prefix_error, float(abs(value - previous[1]).max()))
        for label, change_b, change_r in (
            ("B", True, False),
            ("R", False, True),
            ("BR", True, True),
        ):
            d = material_lookup[donors[r["case_id"]][label]["full"]]
            s = material_lookup[donors[r["case_id"]][label]["anchor"]]
            if d["level"] != r["level"] * (-1 if change_b else 1) or d[
                "room_sign"
            ] != r["room_sign"] * (-1 if change_r else 1):
                raise ValueError("Wrong counterfactual axis")
            if (
                any(d[k] != r[k] for k in ("group_id", "entity", "criterion"))
                or any(
                    s[k] != d[k] for k in ("group_id", "entity", "level", "room_sign")
                )
                or s["criterion"]
                != ("event" if r["target_kind"] == "person" else "venue_object")
            ):
                raise ValueError("Changed intervention target")
            dl = layouts[d["case_id"]]
            if len(dl["ids"]) != len(layout["ids"]) or dl["span"] != layout["span"]:
                raise ValueError("Donor length/body layout")
            if any(
                layout["ids"][j] != dl["ids"][j]
                for j in range(len(layout["ids"]))
                if j not in layout["span"]
            ):
                raise ValueError("Donor modifies query")
    if len(prefixes) != 48 or prefix_error > 1e-5:
        raise ValueError("Prefix state accounting")
    arithmetic, metric, restore = 0.0, 0.0, 0.0
    for table in (base, effects):
        if not np.isfinite(
            table[["margin", "high_logit", "low_logit"]].to_numpy()
        ).all():
            raise ValueError("Nonfinite raw logits")
        arithmetic = max(
            arithmetic,
            float(abs(table.margin - table.high_logit + table.low_logit).max()),
        )
    for case, sub in effects.groupby("case_id"):
        if set(zip(sub.patch, sub.intervention)) != {
            ("anchor", i) for i in ("B", "R", "BR", "ADD")
        } | {("EQD", i) for i in ("B", "R", "BR")}:
            raise ValueError("Missing natural/composed patch")
        for v in sub.itertuples():
            if v.intervention == "ADD":
                if (
                    v.native_donor_case
                    or v.state_source_case != "synthetic_B_plus_R_minus_base"
                ):
                    raise ValueError("Synthetic state mislabeled as natural")
                continue
            expected = donors[case][v.intervention]
            if (
                v.native_donor_case != expected["full"]
                or v.state_source_case
                != expected["anchor" if v.patch == "anchor" else "full"]
            ):
                raise ValueError("Recorded source differs")
            if v.patch == "EQD":
                restore = max(
                    restore, abs(v.margin - index.loc[v.native_donor_case, "margin"])
                )
    geometry_error = 0.0
    for r in pd.read_csv(result / "anchor_geometry.csv").itertuples():
        d = donors[r.case_id]
        n0, nb, nr, nbr = (
            states[r.case_id],
            *(states[d[k]["anchor"]] for k in ("B", "R", "BR")),
        )
        additive = nb + nr - n0
        values = {
            "base_norm": np.linalg.norm(n0.astype(float)),
            "add_norm": np.linalg.norm(additive.astype(float)),
            "B_displacement": np.linalg.norm((nb - n0).astype(float)),
            "R_displacement": np.linalg.norm((nr - n0).astype(float)),
            "BR_displacement": np.linalg.norm((nbr - n0).astype(float)),
            "mixed_state_norm": np.linalg.norm((nbr - additive).astype(float)),
        }
        geometry_error = max(
            geometry_error, *(abs(v - getattr(r, k)) for k, v in values.items())
        )
    fresh = {}
    for case, sub in effects.groupby("case_id"):
        score = sub.set_index(["patch", "intervention"]).margin
        v = [np.longdouble(score["anchor", k]) for k in ("BR", "B", "R", "ADD")]
        zero = np.longdouble(index.loc[case, "margin"])
        fresh[case] = {
            "total": float(v[0] + zero - v[1] - v[2]),
            "state_departure": float(v[0] - v[3]),
            "readout_interaction": float(v[3] + zero - v[1] - v[2]),
            "native_total": float(
                np.longdouble(score["EQD", "BR"])
                + zero
                - np.longdouble(score["EQD", "B"])
                - np.longdouble(score["EQD", "R"])
            ),
        }
    interactions = pd.read_csv(result / "interactions.csv")
    if len(interactions) != 80 or interactions.case_id.duplicated().any():
        raise ValueError("Interaction grid")
    for r in interactions.itertuples():
        metric = max(
            metric,
            *(abs(v - getattr(r, k)) for k, v in fresh[r.case_id].items()),
            abs(r.total - r.state_departure - r.readout_interaction - r.identity_error),
        )
    for r in pd.read_csv(result / "interaction_energy.csv").itertuples():
        cases = [c for c in fresh if index.loc[c, "criterion"] == r.criterion]
        a, b, t, native = [
            np.array([fresh[c][k] for c in cases])
            for k in ("state_departure", "readout_interaction", "total", "native_total")
        ]
        values = {
            "state_sse": np.dot(a, a),
            "readout_sse": np.dot(b, b),
            "total_sse": np.dot(t, t),
            "cross_term": 2 * np.dot(a, b),
            "native_total_sse": np.dot(native, native),
        }
        metric = max(
            metric,
            *(abs(v - getattr(r, k)) for k, v in values.items()),
            abs(np.dot(t, t) - np.dot(a, a) - np.dot(b, b) - 2 * np.dot(a, b)),
        )
        if r.cases != len(cases):
            raise ValueError("Energy denominator")
    native_contrasts = pd.read_csv(result / "native_contrasts.csv")
    if len(native_contrasts) != 80:
        raise ValueError("Native contrast grid")
    for r in native_contrasts.itertuples():
        sub = base[
            (base.group_id == r.group_id)
            & (base.entity == r.entity)
            & (base.criterion == r.criterion)
        ]
        sub = sub[sub["room_sign" if r.axis == "B" else "level"] == r.fixed_value]
        values = sub.set_index("role_sign" if r.axis == "B" else "room_sign").margin
        metric = max(metric, abs(values[1] - values[-1] - r.contrast))
    for r in pd.read_csv(result / "native_summary.csv").itertuples():
        sub = base[(base.criterion == r.criterion) & (base.room_sign == r.room_sign)]
        metric = max(
            metric,
            abs(np.mean(sub.margin * sub.expected_sign > 0) - r.expected_preference),
        )
        if r.cases != len(sub):
            raise ValueError("Native count")
    for r in pd.read_csv(result / "effect_summary.csv").itertuples():
        sub = effects[
            (effects.criterion == r.criterion)
            & (effects.patch == r.patch)
            & (effects.intervention == r.intervention)
        ]
        shift = sub.margin - sub.case_id.map(index.margin)
        metric = max(
            metric,
            abs(np.sqrt(np.mean(shift**2)) - r.shift_rms),
            abs(shift.mean() - r.mean_shift),
        )
        if r.cases != len(sub):
            raise ValueError("Effect count")
        if r.relevant:
            sign = sub.room_sign if r.criterion == "venue_object" else sub.role_sign
            metric = max(
                metric, abs(np.mean(-sign * shift > 0) - r.expected_direction_fraction)
            )
    old_margin = pd.read_csv(parent / "baselines.csv").set_index("case_id").margin
    open_world = base[base.room_sign == 1]
    replay = float(
        abs(open_world.margin - open_world.parent_case.map(old_margin)).max()
    )
    if (
        len(open_world) != 40
        or max(arithmetic, metric, restore, replay) > 1e-8
        or geometry_error > 1e-4
    ):
        raise ValueError("Numerical replay failed")
    return {
        "status": "VERIFIED",
        "result_identity": manifest["identity"],
        "new_neural_calls": 0,
        "cases": 80,
        "arithmetic_max": arithmetic,
        "metric_and_identity_replay_max": metric,
        "geometry_fp32_vs_fp64_max": geometry_error,
        "full_restore_max": restore,
        "parent_replay_max": replay,
        "prefix_state_max": prefix_error,
        "interpretation": "Finite target-conditioned response/composition; additive anchor may be off-manifold and decomposition is reference-dependent.",
    }


def main():
    """Save the independent audit source and receipt outside frozen run outputs."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("result-dir", "parent-dir", "model-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = audit(a.result_dir, a.parent_dir, a.model_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    result["source_sha256"] = sha(Path(__file__))
    (a.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
