#!/usr/bin/env python3
"""Replay P1P training provenance, held material, saved patches and all response ratios."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    """Digest a source artifact without depending on the collector."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result, parent, geometry, model):
    """Use independent vector fits and raw logits; no model forward or outcome fitting."""
    for folder in (result, parent, geometry):
        if json.loads((folder / "run_status.json").read_text())["status"] != "COMPLETE":
            raise ValueError("Incomplete source")
        for name, digest in json.loads(
            (folder / "output_hashes.json").read_text()
        ).items():
            if sha(folder / name) != digest:
                raise ValueError("Artifact digest")
        manifest = json.loads((folder / "manifest.json").read_text())
        with zipfile.ZipFile(folder / "source_snapshot.zip") as z:
            for name, digest in manifest["code"].items():
                if hashlib.sha256(z.read(name)).hexdigest() != digest:
                    raise ValueError("Frozen source digest")
    manifest = json.loads((result / "manifest.json").read_text())
    if manifest["parent_identity"] != json.loads(
        (parent / "manifest.json").read_text()
    )["identity"] or manifest["geometry_manifest_sha256"] != sha(
        geometry / "manifest.json"
    ):
        raise ValueError("Parent chain")
    data = json.loads((result / "materials.json").read_text())
    old = json.loads((parent / "materials.json").read_text())["rows"]
    old_layout = json.loads((parent / "token_layouts.json").read_text())
    if len({r["group_id"] for r in data["rows"]}) != 4 or {
        r["group_id"] for r in old
    } & {r["group_id"] for r in data["rows"]}:
        raise ValueError("Train/held setting overlap")
    directions = np.load(result / "directions.npz")
    direction_error = 0.0
    templates = {}
    for r in old:
        templates.setdefault(
            (r["entity"], r["criterion"]), r["user"][len(r["evidence"]) :]
        )
    for entity in ("Alex", "Blair", "room"):
        train = [
            r
            for r in old
            if r["entity"] == entity
            and r["criterion"] == ("venue_object" if entity == "room" else "event")
        ]
        design, values = [], []
        for r in train:
            b, t = r["role_sign"], r["room_sign"]
            design.append([1, b, t, b * t])
            values.append(
                np.load(parent / "activations" / (r["case_id"] + ".npz"))["block_16"][
                    old_layout[r["case_id"]]["anchor"][0]
                ]
            )
        if len(train) != 16:
            raise ValueError("Training population")
        fitted = np.linalg.lstsq(
            np.array(design, dtype=float), np.array(values, dtype=float), rcond=None
        )[0]
        v = fitted[2 if entity == "room" else 1]
        v /= np.linalg.norm(v)
        d, q = directions[entity + "_direction"], directions[entity + "_random"]
        direction_error = max(
            direction_error,
            float(abs(v - d).max()),
            abs(np.linalg.norm(d) - 1),
            abs(np.linalg.norm(q) - 1),
            abs(np.dot(d, q)),
        )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    layouts = json.loads((result / "token_layouts.json").read_text())
    donors = json.loads((result / "donors.json").read_text())
    rows = {r["case_id"]: r for r in data["rows"]}
    states, prefix = {}, {}
    prefix_error = 0.0
    for case, r in rows.items():
        if r["user"][len(r["evidence"]) :] != templates[r["entity"], r["criterion"]]:
            raise ValueError("Query changed")
        if r["entity"] != "room":
            line = next(
                x for x in r["evidence"].splitlines() if x.startswith(r["entity"] + " ")
            )
            truth = 1 if "walked up" in line or "approached" in line else -1
            if truth != r["role_sign"] or truth != r["expected_sign"]:
                raise ValueError("Action ownership/truth")
        elif r["expected_sign"] != r["room_sign"]:
            raise ValueError("Room truth sign")
        if ("The room was open." in r["evidence"]) != (r["room_sign"] == 1):
            raise ValueError("Room wording")
        text = tok.apply_chat_template(
            [
                {"role": "system", "content": data["system"]},
                {"role": "user", "content": r["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        start = text.index(r["user"])
        a, b = r["anchor_span"]
        positions = [
            i
            for i, (lo, hi) in enumerate(encoded["offset_mapping"])
            if start + a <= lo < hi <= start + b
        ]
        l = layouts[case]
        if (
            encoded["input_ids"] != l["ids"]
            or positions != l["anchor"]
            or len(positions) != 1
            or r["user"][a:b].strip() != r["entity"]
        ):
            raise ValueError("Token/anchor binding")
        states[case] = np.load(result / "activations" / (case + ".npz"))["block_16"][
            positions[0]
        ]
        key = r["group_id"], r["level"], r["room_sign"], r["entity"]
        before, value = prefix.setdefault(
            key, (l["ids"][: positions[0] + 1], states[case])
        )
        if before != l["ids"][: positions[0] + 1]:
            raise ValueError("Shared prefix")
        prefix_error = max(prefix_error, float(abs(value - states[case]).max()))
    base = pd.read_csv(result / "baselines.csv").set_index("case_id")
    effects = pd.read_csv(result / "effects.csv")
    if (
        len(base) != 80
        or len(rows) != 80
        or len(prefix) != 48
        or len(effects) != 640
        or effects.duplicated(["case_id", "axis", "patch"]).any()
    ):
        raise ValueError("Incomplete panel")
    raw_error = max(
        float(abs(base.margin - base.high_logit + base.low_logit).max()),
        float(abs(effects.margin - effects.high_logit + effects.low_logit).max()),
    )
    geom = pd.read_csv(result / "geometry.csv").set_index(["case_id", "axis"])
    state_error = metric_error = 0.0
    fresh = []
    for case, r in rows.items():
        d, q = (
            directions[r["entity"] + "_direction"],
            directions[r["entity"] + "_random"],
        )
        x = states[case].astype(float)
        for axis in ("B", "R"):
            source = donors[case][axis]["anchor"]
            s = rows[source]
            if (
                any(s[k] != r[k] for k in ("group_id", "entity"))
                or s["level"] != r["level"] * (-1 if axis == "B" else 1)
                or s["room_sign"] != r["room_sign"] * (-1 if axis == "R" else 1)
                or s["criterion"]
                != ("event" if r["entity"] != "room" else "venue_object")
            ):
                raise ValueError("Wrong donor axis")
            y = states[source].astype(float)
            delta = y - x
            amplitude = np.linalg.lstsq(d[:, None], delta, rcond=None)[0][0]
            p = d * amplitude
            expected = {
                "natural": y,
                "projected": x + p,
                "complement": y - p,
                "matched_random": x + q * amplitude,
            }
            saved = np.load(result / "patch_states" / (case + "__" + axis + ".npz"))
            if set(saved.files) != set(expected):
                raise ValueError("Patch population")
            state_error = max(
                state_error,
                *(
                    float(abs(saved[k].astype(float) - v).max())
                    for k, v in expected.items()
                ),
            )
            g = geom.loc[(case, axis)]
            metric_error = max(
                metric_error,
                abs(g.delta_norm - np.linalg.norm(delta)),
                abs(g.projected_norm - abs(amplitude)),
                abs(g.projection_energy_fraction - amplitude**2 / np.dot(delta, delta)),
            )
            for patch in expected:
                sub = effects[
                    (effects.case_id == case)
                    & (effects.axis == axis)
                    & (effects.patch == patch)
                ]
                if len(sub) != 1 or sub.iloc[0].source_case != source:
                    raise ValueError("Recorded effect provenance")
                v = sub.iloc[0]
                zero = base.loc[case, "high_logit"] - base.loc[case, "low_logit"]
                fresh.append(
                    {
                        "case_id": case,
                        "group_id": r["group_id"],
                        "criterion": r["criterion"],
                        "axis": axis,
                        "patch": patch,
                        "shift": v.high_logit - v.low_logit - zero,
                        "sign": r["room_sign"]
                        if r["entity"] == "room"
                        else r["role_sign"],
                    }
                )
    fresh = pd.DataFrame(fresh)
    summary = pd.read_csv(result / "effect_summary.csv", keep_default_na=False)
    if len(summary) != 120:
        raise ValueError("Summary population")
    for r in summary.itertuples():
        sub = fresh[(fresh.criterion == r.criterion) & (fresh.axis == r.axis)]
        if r.scope == "setting":
            sub = sub[sub.group_id == r.group_id]
        elif r.scope != "pooled":
            raise ValueError("Summary scope")
        x = sub.pivot(index="case_id", columns="patch", values="shift")
        denominator = np.sqrt(np.dot(x.natural, x.natural) / len(x))
        y = x[r.patch]
        size = np.sqrt(np.dot(y, y) / len(y))
        error = np.sqrt(np.dot(y - x.natural, y - x.natural) / len(y))
        if r.cases != len(x):
            raise ValueError("Summary denominator")
        metric_error = max(
            metric_error,
            abs(r.natural_rms - denominator),
            abs(r.shift_rms - size),
            abs(r.error_rms - error),
            abs(r.relative_rms - size / denominator),
            abs(r.relative_error - error / denominator),
        )
        if r.relevant:
            sign = sub[sub.patch == r.patch].set_index("case_id").loc[x.index, "sign"]
            metric_error = max(
                metric_error, abs(float(r.intended_fraction) - np.mean(-sign * y > 0))
            )
    checks = json.loads((result / "instrument_checks.json").read_text())
    if len(checks) != 12 or max(c["error"] for c in checks) > 1e-5:
        raise ValueError("Recorded instrument checks")
    if (
        direction_error > 1e-10
        or max(raw_error, metric_error, prefix_error) > 1e-8
        or state_error > 1e-5
    ):
        raise ValueError("Numerical replay")
    native = (
        base.assign(correct=base.margin * base.expected_sign > 0)
        .groupby("criterion")
        .agg(cases=("correct", "size"), correct=("correct", "sum"))
    )
    return {
        "status": "VERIFIED",
        "identity": manifest["identity"],
        "new_neural_calls": 0,
        "train_held_disjoint": True,
        "cases": 80,
        "effects": 640,
        "fit_direction_max": direction_error,
        "saved_fp32_patch_max": state_error,
        "raw_logit_max": raw_error,
        "metric_max": metric_error,
        "prefix_max": prefix_error,
        "native": native.to_dict(orient="index"),
        "instrument_limit": "12 collector checks have saved error values but no separate raw control logits; verified as recorded, not independently rerun.",
    }


def main():
    """Save an immutable audit receipt without rerunning the neural experiment."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("result-dir", "parent-dir", "geometry-dir", "model-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    receipt = audit(a.result_dir, a.parent_dir, a.geometry_dir, a.model_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    receipt["source_sha256"] = sha(Path(__file__))
    (a.output_dir / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
