#!/usr/bin/env python3
"""Independently audit paired order effects, vector patches and raw controls for P1R."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    """Compute exact artifact identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh_summary(base, effects):
    """Aggregate raw logits by explicit original/reversed panel, independent of collector."""
    zero = base.set_index("case_id")
    effects = effects.copy()
    effects["shift"] = (effects.high_logit - effects.low_logit) - effects.case_id.map(
        zero.high_logit - zero.low_logit
    )
    output = {}
    for group in ["ALL", *sorted(effects.group_id.unique())]:
        for criterion in ("event", "trait"):
            s = effects[effects.criterion == criterion]
            if group != "ALL":
                s = s[s.group_id == group]
            p = s.pivot(index="case_id", columns="patch", values="shift")
            rms = np.sqrt(np.dot(p.natural, p.natural) / len(p))
            for patch in p:
                x = p[patch]
                error = np.sqrt(np.dot(x - p.natural, x - p.natural) / len(x))
                size = np.sqrt(np.dot(x, x) / len(x))
                sign = zero.loc[p.index, "role_sign"]
                output[group, criterion, patch] = {
                    "cases": len(x),
                    "natural_rms": rms,
                    "shift_rms": size,
                    "error_rms": error,
                    "relative_rms": size / rms,
                    "relative_error": error / rms,
                    "intended_fraction": float(np.mean(-sign * x > 0)),
                }
    return output


def audit(result, parent, model):
    """Rebuild each source change and numerical comparison without neural forwards."""
    for root in (result, parent):
        if json.loads((root / "run_status.json").read_text())["status"] != "COMPLETE":
            raise ValueError("Incomplete source")
        for name, digest in json.loads(
            (root / "output_hashes.json").read_text()
        ).items():
            if sha(root / name) != digest:
                raise ValueError("Artifact digest")
        manifest = json.loads((root / "manifest.json").read_text())
        with zipfile.ZipFile(root / "source_snapshot.zip") as z:
            for name, digest in manifest["code"].items():
                if hashlib.sha256(z.read(name)).hexdigest() != digest:
                    raise ValueError("Source snapshot")
    manifest = json.loads((result / "manifest.json").read_text())
    if manifest["parent_identity"] != json.loads(
        (parent / "manifest.json").read_text()
    )["identity"] or not (
        manifest["direction_sha256"]
        == sha(parent / "directions.npz")
        == sha(result / "directions.npz")
    ):
        raise ValueError("Parent/direction identity")
    data = json.loads((result / "materials.json").read_text())
    previous = json.loads((parent / "materials.json").read_text())
    old = {r["case_id"]: r for r in previous["rows"]}
    rows = {r["case_id"]: r for r in data["rows"]}
    if (
        len(rows) != 64
        or {r["parent_case"] for r in rows.values()}
        != {r["case_id"] for r in old.values() if r["entity"] != "room"}
        or data["system"] != previous["system"]
    ):
        raise ValueError("Declared population/system")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    layouts = json.loads((result / "token_layouts.json").read_text())
    donors = json.loads((result / "donors.json").read_text())
    states, prefixes = {}, {}
    prefix_error = 0.0
    for case, r in rows.items():
        p = old[r["parent_case"]]
        header, *actions = p["evidence"].splitlines()
        if (
            r["evidence"] != "\n".join([header, *actions[::-1]])
            or r["user"][len(r["evidence"]) :] != p["user"][len(p["evidence"]) :]
        ):
            raise ValueError("Change beyond action order")
        if any(
            r[k] != p[k]
            for k in (
                "level",
                "group_id",
                "room_sign",
                "entity",
                "criterion",
                "role_sign",
                "expected_sign",
                "anchor_span",
                "action_family",
            )
        ):
            raise ValueError("Changed truth/target")
        rendered = tok.apply_chat_template(
            [
                {"role": "system", "content": data["system"]},
                {"role": "user", "content": r["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        start = rendered.index(r["user"])
        a, b = r["anchor_span"]
        n = [
            i
            for i, (lo, hi) in enumerate(encoded["offset_mapping"])
            if start + a <= lo < hi <= start + b
        ]
        l = layouts[case]
        if len(n) != 1 or n != l["anchor"] or encoded["input_ids"] != l["ids"]:
            raise ValueError("Token/anchor mismatch")
        states[case] = np.load(result / "activations" / (case + ".npz"))["block_16"][
            n[0]
        ]
        key = r["group_id"], r["level"], r["room_sign"], r["entity"]
        ids, value = prefixes.setdefault(key, (l["ids"][: n[0] + 1], states[case]))
        if ids != l["ids"][: n[0] + 1]:
            raise ValueError("Noncommon prefix")
        prefix_error = max(prefix_error, float(abs(value - states[case]).max()))
        for channel in ("full", "anchor"):
            d = rows[donors[case][channel]]
            if (
                any(r[k] != d[k] for k in ("group_id", "entity", "room_sign"))
                or d["level"] != -r["level"]
                or d["criterion"] != (r["criterion"] if channel == "full" else "event")
            ):
                raise ValueError("Wrong donor")
    base = pd.read_csv(result / "baselines.csv")
    effects = pd.read_csv(result / "effects.csv")
    if (
        len(base) != 64
        or len(effects) != 192
        or base.case_id.duplicated().any()
        or effects.duplicated(["case_id", "patch"]).any()
        or len(prefixes) != 32
    ):
        raise ValueError("Incomplete score grid")
    direction = np.load(result / "directions.npz")
    geometry = pd.read_csv(result / "geometry.csv").set_index("case_id")
    vector_error = metric_error = 0.0
    for case, r in rows.items():
        x = states[case].astype(float)
        y = states[donors[case]["anchor"]].astype(float)
        d = direction[r["entity"] + "_direction"]
        delta = y - x
        beta = np.linalg.lstsq(d[:, None], delta, rcond=None)[0][0]
        expected = {"natural": y, "projected": x + d * beta, "complement": y - d * beta}
        saved = np.load(result / "patch_states" / (case + ".npz"))
        if set(saved.files) != set(expected):
            raise ValueError("Patch population")
        vector_error = max(
            vector_error,
            *(
                float(abs(saved[k].astype(float) - v).max())
                for k, v in expected.items()
            ),
        )
        g = geometry.loc[case]
        metric_error = max(
            metric_error,
            abs(g.delta_norm - np.linalg.norm(delta)),
            abs(g.projection_energy_fraction - beta**2 / np.dot(delta, delta)),
        )
        for patch in expected:
            s = effects[(effects.case_id == case) & (effects.patch == patch)]
            if len(s) != 1 or s.iloc[0].source_case != donors[case]["anchor"]:
                raise ValueError("Effect provenance")
    checks = pd.read_csv(result / "instrument_checks.csv")
    if checks.check.value_counts().to_dict() != {
        "identity": 2,
        "restore": 2,
        "full": 4,
        "original_replay": 4,
    }:
        raise ValueError("Control population")
    old_base = pd.read_csv(parent / "baselines.csv")
    bindex = base.set_index("case_id")
    oindex = old_base.set_index("case_id")
    control_error = raw_error = 0.0
    for table in (base, effects, checks):
        raw_error = max(
            raw_error,
            float(abs(table.margin - table.high_logit + table.low_logit).max()),
        )
        if not np.isfinite(
            table[["margin", "high_logit", "low_logit"]].to_numpy()
        ).all():
            raise ValueError("Nonfinite scores")
    for r in checks.itertuples():
        source = (
            oindex.loc[r.source_case]
            if r.check == "original_replay"
            else bindex.loc[r.source_case]
        )
        expected = source.high_logit - source.low_logit
        diff = (r.high_logit - r.low_logit) - expected
        control_error = max(control_error, abs(diff))
        metric_error = max(
            metric_error, abs(r.expected_margin - expected), abs(r.error - abs(diff))
        )
        if (
            r.check in ("identity", "restore")
            and r.source_case != r.case_id
            or r.check == "full"
            and r.source_case != donors[r.case_id]["full"]
            or r.check == "original_replay"
            and r.source_case != rows[r.case_id]["parent_case"]
        ):
            raise ValueError("Wrong control source")
        positions = (
            layouts[r.case_id]["anchor"]
            if r.check == "identity"
            else sorted(
                i for part in layouts[r.case_id]["partitions"].values() for i in part
            )
            if r.check == "full"
            else []
        )
        if json.loads(r.positions) != positions:
            raise ValueError("Control positions")
    original_ids = {r["parent_case"] for r in rows.values()}
    old_effects = pd.read_csv(parent / "effects.csv")
    old_effects = old_effects[
        old_effects.case_id.isin(original_ids)
        & (old_effects.axis == "B")
        & old_effects.patch.isin(["natural", "projected", "complement"])
    ]
    now = fresh_summary(base, effects)
    before = fresh_summary(old_base[old_base.case_id.isin(original_ids)], old_effects)
    summary = pd.read_csv(result / "effect_summary.csv").fillna({"group_id": "ALL"})
    paired = pd.read_csv(result / "order_comparison.csv").fillna({"group_id": "ALL"})
    if len(summary) != 30 or len(paired) != 30 or len(old_effects) != 192:
        raise ValueError("Paired comparison population")
    for r in summary.itertuples():
        values = now[r.group_id, r.criterion, r.patch]
        metric_error = max(
            metric_error, *(abs(v - getattr(r, k)) for k, v in values.items())
        )
    for r in paired.itertuples():
        key = r.group_id, r.criterion, r.patch
        for k in now[key]:
            metric_error = max(
                metric_error,
                abs(getattr(r, k + "_original") - before[key][k]),
                abs(getattr(r, k + "_reversed") - now[key][k]),
            )
            if k != "cases":
                metric_error = max(
                    metric_error,
                    abs(
                        getattr(r, k + "_reversed_minus_original")
                        - now[key][k]
                        + before[key][k]
                    ),
                )
    if (
        max(raw_error, metric_error, prefix_error) > 1e-8
        or max(vector_error, control_error) > 1e-5
    ):
        raise ValueError("Independent replay failed")
    return {
        "status": "VERIFIED",
        "identity": manifest["identity"],
        "new_neural_calls": 0,
        "cases": 64,
        "effects": 192,
        "prefix_max": prefix_error,
        "saved_fp32_patch_max": vector_error,
        "raw_logit_max": raw_error,
        "metric_max": metric_error,
        "raw_control_replay_max": control_error,
        "paired_summaries": 30,
    }


def main():
    """Save a separate immutable source/receipt without rerunning the model."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("result-dir", "parent-dir", "model-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    receipt = audit(a.result_dir, a.parent_dir, a.model_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    receipt["source_sha256"] = sha(Path(__file__))
    (a.output_dir / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
