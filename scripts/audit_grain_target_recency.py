#!/usr/bin/env python3
"""Independent token/material/order and raw-scalar replay of P1M."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    """Digest a frozen output/source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result, parent, model_dir):
    """Rebuild contrasts from logits and tokenizer offsets without collector helpers."""
    manifest = json.loads((result / "manifest.json").read_text())
    status = json.loads((result / "run_status.json").read_text())
    if status["status"] != "COMPLETE" or status["calls"] != 340:
        raise ValueError("Incomplete P1M")
    for directory in (result, parent):
        for name, digest in json.loads(
            (directory / "output_hashes.json").read_text()
        ).items():
            if sha(directory / name) != digest:
                raise ValueError("Source output changed")
    if (
        manifest["parent_identity"]
        != json.loads((parent / "manifest.json").read_text())["identity"]
    ):
        raise ValueError("Wrong source identity")
    with zipfile.ZipFile(result / "source_snapshot.zip") as z:
        for name, digest in manifest["code"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError("Snapshot changed")
    data = json.loads((result / "materials.json").read_text())
    old = {
        r["case_id"]: r
        for r in json.loads((parent / "materials.json").read_text())["rows"]
    }
    layouts = json.loads((result / "token_layouts.json").read_text())
    base, effects = [
        pd.read_csv(result / (n + ".csv")) for n in ("baselines", "effects")
    ]
    if (
        len(data["rows"]) != 112
        or len(base) != 112
        or len(effects) != 224
        or len({r["user"] for r in data["rows"]}) != 112
    ):
        raise ValueError("Incomplete/distinct-prompt grid")
    if (
        base.case_id.duplicated().any()
        or effects.duplicated(["case_id", "patch"]).any()
    ):
        raise ValueError("Duplicate scores")
    if set(base.case_id) != set(effects.case_id) or set(base.case_id) != {
        r["case_id"] for r in data["rows"]
    }:
        raise ValueError("Different populations")
    index = base.set_index("case_id")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    prefix_groups, room_keys, order_keys = {}, set(), {}
    prefix_error = 0.0
    for r in data["rows"]:
        p = old[r["parent_case"]]
        lines = p["evidence"].splitlines()
        if len(lines) != 3:
            raise ValueError("Unexpected source lines")
        expected_evidence = (
            "\n".join([lines[0], lines[2], lines[1]])
            if r["record_order"] == "reversed"
            else p["evidence"]
        )
        if (
            r["evidence"] != expected_evidence
            or r["group_id"] != p["group_id"]
            or r["level"] != p["level"]
        ):
            raise ValueError("Line order changed content")
        if r["last_named"] != expected_evidence.splitlines()[-1].split()[0]:
            raise ValueError("Incorrect final actor metadata")
        if r["target_kind"] == "person":
            suffix = p["user"][len(p["evidence"]) :]
            if (
                r["user"] != expected_evidence + suffix
                or r["entity"] != p["entity"]
                or r["criterion"] != p["criterion"]
            ):
                raise ValueError("Person question changed")
            action = next(
                line for line in lines[1:] if line.startswith(r["entity"] + " ")
            )
            role = (
                1
                if "greeted someone new" in action
                else -1
                if "stayed silent" in action
                else None
            )
            expected = role if r["criterion"] in ("event", "trait") else 1
        else:
            key = (r["group_id"], r["level"], r["record_order"])
            if (
                key in room_keys
                or r["entity"] != "room"
                or r["criterion"] != "venue_object"
            ):
                raise ValueError("Object duplicated/mislabeled")
            room_keys.add(key)
            if (
                r["user"]
                != expected_evidence
                + "\n\nQuestion: For the room, was the room open during the event? Return A for Yes and B for No."
            ):
                raise ValueError("Object question changed")
            role, expected = r["level"], 1
        if (
            r["role_sign"] != role
            or r["expected_sign"] != expected
            or index.loc[r["case_id"], "expected_sign"] != expected
        ):
            raise ValueError("Role/expected direction wrong")
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
        positions = [
            j
            for j, (lo, hi) in enumerate(enc["offset_mapping"])
            if start + a <= lo < hi <= start + b
        ]
        if (
            enc["input_ids"] != layout["ids"]
            or positions != layout["anchor"]
            or len(positions) != 1
            or r["user"][a:b].strip() != r["anchor_text"]
        ):
            raise ValueError("Token/anchor mismatch")
        key = (r["group_id"], r["level"], r["entity"], r["record_order"])
        state = np.load(result / "activations" / (r["case_id"] + ".npz"))["block_16"][
            positions
        ]
        prefix = layout["ids"][: positions[-1] + 1]
        previous = prefix_groups.setdefault(key, (prefix, state))
        if previous[0] != prefix:
            raise ValueError("Prefix changed across criteria")
        prefix_error = max(prefix_error, float(abs(state - previous[1]).max()))
        orderkey = (r["group_id"], r["level"], r["entity"], r["criterion"])
        comparison = (len(layout["ids"]), layout["span"])
        if order_keys.setdefault(orderkey, comparison) != comparison:
            raise ValueError("Order-specific layout changed")
    if len(room_keys) != 16 or len(prefix_groups) != 48 or prefix_error > 1e-5:
        raise ValueError("Object/common-state accounting")
    arithmetic, metric, restore = 0.0, 0.0, 0.0
    for table in (base, effects):
        if not np.isfinite(
            table[["margin", "high_logit", "low_logit"]].to_numpy()
        ).all():
            raise ValueError("Nonfinite logits")
        arithmetic = max(
            arithmetic,
            float(abs(table.margin - table.high_logit + table.low_logit).max()),
        )
    for case, sub in effects.groupby("case_id"):
        if set(sub.patch) != {"anchor", "EQD"}:
            raise ValueError("Missing patch")
        r = index.loc[case]
        for v in sub.itertuples():
            d, s = index.loc[v.donor_case], index.loc[v.state_source_case]
            if d.level != -r.level or any(
                r[k] != d[k]
                for k in ("group_id", "entity", "criterion", "record_order")
            ):
                raise ValueError("Paired donor changed target/order")
            if any(
                s[k] != d[k] for k in ("group_id", "entity", "level", "record_order")
            ):
                raise ValueError("Anchor source changed person/order")
            if v.patch == "anchor" and s.criterion != (
                "event" if r.target_kind == "person" else "venue_object"
            ):
                raise ValueError("Anchor not from common canonical source")
            if v.patch == "EQD":
                if v.state_source_case != v.donor_case:
                    raise ValueError("Whole source mismatch")
                restore = max(restore, abs(v.margin - d.margin))
    reconstructed = {}
    for channel in ("native", "anchor", "EQD"):
        source = base if channel == "native" else effects[effects.patch == channel]
        for key, sub in source.groupby(
            ["group_id", "entity", "criterion", "record_order"]
        ):
            if len(sub) != 2 or set(sub.role_sign) != {-1, 1}:
                raise ValueError("Not a paired contrast")
            if channel == "native":
                s = sub.set_index("role_sign").margin
                value = s[1] - s[-1]
            else:
                value = float(
                    np.mean(
                        -sub.role_sign * (sub.margin - sub.case_id.map(index.margin))
                    )
                )
            reconstructed[(*key, channel)] = float(value)
    saved = pd.read_csv(result / "role_pairs.csv")
    if len(saved) != 168:
        raise ValueError("Role-pair grid")
    for r in saved.itertuples():
        metric = max(
            metric,
            abs(
                reconstructed[
                    r.group_id, r.entity, r.criterion, r.record_order, r.channel
                ]
                - r.contrast
            ),
        )
    for r in pd.read_csv(result / "summary.csv").itertuples():
        values = np.array(
            [
                v
                for k, v in reconstructed.items()
                if k[2:] == (r.criterion, r.record_order, r.channel)
            ]
        )
        source = base[
            (base.criterion == r.criterion) & (base.record_order == r.record_order)
        ]
        if len(values) != r.pairs:
            raise ValueError("Summary denominator")
        expected = {
            "mean_contrast": np.mean(values),
            "rms_contrast": np.sqrt(np.mean(values**2)),
            "native_expected_preference": np.mean(
                source.margin * source.expected_sign > 0
            ),
        }
        if r.criterion in ("event", "trait"):
            expected["positive_role_pairs"] = np.sum(values > 0)
        for k, v in expected.items():
            metric = max(metric, abs(v - getattr(r, k)))
    orders = pd.read_csv(result / "order_contrasts.csv")
    venue = pd.read_csv(result / "venue_comparison.csv")
    if len(orders) != 84 or len(venue) != 24:
        raise ValueError("Order/venue grid")
    for r in orders.itertuples():
        values = [
            reconstructed[r.group_id, r.entity, r.criterion, o, r.channel]
            for o in ("original", "reversed")
        ]
        metric = max(
            metric,
            abs(values[0] - r.original),
            abs(values[1] - r.reversed),
            abs(values[1] - values[0] - r.order_delta),
        )
        if (values[0] * values[1] > 0) != r.same_sign:
            raise ValueError("Order sign mismatch")
    for r in venue.itertuples():
        person = np.sqrt(
            np.mean(
                [
                    reconstructed[r.group_id, n, "venue", r.record_order, r.channel]
                    ** 2
                    for n in ("Alex", "Blair")
                ]
            )
        )
        obj = abs(
            reconstructed[r.group_id, "room", "venue_object", r.record_order, r.channel]
        )
        metric = max(
            metric,
            abs(person - r.person_rms),
            abs(obj - r.object_abs),
            abs(obj - person - r.object_minus_person),
        )
    old_margin = pd.read_csv(parent / "baselines.csv").set_index("case_id").margin
    canonical = base[(base.target_kind == "person") & (base.record_order == "original")]
    replay = float(abs(canonical.margin - canonical.parent_case.map(old_margin)).max())
    if len(canonical) != 48 or max(arithmetic, metric, restore, replay) > 1e-8:
        raise ValueError("Numerical replay failed")
    return {
        "status": "VERIFIED",
        "result_identity": manifest["identity"],
        "new_neural_calls": 0,
        "cases": 112,
        "object_cases": 16,
        "arithmetic_max": arithmetic,
        "metric_replay_max": metric,
        "full_restore_max": restore,
        "parent_replay_max": replay,
        "prefix_state_max": prefix_error,
        "interpretation": "Finite query-target/order comparison; grammatical target is not an isolated token-level causal factor.",
    }


def main():
    """Save an independent zero-forward audit with reproducible source."""
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
