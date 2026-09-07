#!/usr/bin/env python3
"""Independent material, token, code and preference audit for the native P1T panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    """Hash exact artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result, parent, model):
    """Check all material changes and reconstruct margins directly from candidate logits."""
    for root in (result, parent):
        if json.loads((root / "run_status.json").read_text())["status"] != "COMPLETE":
            raise ValueError("Incomplete source")
        for name, digest in json.loads(
            (root / "output_hashes.json").read_text()
        ).items():
            if sha(root / name) != digest:
                raise ValueError("Artifact digest")
        m = json.loads((root / "manifest.json").read_text())
        with zipfile.ZipFile(root / "source_snapshot.zip") as z:
            for name, digest in m["code"].items():
                if hashlib.sha256(z.read(name)).hexdigest() != digest:
                    raise ValueError("Source snapshot")
    manifest = json.loads((result / "manifest.json").read_text())
    if (
        manifest["parent_identity"]
        != json.loads((parent / "manifest.json").read_text())["identity"]
    ):
        raise ValueError("Wrong parent")
    data = json.loads((result / "materials.json").read_text())
    old_data = json.loads((parent / "materials.json").read_text())
    old = {r["case_id"]: r for r in old_data["rows"]}
    layouts = json.loads((result / "token_layouts.json").read_text())
    base = pd.read_csv(result / "baselines.csv").set_index("case_id")
    if (
        len(data["rows"]) != 256
        or len(base) != 256
        or base.index.duplicated().any()
        or data["system"] != old_data["system"]
    ):
        raise ValueError("Population/system")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    avoids = {
        0: "avoided the newcomer, declined an invitation to chat, and remained silent.",
        1: "turned away from the unfamiliar attendee, refused to converse, and kept to themselves.",
    }
    prefixes = {}
    prefix_error = 0.0
    for r in data["rows"]:
        p = old[r["parent_case"]]
        header, *lines = p["evidence"].splitlines()
        if p["room_sign"] != 1 or p["entity"] == "room":
            raise ValueError("Wrong parent subset")
        negative = [
            i
            for i, line in enumerate(lines)
            if line.startswith(r["negative_person"] + " ")
        ]
        if len(negative) != 1 or not any(
            s in lines[negative[0]]
            for s in ("waited quietly near", "listened quietly near")
        ):
            raise ValueError("Changed wrong actor")
        if r["role_sign"] != (-1 if r["entity"] == r["negative_person"] else 1):
            raise ValueError("Actor/role assignment")
        if r["contrast"] == "avoidance":
            lines[negative[0]] = r["negative_person"] + " " + avoids[r["action_family"]]
        elif r["contrast"] != "waiting":
            raise ValueError("Unexpected contrast")
        if r["record_order"] == "reversed":
            lines = lines[::-1]
        elif r["record_order"] != "original":
            raise ValueError("Unexpected order")
        if r["evidence"] != "\n".join([header, *lines]):
            raise ValueError("Extra evidence edit")
        suffix = p["user"][len(p["evidence"]) :]
        if r["code_sign"] == -1:
            suffix = suffix.replace(
                "Return A for Yes and B for No.", "Return A for No and B for Yes."
            )
        elif r["code_sign"] != 1:
            raise ValueError("Unexpected code")
        if r["user"] != r["evidence"] + suffix:
            raise ValueError("Changed query")
        for k in (
            "group_id",
            "level",
            "entity",
            "criterion",
            "role_sign",
            "action_family",
        ):
            if r[k] != p[k] or r[k] != base.loc[r["case_id"], k]:
                raise ValueError("Material/score metadata")
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
        positions = [
            i
            for i, (lo, hi) in enumerate(encoded["offset_mapping"])
            if start + a <= lo < hi <= start + b
        ]
        l = layouts[r["case_id"]]
        if (
            len(positions) != 1
            or positions != l["anchor"]
            or encoded["input_ids"] != l["ids"]
            or r["user"][a:b].strip() != r["entity"]
        ):
            raise ValueError("Token/anchor layout")
        vector = np.load(result / "activations" / (r["case_id"] + ".npz"))[
            "block_16_anchor"
        ]
        key = r["group_id"], r["level"], r["entity"], r["contrast"], r["record_order"]
        ids, value = prefixes.setdefault(key, (l["ids"][: positions[0] + 1], vector))
        if ids != l["ids"][: positions[0] + 1]:
            raise ValueError("Common prefix")
        prefix_error = max(prefix_error, float(abs(vector - value).max()))
    raw_error = float(abs(base.margin - base.high_logit + base.low_logit).max())
    if (
        not np.isfinite(base[["margin", "high_logit", "low_logit"]].to_numpy()).all()
        or len(prefixes) != 64
    ):
        raise ValueError("Nonfinite/incomplete anchors")
    frame = base.copy()
    frame["semantic"] = (frame.high_logit - frame.low_logit) * frame.code_sign
    semantic = pd.read_csv(result / "semantic_rows.csv").set_index("case_id")
    metric = float(abs(frame.semantic - semantic.semantic_margin).max())
    keys = ["group_id", "entity", "criterion", "contrast", "record_order", "code_sign"]
    table = frame.reset_index().pivot(
        index=keys, columns="role_sign", values="semantic"
    )
    gap = table[1] - table[-1]
    saved = pd.read_csv(result / "role_gaps.csv").set_index(keys).gap
    if len(gap) != 128 or not gap.index.equals(saved.index):
        raise ValueError("Role-pair population")
    metric = max(metric, float(abs(gap - saved).max()))
    summary = pd.read_csv(result / "summary.csv", keep_default_na=False)
    if len(summary) != 80:
        raise ValueError("Summary population")
    for r in summary.itertuples():
        s = frame[
            (frame.criterion == r.criterion)
            & (frame.contrast == r.contrast)
            & (frame.record_order == r.record_order)
            & (frame.code_sign == r.code_sign)
        ]
        if r.scope == "setting":
            s = s[s.group_id == r.group_id]
        elif r.scope != "pooled":
            raise ValueError("Summary scope")
        p = s.reset_index().pivot(
            index=["group_id", "entity"], columns="role_sign", values="semantic"
        )
        g = p[1] - p[-1]
        expected = {
            "cases": len(s),
            "role_pairs": len(g),
            "positive_gap_pairs": int((g > 0).sum()),
            "mean_gap": float(g.mean()),
            "rms_gap": float(np.sqrt(np.dot(g, g) / len(g))),
            "positive_role_yes_rate": float((p[1] > 0).mean()),
            "negative_role_no_rate": float((p[-1] < 0).mean()),
            "authored_contrast_preference": float(
                ((p[1] > 0).sum() + (p[-1] < 0).sum()) / p.size
            ),
        }
        metric = max(metric, *(abs(v - getattr(r, k)) for k, v in expected.items()))
    codekeys = [k for k in keys if k != "code_sign"] + ["level"]
    pivot = frame.reset_index().pivot(
        index=codekeys, columns="code_sign", values="semantic"
    )
    pairs = pd.read_csv(result / "code_pairs.csv").set_index(codekeys)
    if len(pairs) != 128 or set(pivot.index) != set(pairs.index):
        raise ValueError("Code-pair population")
    pairs = pairs.loc[pivot.index]
    metric = max(
        metric,
        float(abs(pivot[1] - pairs.standard_margin).max()),
        float(abs(pivot[-1] - pairs.reversed_margin).max()),
        float(abs(pivot[-1] - pivot[1] - pairs.code_margin_difference).max()),
    )
    if not ((pivot[1] * pivot[-1] > 0) == pairs.semantic_preference_agrees).all():
        raise ValueError("Code semantic preference")
    controls = pd.read_csv(result / "instrument_checks.csv")
    if len(controls) != 4:
        raise ValueError("Control count")
    repeat_error = 0.0
    for r in controls.itertuples():
        expected = base.loc[r.case_id, "high_logit"] - base.loc[r.case_id, "low_logit"]
        error = abs(r.high_logit - r.low_logit - expected)
        repeat_error = max(repeat_error, error)
        metric = max(metric, abs(error - r.error), abs(expected - r.expected_margin))
    replay = frame[
        (frame.contrast == "waiting")
        & (frame.record_order == "original")
        & (frame.code_sign == 1)
    ]
    old_base = pd.read_csv(parent / "baselines.csv").set_index("case_id")
    original = old_base.high_logit - old_base.low_logit
    replay_error = float(abs(replay.semantic - replay.parent_case.map(original)).max())
    if (
        len(replay) != 32
        or max(metric, raw_error, prefix_error) > 1e-8
        or max(replay_error, repeat_error) > 1e-5
    ):
        raise ValueError("Numerical replay")
    return {
        "status": "VERIFIED",
        "identity": manifest["identity"],
        "new_neural_calls": 0,
        "native_cases": 256,
        "role_pairs": 128,
        "code_pairs": 128,
        "summaries": 80,
        "prefix_max": prefix_error,
        "raw_logit_max": raw_error,
        "metric_max": metric,
        "raw_repeat_max": repeat_error,
        "raw_parent_replay_max": replay_error,
        "interpretation": "Explicit event facts and authored behavioral-description contrasts; not human trait truth or calibrated Big5 accuracy.",
    }


def main():
    """Write an immutable independent audit receipt."""
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
