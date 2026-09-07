#!/usr/bin/env python3
"""P1T: explicit avoidance versus waiting with paired order and answer-code controls."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_persona_reciprocity import (
    EXPECTED_METADATA,
    BudgetExceeded,
    file_hash,
)
from scripts.run_grain_vector_mechanism_controls import write_json

AVOIDANCE = (
    "avoided the newcomer, declined an invitation to chat, and remained silent.",
    "turned away from the unfamiliar attendee, refused to converse, and kept to themselves.",
)
MAPPING = "Return A for Yes and B for No."


def material(parent):
    """Change only the inactive person's behavior, order and answer map."""
    rows = []
    for r in parent:
        if r["entity"] == "room" or r["room_sign"] != 1:
            continue
        header, *lines = r["evidence"].splitlines()
        negatives = [
            j
            for j, line in enumerate(lines)
            if "waited quietly near" in line or "listened quietly near" in line
        ]
        if len(lines) != 2 or len(negatives) != 1:
            raise ValueError("Expected one noninitiating actor")
        negative = negatives[0]
        negative_person = lines[negative].split()[0]
        for contrast in ("waiting", "avoidance"):
            changed = lines.copy()
            if contrast == "avoidance":
                changed[negative] = (
                    negative_person + " " + AVOIDANCE[r["action_family"]]
                )
            for order in ("original", "reversed"):
                evidence = "\n".join(
                    [header, *(changed if order == "original" else changed[::-1])]
                )
                delta = len(evidence) - len(r["evidence"])
                for code in (1, -1):
                    suffix = r["user"][len(r["evidence"]) :]
                    if suffix.count(MAPPING) != 1:
                        raise ValueError("Unexpected answer code")
                    if code == -1:
                        suffix = suffix.replace(
                            MAPPING, "Return A for No and B for Yes."
                        )
                    rows.append(
                        {
                            **r,
                            "case_id": r["case_id"] + f"__{contrast}__{order}__{code}",
                            "parent_case": r["case_id"],
                            "contrast": contrast,
                            "record_order": order,
                            "code_sign": code,
                            "negative_person": negative_person,
                            "evidence": evidence,
                            "user": evidence + suffix,
                            "anchor_span": [v + delta for v in r["anchor_span"]],
                        }
                    )
    if len(rows) != 256 or len({r["user"] for r in rows}) != 256:
        raise ValueError("Complete256-prompt panel required")
    return rows


def summaries(frame):
    """Retain paired gaps, native preferences and code failures without outcome selection."""
    rows = frame.copy()
    rows["semantic_margin"] = rows.margin * rows.code_sign
    pairkeys = [
        "group_id",
        "entity",
        "criterion",
        "contrast",
        "record_order",
        "code_sign",
    ]
    gaps = []
    for key, s in rows.groupby(pairkeys):
        values = s.set_index("role_sign").semantic_margin
        if len(s) != 2 or set(values.index) != {-1, 1}:
            raise ValueError("Incomplete role contrast")
        gaps.append({**dict(zip(pairkeys, key)), "gap": float(values[1] - values[-1])})
    gaps = pd.DataFrame(gaps)
    summary = []
    for scope, keys in (
        ("pooled", ["criterion", "contrast", "record_order", "code_sign"]),
        ("setting", ["group_id", "criterion", "contrast", "record_order", "code_sign"]),
    ):
        for key, s in rows.groupby(keys):
            info = dict(zip(keys, key))
            g = gaps.copy()
            for k, v in info.items():
                g = g[g[k] == v]
            summary.append(
                {
                    **info,
                    "scope": scope,
                    "cases": len(s),
                    "role_pairs": len(g),
                    "positive_gap_pairs": int((g.gap > 0).sum()),
                    "mean_gap": float(g.gap.mean()),
                    "rms_gap": float(np.sqrt(np.mean(g.gap**2))),
                    "positive_role_yes_rate": float(
                        (s[s.role_sign == 1].semantic_margin > 0).mean()
                    ),
                    "negative_role_no_rate": float(
                        (s[s.role_sign == -1].semantic_margin < 0).mean()
                    ),
                    "authored_contrast_preference": float(
                        (s.semantic_margin * s.role_sign > 0).mean()
                    ),
                }
            )
    codepairs = []
    for key, s in rows.groupby([k for k in pairkeys if k != "code_sign"] + ["level"]):
        values = s.set_index("code_sign").semantic_margin
        if len(s) != 2 or set(values.index) != {-1, 1}:
            raise ValueError("Incomplete code pair")
        keys = [k for k in pairkeys if k != "code_sign"] + ["level"]
        codepairs.append(
            {
                **dict(zip(keys, key)),
                "standard_margin": float(values[1]),
                "reversed_margin": float(values[-1]),
                "semantic_preference_agrees": bool(values[1] * values[-1] > 0),
                "code_margin_difference": float(values[-1] - values[1]),
            }
        )
    return rows, gaps, pd.DataFrame(summary), pd.DataFrame(codepairs)


def run(args):
    """Execute one native-only diagnostic panel on the already-authorized observer."""
    parent = args.parent_dir
    if json.loads((parent / "run_status.json").read_text())["status"] != "COMPLETE":
        raise ValueError("Complete P1P required")
    for name, digest in json.loads((parent / "output_hashes.json").read_text()).items():
        if file_hash(parent / name) != digest:
            raise ValueError("Parent changed")
    source = json.loads((parent / "materials.json").read_text())
    rows = material(source["rows"])
    pm = json.loads((parent / "manifest.json").read_text())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "activations").mkdir()
    paths = list(
        dict.fromkeys(
            [
                Path(__file__),
                ROOT / "docs/GRAIN_CUE_DIAGNOSTICITY_P1T_PLAN.md",
                ROOT / "tests/test_grain_cue_diagnosticity.py",
            ]
            + [ROOT / p for p in pm["code"]]
        )
    )
    manifest = {
        "parent_identity": pm["identity"],
        "material_hash": material_hash(rows),
        "max_calls": 300,
        "max_seconds": 180,
        "private_data_read": False,
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "materials.json", {"system": source["system"], "rows": rows})
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.relative_to(ROOT))
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    base, checks = [], []
    observer, start = None, time.monotonic()

    def flush(status, **extra):
        """Save partial evidence explicitly, without a complete-panel scientific claim."""
        pd.DataFrame(base).to_csv(out / "baselines.csv", index=False)
        pd.DataFrame(checks).to_csv(out / "instrument_checks.csv", index=False)
        receipt = {
            "status": status,
            "identity": manifest["identity"],
            "calls": observer.calls if observer else 0,
            "seconds": time.monotonic() - start,
            "baselines": len(base),
            **extra,
        }
        write_json(out / "run_status.json", receipt)
        return receipt

    try:
        expected = EXPECTED_METADATA | {
            "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
        }
        if {k: file_hash(args.model_dir / k) for k in expected} != expected:
            raise ValueError("Changed model")
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(args.model_dir),
            device="mps",
            dtype="float32",
            max_tokens=256,
            pad_to_length=256,
        )
        observer.sites = {"block_16": observer.model.model.layers[16]}
        write_json(out / "model_manifest.json", observer.manifest())
        layouts = {}
        for r in rows:
            ids = observer.prompt_ids(source["system"], r["user"])
            n = observer.content_intervals(
                source["system"], r["user"], [r["anchor_span"]]
            )
            if len(n) != 1 or any(
                len(observer.candidate_ids(ids, v)) != 1 for v in ("A", "B")
            ):
                raise ValueError("Expected single target/answer tokens")
            layouts[r["case_id"]] = {"ids": ids, "anchor": n}
        write_json(out / "token_layouts.json", layouts)

        def budget():
            """Apply the fixed cost bound before every forward."""
            if (
                observer.calls >= 300
                or time.monotonic() - start > 180
                or torch.mps.current_allocated_memory() > 9 * 1024**3
            ):
                raise BudgetExceeded("P1T cap")

        def score(ids, logits):
            """Save both candidate logits, not just a serialized answer."""
            a, b = [observer.candidate_ids(ids, v)[0] for v in ("A", "B")]
            return {
                "high_logit": float(logits[a]),
                "low_logit": float(logits[b]),
                "margin": float(logits[a]) - float(logits[b]),
            }

        seen, prefixes = set(), {}
        prefix_error = 0.0
        for i, r in enumerate(rows):
            l = layouts[r["case_id"]]
            budget()
            captured, logits = observer.capture(l["ids"], positions=l["anchor"])
            vector = captured["block_16"]
            np.savez_compressed(
                out / "activations" / (r["case_id"] + ".npz"), block_16_anchor=vector
            )
            key = (
                r["group_id"],
                r["level"],
                r["entity"],
                r["contrast"],
                r["record_order"],
            )
            ids, value = prefixes.setdefault(
                key, (l["ids"][: l["anchor"][0] + 1], vector)
            )
            if ids != l["ids"][: l["anchor"][0] + 1]:
                raise ValueError("Common prefix mismatch")
            prefix_error = max(prefix_error, float(abs(value - vector).max()))
            values = score(l["ids"], logits)
            base.append(
                {
                    k: r[k]
                    for k in (
                        "case_id",
                        "parent_case",
                        "group_id",
                        "level",
                        "entity",
                        "criterion",
                        "role_sign",
                        "contrast",
                        "record_order",
                        "code_sign",
                        "action_family",
                    )
                }
                | values
            )
            checkkey = r["criterion"], r["code_sign"]
            if checkkey not in seen:
                seen.add(checkkey)
                budget()
                _, repeat = observer.capture(l["ids"])
                rep = score(l["ids"], repeat)
                error = abs(rep["margin"] - values["margin"])
                checks.append(
                    {
                        "case_id": r["case_id"],
                        **rep,
                        "expected_margin": values["margin"],
                        "error": error,
                    }
                )
                if error > 1e-5:
                    raise ValueError("Repeat control")
            if (i + 1) % 64 == 0:
                print(
                    json.dumps(
                        flush("NATIVE_RUNNING", completed_contexts=(i + 1) // 64)
                    ),
                    flush=True,
                )
        frame = pd.DataFrame(base)
        replay = frame[
            (frame.contrast == "waiting")
            & (frame.record_order == "original")
            & (frame.code_sign == 1)
        ]
        old = pd.read_csv(parent / "baselines.csv").set_index("case_id").margin
        replay_error = float(abs(replay.margin - replay.parent_case.map(old)).max())
        if (
            len(replay) != 32
            or len(prefixes) != 64
            or observer.calls != 260
            or max(prefix_error, replay_error) > 1e-5
        ):
            raise ValueError("Complete panel/prefix/replay gate")
        names = ("semantic_rows", "role_gaps", "summary", "code_pairs")
        for name, table in zip(names, summaries(frame)):
            table.to_csv(out / (name + ".csv"), index=False)
        status = flush(
            "COMPLETE", prefix_state_max=prefix_error, parent_replay_max=replay_error
        )
        write_json(
            out / "output_hashes.json",
            {
                str(p.relative_to(out)): file_hash(p)
                for p in out.rglob("*")
                if p.is_file() and not p.name.startswith("._")
            },
        )
        print(json.dumps(status))
    except Exception as exc:
        flush(
            "PARTIAL_BUDGET" if isinstance(exc, BudgetExceeded) else "FAILED",
            error=str(exc),
        )
        (out / "error_log.txt").write_text(traceback.format_exc())
        raise


def main():
    """Run only from explicit offline paths and a fresh output location."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-dir", "model-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
