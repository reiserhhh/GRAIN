#!/usr/bin/env python3
"""Independent least-squares parity and source/token audit for P1K."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    """Hash an immutable artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parity_replay(contrasts, coefficients, summary):
    """Use a separate design-matrix solver to audit exact contrast coefficients."""
    names = ["constant", "polarity", "code", "semantic"]
    error = 0.0
    if len(contrasts) != 288 or len(coefficients) != 72 or len(summary) != 9:
        raise ValueError("Incomplete parity grid")
    keys = ["group_id", "entity", "criterion", "channel"]
    for key, sub in contrasts.groupby(keys):
        if (
            set(zip(sub.polarity, sub.code)) != {(-1, -1), (-1, 1), (1, -1), (1, 1)}
            or len(sub) != 4
        ):
            raise ValueError("Incomplete parity cells")
        x = np.column_stack(
            [np.ones(4), sub.polarity, sub.code, sub.polarity * sub.code]
        )
        beta = np.linalg.lstsq(x, sub.contrast.to_numpy(), rcond=None)[0]
        selected = coefficients
        for k, v in zip(keys, key):
            selected = selected[selected[k] == v]
        if len(selected) != 1:
            raise ValueError("Coefficient record mismatch")
        saved = selected.iloc[0]
        error = max(
            error,
            float(abs(beta - saved[names].to_numpy(dtype=float)).max()),
            float(abs(x @ beta - sub.contrast.to_numpy()).max()),
        )
        if np.dot(beta, beta) > 1e-12:
            error = max(
                error,
                abs(beta[3] ** 2 / np.dot(beta, beta) - saved.semantic_energy_share),
            )
    for row in summary.itertuples():
        sub = coefficients[
            (coefficients.criterion == row.criterion)
            & (coefficients.channel == row.channel)
        ]
        denom = float(np.sum(sub[names].to_numpy() ** 2))
        if (
            row.pairs != 8
            or len(sub) != 8
            or row.positive_semantic_pairs != int(np.sum(sub.semantic > 0))
        ):
            raise ValueError("Wrong setting/entity counts")
        for name in names:
            error = max(error, abs(sub[name].mean() - getattr(row, name + "_mean")))
            if denom > 1e-12:
                error = max(
                    error,
                    abs(
                        np.dot(sub[name], sub[name]) / denom
                        - getattr(row, name + "_energy_share")
                    ),
                )
    if error > 1e-8:
        raise ValueError("Parity replay failed")
    return error


def audit(result, parent, model_dir):
    """Validate common-prefix states, inherited data and all output statistics."""
    status = json.loads((result / "run_status.json").read_text())
    manifest = json.loads((result / "manifest.json").read_text())
    if status["status"] != "COMPLETE" or status["calls"] != 580:
        raise ValueError("Incomplete P1K")
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
        raise ValueError("Parent identity mismatch")
    with zipfile.ZipFile(result / "source_snapshot.zip") as z:
        for name, digest in manifest["code"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError("Source snapshot mismatch")
    data = json.loads((result / "materials.json").read_text())
    parents = {
        r["case_id"]: r
        for r in json.loads((parent / "materials.json").read_text())["rows"]
    }
    layouts = json.loads((result / "token_layouts.json").read_text())
    base, effects = [
        pd.read_csv(result / (k + ".csv")) for k in ("baselines", "effects")
    ]
    if (
        len(base) != 192
        or len(effects) != 384
        or base.case_id.duplicated().any()
        or effects.duplicated(["case_id", "patch"]).any()
    ):
        raise ValueError("Invalid score grid")
    if set(base.case_id) != set(effects.case_id) or set(base.case_id) != {
        r["case_id"] for r in data["rows"]
    }:
        raise ValueError("Population mismatch")
    for table in (base, effects):
        if not np.isfinite(
            table[["margin", "high_logit", "low_logit"]].to_numpy()
        ).all():
            raise ValueError("Nonfinite score")
    arithmetic = max(
        float(abs(t.margin - t.high_logit + t.low_logit).max()) for t in (base, effects)
    )
    index = base.set_index("case_id")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    prefixes = {}
    state_error = 0.0
    for r in data["rows"]:
        original = parents[r["parent_case"]]
        if any(
            r[k] != original[k]
            for k in (
                "evidence",
                "entity",
                "role_sign",
                "criterion",
                "level",
                "group_id",
                "entity_span",
            )
        ):
            raise ValueError("Inherited material changed")
        wanted = (
            r["polarity"]
            * r["code"]
            * (1 if r["criterion"] == "venue" else original["role_sign"])
        )
        if (
            r["expected_sign"] != wanted
            or index.loc[r["case_id"], "expected_sign"] != wanted
        ):
            raise ValueError("Expected orientation changed")
        if r["polarity"] == r["code"] == 1 and r["user"] != original["user"]:
            raise ValueError("Canonical prompt changed")
        text = tok.apply_chat_template(
            [
                {"role": "system", "content": data["system"]},
                {"role": "user", "content": r["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        layout = layouts[r["case_id"]]
        if encoded["input_ids"] != layout["ids"]:
            raise ValueError("Tokenizer changed")
        start = text.index(r["user"])
        a, b = r["entity_span"]
        n = [
            j
            for j, (lo, hi) in enumerate(encoded["offset_mapping"])
            if start + a <= lo < hi <= start + b
        ]
        if n != layout["N"] or len(n) != 1 or r["user"][a:b].strip() != r["entity"]:
            raise ValueError("Name token coverage changed")
        current = (
            layout["ids"][: n[-1] + 1],
            np.load(result / "activations" / (r["case_id"] + ".npz"))["block_16"][n],
        )
        key = (r["group_id"], r["level"], r["entity"])
        previous = prefixes.setdefault(key, current)
        if previous[0] != current[0]:
            raise ValueError("Noncommon prefix")
        state_error = max(state_error, float(abs(previous[1] - current[1]).max()))
    if len(prefixes) != 16 or state_error > 1e-5:
        raise ValueError("Common state failed")
    metric, restore = 0.0, 0.0
    for case, sub in effects.groupby("case_id"):
        if set(sub.patch) != {"N", "EQD"}:
            raise ValueError("Missing patch")
        r = index.loc[case]
        for v in sub.itertuples():
            d, s = index.loc[v.donor_case], index.loc[v.state_source_case]
            if (
                any(
                    d[k] != r[k]
                    for k in ("group_id", "entity", "criterion", "polarity", "code")
                )
                or d.level != -r.level
            ):
                raise ValueError("Paired donor mismatch")
            if any(s[k] != d[k] for k in ("group_id", "entity", "level")):
                raise ValueError("Cross-person/state source changed")
            if v.patch == "N" and (
                s.criterion != "event" or s.polarity != 1 or s.code != 1
            ):
                raise ValueError("N source is not common event cache")
            if v.patch == "EQD":
                if v.state_source_case != v.donor_case:
                    raise ValueError("Full donor mismatch")
                restore = max(restore, abs(v.margin - d.margin))
    for row in pd.read_csv(result / "role_contrasts.csv").itertuples():
        sub = base if row.channel == "native" else effects[effects.patch == row.channel]
        for k in ("group_id", "entity", "criterion", "polarity", "code"):
            sub = sub[sub[k] == getattr(row, k)]
        if len(sub) != 2 or set(sub.role_sign) != {-1, 1}:
            raise ValueError("Role contrast is not paired")
        if row.channel == "native":
            s = sub.set_index("role_sign").margin
            value = s[1] - s[-1]
        else:
            value = float(
                np.mean(-sub.role_sign * (sub.margin - sub.case_id.map(index.margin)))
            )
        metric = max(metric, abs(value - row.contrast))
    for row in pd.read_csv(result / "native_summary.csv").itertuples():
        sub = base[
            (base.criterion == row.criterion)
            & (base.polarity == row.polarity)
            & (base.code == row.code)
        ]
        metric = max(
            metric,
            abs(np.mean(sub.margin * sub.expected_sign > 0) - row.expected_preference),
        )
        if len(sub) != row.cases or row.cases != 16:
            raise ValueError("Native grid count")
    for row in pd.read_csv(result / "transport_summary.csv").itertuples():
        sub = effects[
            (effects.criterion == row.criterion)
            & (effects.polarity == row.polarity)
            & (effects.code == row.code)
            & (effects.patch == row.patch)
        ]
        shift = sub.margin.to_numpy() - sub.case_id.map(index.margin).to_numpy()
        gap = (
            sub.donor_case.map(index.margin).to_numpy()
            - sub.case_id.map(index.margin).to_numpy()
        )
        values = {
            "shift_rms": np.sqrt(np.mean(shift**2)),
            "gap_rms": np.sqrt(np.mean(gap**2)),
            "relative_error_energy": np.dot(shift - gap, shift - gap)
            / np.dot(gap, gap),
        }
        if row.criterion != "venue":
            values["intended_fraction"] = np.mean(
                -sub.role_sign.to_numpy() * row.polarity * row.code * shift > 0
            )
        for k, v in values.items():
            metric = max(metric, abs(v - getattr(row, k)))
        if len(sub) != row.cases or row.cases != 16:
            raise ValueError("Patch grid count")
    canonical = base[(base.polarity == 1) & (base.code == 1)]
    old = pd.read_csv(parent / "baselines.csv").set_index("case_id").margin
    replay = float(abs(canonical.margin - canonical.parent_case.map(old)).max())
    parity_error = parity_replay(
        *[
            pd.read_csv(result / (k + ".csv"))
            for k in ("role_contrasts", "coefficients", "parity_summary")
        ]
    )
    if max(arithmetic, metric, restore, replay) > 1e-8:
        raise ValueError("Scalar/source replay failed")
    return {
        "status": "VERIFIED",
        "result_identity": manifest["identity"],
        "new_neural_calls": 0,
        "cases": 192,
        "arithmetic_max": arithmetic,
        "metric_replay_max": metric,
        "full_restore_max": restore,
        "parent_replay_max": replay,
        "prefix_state_max": state_error,
        "parity_replay_max": parity_error,
        "interpretation": "Exact response-transformation contrasts; not unique internal causal components or psychological factors.",
    }


def main():
    """Persist independent audit source and receipt away from frozen model outputs."""
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
