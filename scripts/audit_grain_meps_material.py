#!/usr/bin/env python3
"""Audit frozen MEPS metadata without reopening private texts or psychological data."""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from scripts.run_grain_vector_mechanism_controls import write_json


def verify_outputs(folder):
    """Require recorded hashes for each output, not just a COMPLETE headline."""
    summary = json.loads((folder / "summary.json").read_text())
    for name, expected in summary["outputs_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Output path escapes metadata directory")
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Changed metadata artifact: " + name)
    return summary


def main():
    """Verify role, input, and lossless-span identities; report aggregate anomalies."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--windows-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(p for p in out.iterdir() if not p.name.startswith("._")):
        raise ValueError("Use a fresh metadata audit directory")
    pre, windows = verify_outputs(a.preflight_dir), verify_outputs(a.windows_dir)
    manifest = json.loads((a.preflight_dir / "manifest.json").read_text())
    identity = material_hash({k: v for k, v in manifest.items() if k != "identity"})
    if (
        not identity
        == manifest["identity"]
        == pre["parent_identity"]
        == windows["preflight_identity"]
    ):
        raise ValueError("Unbound source/window design")
    if (
        pre["status"] != "COMPLETE_MATERIAL_PREFLIGHT_ONLY"
        or windows["status"] != "COMPLETE_WINDOW_DESIGN_ONLY"
    ):
        raise ValueError("Incomplete design outputs")
    if (
        pre["labels_read"]
        or windows["labels_read"]
        or pre["model_forwards"]
        or windows["model_forwards"]
    ):
        raise ValueError("Unexpected label/model execution scope")
    views = pd.read_csv(a.preflight_dir / "private_view_metadata.csv")
    tokens = pd.read_csv(a.preflight_dir / "private_token_metadata.csv")
    turns = pd.read_csv(a.windows_dir / "private_turn_integrity.csv")
    spans = pd.read_csv(a.windows_dir / "private_window_metadata.csv")
    keys = ["participant_code", "view_kind", "condition"]
    turn_keys = keys + ["role", "turn_sequence"]
    if (
        views.duplicated(keys).any()
        or tokens.duplicated(keys + ["model", "arm"]).any()
        or turns.duplicated(turn_keys).any()
    ):
        raise ValueError("Duplicate observation keys")
    if not tokens.groupby(keys + ["arm"]).text_sha256.nunique().eq(1).all():
        raise ValueError("Different models received different text")
    if not tokens.within_complete_view_budget.eq(tokens.tokens <= 512).all():
        raise ValueError("Incorrect complete-text budget flag")
    if not np.allclose(
        tokens.unknown_fraction, tokens.unknown_tokens / tokens.content_tokens
    ):
        raise ValueError("Incorrect unknown-token denominator")
    groups = {k: f.sort_values("window") for k, f in spans.groupby(turn_keys)}
    if set(groups) != set(turns.set_index(turn_keys).index):
        raise ValueError("Missing or extra speaker turn spans")
    for turn in turns.itertuples(index=False):
        key = tuple(getattr(turn, k) for k in turn_keys)
        s = groups[key]
        if not (
            len(s) == turn.windows
            and np.array_equal(s.window, np.arange(len(s)))
            and s.start.iloc[0] == 0
            and s.end.iloc[-1] == turn.source_chars == turn.covered_chars
            and np.array_equal(s.end.to_numpy()[:-1], s.start.to_numpy()[1:])
            and (s.end > s.start).all()
            and s.tokens.between(1, 512).all()
        ):
            raise ValueError("Nonconserving or over-budget speaker partition")
    counted = turns.groupby(keys + ["role"]).size().unstack("role", fill_value=0)
    for view in views.itertuples(index=False):
        key = tuple(getattr(view, k) for k in keys)
        for role in ("user", "assistant", "system"):
            n = int(counted.loc[key, role]) if role in counted else 0
            if n != getattr(view, role + "_turns"):
                raise ValueError("Speaker count differs from input")
    chat = pd.read_csv(a.preflight_dir / "private_chat_integrity.csv")
    counters = [c for c in chat.columns if c not in keys + ["has_user"]]
    anomalies = {c: int(chat[c].fillna(0).sum()) for c in counters if c != "rows"}
    result = {
        "status": "COMPLETE_METADATA_AUDIT_ONLY",
        "preflight_identity": identity,
        "participants": int(views.participant_code.nunique()),
        "views": len(views),
        "tokenizer_conditions": len(tokens),
        "turns": len(turns),
        "windows": len(spans),
        "source_chars": int(turns.source_chars.sum()),
        "covered_chars": int(turns.covered_chars.sum()),
        "chat_anomalies": anomalies,
        "raw_source_reopened": False,
        "new_model_forwards": 0,
        "labels_read": False,
        "scope": "hashes, speaker/coverage and span identities; not model or personality validation",
    }
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in [
            Path(__file__),
            ROOT / "grain_research/vector_mechanisms/controls.py",
            ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ]:
            z.write(path, path.relative_to(ROOT))
    write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
