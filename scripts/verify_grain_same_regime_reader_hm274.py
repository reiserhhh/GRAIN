#!/usr/bin/env python3
"""Read-only HM274 verification after the frozen verifier's order/odd typo."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_same_regime_reader_hm274 as E
from scripts import run_grain_same_regime_reader_hm274_recovery as C


def reader_channel(values, channel):
    """Derive a contact channel literally from the four archived readings."""
    x = np.asarray(values, float)
    if x.shape != (48, 4, 5) or not np.isfinite(x).all():
        raise ValueError("Complete finite reader observations required")
    if channel == "join":
        return (x[:, 0] + x[:, 1] - x[:, 2] - x[:, 3]) / 2
    if channel == "order":
        return (x[:, 0] - x[:, 1]) / 2
    raise ValueError("Only the two registered contact channels are archived")


def verify(out):
    """Verify sealed numerical outputs without fitting, retokenizing or forwards."""
    manifest = E.R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    dest = out / "readonly_verification_r1"
    dest.mkdir()
    E.R.write(
        dest / "manifest.json",
        {
            "stage": "POSTCOLLECTION_READONLY_VERIFICATION",
            "original_verifier_failure": "KeyError: 'order'; reader parts call it odd",
            "script_hash": E.R.digest(Path(__file__)),
            "input_manifest_hash": E.R.digest(out / "manifest.json"),
            "analyzed_seal_hash": E.R.digest(out / "analyzed_seal.json"),
            "no_scientific_output_edit_or_model_rerun": True,
        },
    )
    sums, fields, panel = [
        E.load(out / p)
        for p in ("observed_sums.npz", "source_readings.npz", "numeric_pairs.npz")
    ]
    errors = {"token_resum": 0.0, "old_source_forward": 0.0, "literal_contacts": 0.0}
    candidates = 0
    for u in range(48):
        for c in range(4):
            cap = E.load(out / f"captures/{u:02d}_{c}.npz")
            for k in ("chosen", "partition"):
                direct = np.array(
                    [
                        sum(float(v) for v in cap[k + "_tokens"][a:b])
                        for a, b in zip(
                            cap["offsets"][:-1], cap["offsets"][1:], strict=True
                        )
                    ]
                )
                errors["token_resum"] = max(
                    errors["token_resum"], float(np.max(abs(direct - sums[k][u, c])))
                )
            candidates += len(cap["offsets"]) - 1
    old = E.load(
        ROOT / manifest["config"]["source_parent"] / "conditional_profiles.npz"
    )
    for key, value in fields.items():
        prefix, field = key.rsplit("/", 1)
        old_prefix = prefix.replace(
            "fixed_profile_relink", "fixed_within_bank_profile_relink"
        )
        errors["old_source_forward"] = max(
            errors["old_source_forward"],
            float(
                np.max(
                    abs(
                        value[:, 0] - old[f"{old_prefix}/{field}"][panel["panel_edges"]]
                    )
                )
            ),
        )
    behavior = E.R.read(out / "cold_readback.json")["accounts"]
    rows = 0
    with (out / "all_contacts.csv").open() as f:
        for row in csv.DictReader(f):
            key = f"{row['assignment']}/{row['space']}/{row['bank']}/targets"
            values = E.P.components(fields[key])[row["component"]]
            ix = (
                np.ones(48, bool)
                if row["window"] == "pooled"
                else panel["windows"] == int(row["window"])
            )
            a = (
                (values[:, 0] + values[:, 1]) / 2
                if row["channel"] == "join"
                else (values[:, 0] - values[:, 1]) / 2
            )
            b = reader_channel(
                behavior[row["scale"]]["balanced_key"]["values"], row["channel"]
            )
            i, j = (
                E.B.TARGETS.index(row["source_target"]),
                E.B.TARGETS.index(row["reader_target"]),
            )
            literal = sum(
                float(x) * float(y) for x, y in zip(a[ix, i], b[ix, j], strict=True)
            ) / int(ix.sum())
            errors["literal_contacts"] = max(
                errors["literal_contacts"], abs(literal - float(row["contact"]))
            )
            rows += 1
    cli = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(ROOT / "scripts/score_grain_same_regime_reader_cold.py"),
                str(out / "cold_packet.json"),
            ],
            text=True,
        )
    )
    if (
        candidates != 9600
        or rows != 14400
        or cli != E.R.read(out / "cold_readback.json")
        or max(errors.values()) > 1e-10
    ):
        raise ArithmeticError("Frozen numerical readback failed")
    E.R.write(
        dest / "verification.json",
        {
            "PASS": True,
            "errors": errors,
            "candidates": candidates,
            "contacts": rows,
            "cold_CLI_equal": True,
            "same_agent_not_blind": True,
            "original_scope_complete": False,
            "no_model_rerun_or_scientific_output_edit": True,
        },
    )
    E.R.seal(dest, "verification_seal.json", ("manifest.json", "verification.json"))
    E.R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    print(json.dumps(E.R.read(dest / "verification.json")), flush=True)


def main():
    """Run only the declared verification or readback; never collect observations."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("verify", "check"))
    p.add_argument("--out", type=Path, default=C.DEFAULT)
    args = p.parse_args()
    if args.stage == "verify":
        verify(args.out)
    else:
        E.R.check(args.out, ("collected_seal.json", "analyzed_seal.json"))
        E.R.check(args.out / "readonly_verification_r1", ("verification_seal.json",))
        print("HM274 primary/recovery outputs and readonly verification seals PASS")


if __name__ == "__main__":
    main()
