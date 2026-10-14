#!/usr/bin/env python3
"""Retain source target means and joint configuration in finite numeric updates."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_likelihood_transport as M
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_relational_diagnostic_update_hm209 import FRAME_NAMES

CONFIG = "configs/grain_likelihood_transport_hm240.json"
DEFAULT = ROOT / "results/grain_math_response_v1/likelihood_transport_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_LIKELIHOOD_TRANSPORT_PLAN_HM240.md",
    "grain_research/construction/math_likelihood_transport.py",
    "scripts/run_grain_likelihood_transport_hm240.py",
    "tests/test_grain_likelihood_transport.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_relational_diagnostic_update_hm209.py",
)
STAGES = ("prepared_seal.json", "computed_seal.json", "verification_seal.json")


def check(out, stages=STAGES):
    """Check frozen methods, inherited input and completed stages without reruns."""
    return R.check(out, stages)


def prepare(out):
    """Bind only old normalized profile arrays and their exact frame addresses."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    R.check(
        parent,
        (
            "prepared_seal.json",
            "built_seal.json",
            "analyzed_seal.json",
            "verified_seal.json",
        ),
    )
    table = pd.read_csv(
        parent / "cold_readings.csv",
        dtype=str,
        usecols=["source_bank", "application_author", "frame"],
    )
    addresses = table[table.source_bank == "0"].drop_duplicates()
    ids = list(addresses.application_author.unique())
    if len(ids) != 24 or len(addresses) != 216:
        raise ValueError("Changed complete24-author/fixed-nine-frame contract")
    expected = [(uid, frame) for uid in ids for frame in FRAME_NAMES]
    if (
        list(zip(addresses.application_author, addresses.frame, strict=True))
        != expected
    ):
        raise ValueError("Profile array address ordering changed")
    for bank in (0, 1):
        bank_addresses = table[table.source_bank == str(bank)].drop_duplicates()
        if (
            list(
                zip(
                    bank_addresses.application_author, bank_addresses.frame, strict=True
                )
            )
            != expected
        ):
            raise ValueError("Source-bank frame addresses disagree")
    with np.load(parent / "analysis.npz", allow_pickle=False) as z:
        arrays = [z[f"panel{b}_log_profile"] for b in (0, 1)]
    if any(a.shape != (216, 2, 32) for a in arrays):
        raise ValueError("Inherited endpoint dimensions changed")
    arrays = [a.reshape(24, 9, 2, 32) for a in arrays]
    start, end, metadata = [], [], []
    for bank, lp in enumerate(arrays):
        for u, uid in enumerate(ids):
            for frame in range(9):
                start.append(lp[u, frame, 1])
                end.append(lp[u, frame, 0])
                metadata.append(
                    {
                        "bank": bank,
                        "author": uid,
                        "operation": "relation",
                        "law": "product_to_paired",
                        "frame": FRAME_NAMES[frame],
                        "primary": frame == 0,
                    }
                )
            for law in range(2):
                for frame in range(1, 9):
                    start.append(lp[u, 0, law])
                    end.append(lp[u, frame, law])
                    metadata.append(
                        {
                            "bank": bank,
                            "author": uid,
                            "operation": "evidence",
                            "law": ("paired", "product")[law],
                            "frame": FRAME_NAMES[frame],
                            "primary": False,
                        }
                    )
    if len(metadata) != 1200:
        raise ValueError("All inherited relation and evidence updates must remain")
    out.mkdir(parents=True, exist_ok=False)
    files = (
        "analysis.npz",
        "cold_readings.csv",
        "manifest.json",
        "analyzed_seal.json",
        "verified_seal.json",
    )
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p) for p in files
            },
            "source_keys_read": ["panel0_log_profile", "panel1_log_profile"],
            "scope": "historical source-criterion Gaussian instrument,not human truth",
            "original_scope_complete": False,
        },
    )
    R.archive(out / "endpoints.npz", log_start=np.array(start), log_end=np.array(end))
    R.table(out / "addresses.csv", metadata)
    example = {
        "evidence_key": f"{ids[0]}/full/relation",
        "source_contract": "HM209/bank0/product-to-paired/unchanged-evidence",
        "log_start": start[0].tolist(),
        "log_end": end[0].tolist(),
    }
    R.write(out / "cold_input.json", example)
    R.seal(out, STAGES[0], ("endpoints.npz", "addresses.csv", "cold_input.json"))
    print(
        json.dumps(
            {
                "authors": 24,
                "source_banks": 2,
                "updates": 1200,
                "primary_bank_author_accounts": 48,
                "new_neural_calls": 0,
            }
        )
    )


def group_summary(result, indices):
    """Describe a predeclared dependent group, without p values or accuracy gates."""
    ix = np.asarray(indices)
    total = result["total_energy"][ix]
    fractions = np.divide(
        result["joint_energy"][ix],
        total,
        out=np.full_like(total, np.nan),
        where=total > 0,
    )
    return {
        "accounts": int(ix.sum()) if ix.dtype == bool else len(ix),
        "mean_total_energy": float(np.mean(total)),
        "mean_main_energy": float(np.mean(result["main_energy"][ix])),
        "mean_joint_energy": float(np.mean(result["joint_energy"][ix])),
        "mean_joint_energy_ratio": float(np.nanmean(fractions)),
        "ratio_scope": "Fisher score energy of this source instrument,NOTpsychologicalinformation",
        "target_delta_RMS": np.sqrt(np.mean(result["delta"][ix] ** 2, axis=0)).tolist(),
        "frozen_joint_tilt_target_delta_RMS": np.sqrt(
            np.mean(result["joint_delta"][ix] ** 2, axis=0)
        ).tolist(),
        "finite_interaction_target_delta_RMS": np.sqrt(
            np.mean(result["interaction_delta"][ix] ** 2, axis=0)
        ).tolist(),
        "mean_main_only_endpoint_KL": float(
            np.mean(result["main_only_endpoint_KL"][ix])
        ),
        "joint_character_delta_RMS": np.sqrt(
            np.mean(result["character_joint_delta"][ix] ** 2, axis=0)
        ).tolist(),
    }


def compute(out):
    """Compile all numeric transport capsules once under the frozen finite cap."""
    cfg = check(out, (STAGES[0],))["config"]
    if (out / "accounts.npz").exists():
        raise ValueError("Exclusive calculation: preserve prior outputs, no rerun")
    with np.load(out / "endpoints.npz", allow_pickle=False) as z:
        a, b = z["log_start"], z["log_end"]
    begin = time.perf_counter()
    result = M.transport(a, b, nodes=cfg["quadrature_nodes"])
    elapsed = time.perf_counter() - begin
    if elapsed > cfg["max_seconds"]:
        raise RuntimeError("Finite cap reached; no automatic repetition")
    R.archive(out / "accounts.npz", **result)
    addresses = pd.read_csv(out / "addresses.csv", dtype={"author": str})
    rows = []
    for n, address in enumerate(addresses.to_dict("records")):
        row = {
            **address,
            "total_energy": float(result["total_energy"][n]),
            "main_energy": float(result["main_energy"][n]),
            "joint_energy": float(result["joint_energy"][n]),
            "main_only_endpoint_KL": float(result["main_only_endpoint_KL"][n]),
        }
        for q, target in enumerate(cfg["targets"]):
            for name in ("delta", "main_delta", "joint_delta", "interaction_delta"):
                row[f"{target}_{name}"] = float(result[name][n, q])
        rows.append(row)
    R.table(out / "update_accounts.csv", rows)
    joint_rows = [
        {
            "update": n,
            "mask": int(mask),
            "full": float(result["character_delta"][n, j]),
            "main": float(result["character_main_delta"][n, j]),
            "joint": float(result["character_joint_delta"][n, j]),
        }
        for n in range(len(a))
        for j, mask in enumerate(M.JOINT_MASKS)
    ]
    R.table(out / "joint_configuration_accounts.csv", joint_rows)
    groups = {"full_relation": group_summary(result, addresses.primary.to_numpy())}
    for bank in (0, 1):
        for op in ("relation", "evidence"):
            ix = ((addresses.bank == bank) & (addresses.operation == op)).to_numpy()
            groups[f"bank{bank}_{op}"] = group_summary(result, ix)
        ix = ((addresses.bank == bank) & addresses.primary).to_numpy()
        groups[f"bank{bank}_full_relation"] = group_summary(result, ix)
    summary = {
        "groups": groups,
        "updates": len(a),
        "authors": 24,
        "source_authors_inherited": 386,
        "quadrature_nodes": 16,
        "targets": cfg["targets"],
        "CPU_seconds_compute": elapsed,
        "independent_units": "24 historical authors;frames/banks/paths are not new people",
        "new_neural_API_training_application_labels": 0,
        "new_capability": "source-posterior joint-update capsule alongside five means",
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.write(out / "cold_output.json", M.cold(R.read(out / "cold_input.json")))
    R.seal(
        out,
        STAGES[1],
        (
            "accounts.npz",
            "update_accounts.csv",
            "joint_configuration_accounts.csv",
            "summary.json",
            "cold_output.json",
        ),
    )
    print(json.dumps(summary))


def verify(out):
    """Check direct finite-profile sums and endpoint identities without tuning."""
    cfg = check(out, STAGES[:2])["config"]
    with np.load(out / "accounts.npz", allow_pickle=False) as z:
        r = {k: z[k] for k in z.files}
    errors = {
        "integrated_target_response": float(
            np.max(np.abs(r["delta"] - r["integrated_slope"]))
        ),
        "integrated_Jeffreys": float(np.max(np.abs(r["total_energy"] - r["jeffreys"]))),
        "energy_partition": float(
            np.max(np.abs(r["total_energy"] - r["main_energy"] - r["joint_energy"]))
        ),
        "joint_character_response": float(
            np.max(
                np.abs(
                    r["character_delta"]
                    - r["character_main_delta"]
                    - r["character_joint_delta"]
                )
            )
        ),
        "finite_corner_partition": float(
            np.max(
                np.abs(
                    r["delta"]
                    - r["main_delta"]
                    - r["joint_delta"]
                    - r["interaction_delta"]
                )
            )
        ),
    }
    direct = 0.0
    # Literal sums over profiles at both extreme quadrature nodes of EVERY update.
    for n in range(len(r["delta"])):
        for k in (0, cfg["quadrature_nodes"] - 1):
            p, ell, joint = (
                r["path_probability"][n, k],
                r["increment"][n],
                r["path_joint"][n, k],
            )
            mean_ell = sum(float(p[g] * ell[g]) for g in range(32))
            residual_mean = sum(float(p[g] * joint[g]) for g in range(32))
            direct = max(direct, abs(residual_mean))
            for q in range(5):
                mu = sum(float(p[g] * M.BITS[g, q]) for g in range(32))
                slope = sum(
                    float(p[g] * (M.BITS[g, q] - mu) * (ell[g] - mean_ell))
                    for g in range(32)
                )
                zero = sum(
                    float(p[g] * (M.BITS[g, q] - mu) * joint[g]) for g in range(32)
                )
                direct = max(direct, abs(zero), abs(slope - r["path_slope"][n, k, q]))
    errors["literal_target_covariance_and_joint_orthogonality"] = direct
    record = R.read(out / "cold_output.json")
    errors["cold_endpoint"] = float(
        np.max(np.abs(np.array(record["target_delta"]) - r["delta"][0]))
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "cold",
        "--input",
        str(out / "cold_input.json"),
    ]
    independent = json.loads(
        subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        ).stdout
    )
    numeric_keys = [
        key
        for key, value in record.items()
        if not isinstance(value, str) and key != "target_order"
    ]
    errors["separate_process_cold_readback"] = max(
        float(np.max(np.abs(np.asarray(independent[key]) - np.asarray(record[key]))))
        for key in numeric_keys
    )
    if any(
        independent[key] != record[key]
        for key in ("evidence_key", "source_contract", "target_order", "scope")
    ):
        raise ValueError("Cold provenance contract changed")
    R.write(out / "cold_cli_readback.json", independent)
    receipt = {
        "errors": errors,
        "updates": 1200,
        "literal_node_accounts": 2400,
        "PASS": bool(max(errors.values()) <= cfg["arithmetic_tolerance"]),
        "same_agent_not_blind": True,
        "no_precision_grid_or_rerun": True,
    }
    R.write(out / "verification.json", receipt)
    if not receipt["PASS"]:
        raise ValueError(
            "Declared finite check failed; preserve failure and do not refine"
        )
    R.seal(out, STAGES[2], ("verification.json", "cold_cli_readback.json"))
    print(json.dumps(receipt))


def main():
    """Expose finite stages and an independent numeric cold process."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "compute", "verify", "check", "cold"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument("--input", type=Path)
    args = p.parse_args()
    if args.stage == "check":
        check(args.out)
        print(json.dumps({"PASS": True, "read_only": True}))
    elif args.stage == "cold":
        if args.input is None:
            p.error("cold requires --input")
        print(json.dumps(M.cold(R.read(args.input))))
    else:
        {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](args.out)


if __name__ == "__main__":
    main()
