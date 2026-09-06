#!/usr/bin/env python3
"""Post-hoc, label-free geometry diagnostics for the original12 technical bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from scripts.run_grain_vector_mechanism_controls import write_json


def energy_partition(x):
    """Partition squared Euclidean energy, not psychological signal/noise variance."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Finite bundle-by-coordinate matrix required")
    center = x.mean(0)
    total = float(np.sum(x * x))
    common = float(len(x) * np.sum(center * center))
    residual = float(np.sum((x - center) ** 2))
    if not np.isclose(total, common + residual, rtol=1e-12, atol=1e-12):
        raise ValueError("Energy identity does not close")
    return {
        "common_energy_fraction": common / total if total else None,
        "between_bundle_residual_fraction": residual / total if total else None,
        "energy_identity_error": abs(total - common - residual),
        "zero_total_energy": total == 0,
    }


def main():
    """Report every arm/site, with no use of the results for source model selection."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("qualification-dir", "audit-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private diagnostic results only")
    out.mkdir(parents=True, exist_ok=False)
    m, s = verified_packet(a.qualification_dir)
    audit = json.loads((a.audit_dir / "summary.json").read_text())
    if (
        audit["status"] != "COMPLETE"
        or audit["qualification_identity"] != m["identity"]
        or s["authors"] != 12
    ):
        raise ValueError("Require independently reconstructed12-case technical packet")
    with np.load(a.qualification_dir / "private_readouts.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    rows = []
    for arm in ("raw", "cue_mask", "placebo_mask"):
        for site in ("block_04", "block_10", "block_16", "final_norm"):
            prefix = f"{arm}__{site}__"
            delta = arrays[prefix + "decision_task_delta"]
            n = delta.shape[1] // 2
            parts = arrays[prefix + "body_variance_parts"]
            if delta.shape != (12, 2 * n) or parts.shape != (12, 3 * n):
                raise ValueError("Unexpected derived dimensions")
            within = float(np.sum(parts[:, n : 2 * n] ** 2))
            between = float(np.sum(parts[:, 2 * n :] ** 2))
            token, equal = (
                arrays[prefix + "body_token"],
                arrays[prefix + "body_equal_view"],
            )
            denom = np.linalg.norm(token, axis=1)
            ratios = np.divide(
                np.linalg.norm(token - equal, axis=1),
                denom,
                out=np.full(12, np.nan),
                where=denom > 0,
            )
            rows.append(
                {
                    "arm": arm,
                    "site": site,
                    "bundles": 12,
                    "dimensions": n,
                    **energy_partition(delta[:, :n]),
                    "body_between_view_variance_fraction": between / (within + between)
                    if within + between
                    else None,
                    "pooling_relative_change_median": float(np.nanmedian(ratios)),
                    "pooling_undefined_ratios": int(np.isnan(ratios).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(out / "technical_geometry.csv", index=False)
    fields = {
        "protocol": "VM05_TECHNICAL_READOUT_GEOMETRY_POSTHOC_V1",
        "qualification_identity": m["identity"],
        "audit_sha256": digest(a.audit_dir / "summary.json"),
        "code_sha256": digest(Path(__file__)),
        "labels_read": False,
        "hypothesis_status": "post-hoc instrumentation diagnostic, not confirmatory personality evidence",
        "not_used_to_select_source_models": True,
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    archive_source(ROOT, out / "source_snapshot.zip")
    summary = {
        "status": "COMPLETE",
        "parent_identity": fields["identity"],
        "bundles": 12,
        "rows": len(rows),
        "labels_read": False,
        "neural_forwards": 0,
        "model_refits": 0,
        "outputs_sha256": {
            "technical_geometry.csv": digest(out / "technical_geometry.csv")
        },
        "interpretation": "Common vector component includes question and average-evidence effects; residual is not established author/personality signal",
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
