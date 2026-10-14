#!/usr/bin/env python3
"""Independent cell, completion-envelope and cold-scorer arithmetic audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_grain_predecessor_reference import BASE, digest, write
from scripts.score_grain_conditional_reference import restore_reference, score_packet


def audit(out):
    """Recompute source weighting and all held targets without the builder API."""
    summary = json.loads((out / "summary.json").read_text())
    contract = json.loads((out / "contract.json").read_text())
    for name, sha in summary["root_files"].items():
        if digest(out / name) != sha:
            raise ValueError("Changed root artifact")
    for domain, sha in summary["domain_summaries"].items():
        if digest(out / domain / "summary.json") != sha:
            raise ValueError("Changed domain summary")
    for path, sha in contract["anchor_inputs"].items():
        if digest(Path(path)) != sha:
            raise ValueError("Changed fixed anchors")
    if (
        digest(ROOT / "grain_research/conditional_reference.py")
        != contract["module_sha256"]
    ):
        raise ValueError("Changed producing module")
    with np.load(out / "private_directions.npz", allow_pickle=False) as z:
        d, targets = z["direction"], z["targets"]
    norm = np.sqrt(np.sum((d @ targets) ** 2, axis=0))
    maximum_error, counts = 0.0, {}
    for domain in ("pandora", "essays"):
        folder = out / domain
        meta = json.loads((folder / "summary.json").read_text())
        for name, sha in meta["files"].items():
            if digest(folder / name) != sha:
                raise ValueError("Changed domain result")
        ref = restore_reference(
            json.loads((folder / "private_reference.json").read_text())
        )
        with np.load(folder / "private_units.npz", allow_pickle=False) as z:
            x, c, author, source, held = (
                z[k] for k in ("scores", "conditions", "authors", "source", "held")
            )
        frame = pd.DataFrame(x)
        frame["author"], frame["condition"] = author, c
        src = frame[frame.author.isin(source)]
        groups = src.groupby(["author", "condition"], sort=True)
        n = groups.size()
        freq = (n / n.groupby(level=0).transform("sum")).groupby(level=1).sum() / len(
            source
        )
        means = groups[list(range(x.shape[1]))].mean().groupby(level=1).mean()
        people = n.groupby(level=1).size()
        np.testing.assert_allclose(freq.loc[ref["keys"]], ref["weights"], atol=1e-13)
        np.testing.assert_allclose(means.loc[ref["keys"]], ref["mean"], atol=1e-13)
        np.testing.assert_array_equal(
            people.loc[ref["keys"]], ref["source_authors_per_cell"]
        )
        np.testing.assert_array_equal(ref["source_indices"], source)
        if set(source) & set(held):
            raise ValueError("Source/held overlap")
        all_cells = frame.groupby(["author", "condition"])[
            list(range(x.shape[1]))
        ].mean()
        score_count = 0
        for target in ("full", "common5"):
            nu = ref["weights" if target == "full" else "common5_weights"]
            with np.load(
                folder / f"private_scores_{target}.npz", allow_pickle=False
            ) as z:
                for u in held:
                    cells = all_cells.loc[u]
                    known = np.asarray([key in cells.index for key in ref["keys"]])
                    center = np.zeros(x.shape[1])
                    for j, key in enumerate(ref["keys"]):
                        if known[j]:
                            center += nu[j] * cells.loc[key].to_numpy()
                    missing = float(nu[~known].sum())
                    lo, hi = (
                        center @ targets - missing * norm,
                        center @ targets + missing * norm,
                    )
                    np.testing.assert_allclose(
                        [lo, hi], [z["lower"][u], z["upper"][u]], atol=1e-12
                    )
                    np.testing.assert_allclose(
                        [1 - missing, missing],
                        [z["coverage"][u], z["missing"][u]],
                        atol=1e-13,
                    )
                    actual = x[author == u].mean(0)
                    np.testing.assert_allclose(actual, z["actual"][u], atol=1e-13)
                    pieces = z["context"][u] + z["departure"][u] + z["unseen"][u]
                    np.testing.assert_allclose(actual, pieces, atol=1e-13)
                    maximum_error = max(
                        maximum_error,
                        float(abs(actual - pieces).max()),
                        float(abs(lo - z["lower"][u]).max()),
                    )
                    score_count += 1
        original = BASE / "all_plan_closure_v1/local_bilinear_r2" / domain
        with np.load(original / "private_units.npz", allow_pickle=False) as z:
            h = z["h"]
        for target in ("full", "common5"):
            with np.load(
                folder / f"private_scores_{target}.npz", allow_pickle=False
            ) as z:
                for u in held:
                    ix = author == u
                    packet = score_packet(c[ix], h[ix], ref, d, targets, target)
                    for key in (
                        "actual",
                        "center",
                        "context",
                        "departure",
                        "unseen",
                        "lower",
                        "upper",
                    ):
                        np.testing.assert_allclose(packet[key], z[key][u], atol=1e-12)
                    if (packet["point"] is not None) != bool(z["complete"][u]):
                        raise ValueError(
                            "Incomplete target misrepresented as a point score"
                        )
        first = author == held[0]
        np.savez_compressed(
            folder / "private_single_input.npz", conditions=c[first], vectors=h[first]
        )
        table = pd.read_csv(folder / "held_variance_accounting.csv")
        terms = [k for k in table.columns if k not in ("facet", "var_actual")]
        np.testing.assert_allclose(
            table[terms].sum(axis=1), table.var_actual, atol=1e-13
        )
        reveal = pd.read_csv(folder / "private_reveal_checks.csv")
        if (
            not reveal.envelope_nested.all()
            or reveal.max_width_identity_error.max() > 1e-12
        ):
            raise ValueError("Reveal checks failed")
        counts[domain] = {
            "units": len(author),
            "source_authors": len(source),
            "held_authors": len(held),
            "source_reference_cells": len(ref["keys"]),
            "held_target_objects_replayed": score_count,
            "held_contrast_intervals_replayed": score_count * targets.shape[1],
            "single_author_scores_replayed": score_count,
            "variance_identities": len(table),
            "reveal_checks": len(reveal),
        }
    result = {
        "status": "PASS",
        "domains": counts,
        "max_arithmetic_error": maximum_error,
        "scope": "same-agent independent grouping/arithmetic and cold per-author scoring, not independent-executor replication",
        "summary_sha256": digest(out / "summary.json"),
        "audit_code_sha256": digest(Path(__file__)),
    }
    write(out / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=BASE / "all_plan_closure_v1/conditional_reference_r2",
    )
    with threadpool_limits(limits=2):
        audit(p.parse_args().output)
