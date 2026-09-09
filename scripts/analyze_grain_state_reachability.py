#!/usr/bin/env python3
"""Determine whether a reduced state error is missing span or wrong coefficients."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    OUTPUT,
    digest,
    quality,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.state_geometry import (
    compile_projection,
    decompose_state_error,
    linear_reader,
    residual_basis,
    score_compiled_projection,
)
from grain_research.vector_mechanisms.state_transport import pack_state, unpack_state
from scripts.run_grain_navigation_baseline import checked_arrays
from scripts.run_grain_state_transport import DEFAULT, load_operators


def analyze(out):
    """Use later-layer states only in explicitly oracle-labeled diagnostic rows."""
    if out.exists():
        raise FileExistsError("Choose a fresh geometry packet")
    program, models, functions = load_operators(DEFAULT)
    out.mkdir(parents=True)
    bases = {q: residual_basis(m) for q, m in models.items()}
    compact = {q: compile_projection(models[q], functions[q]) for q in models}
    compiled_files = {}
    for q, model in compact.items():
        path = out / f"{q}_projection.npz"
        np.savez_compressed(path, **model)
        compiled_files[path.name] = digest(path)
    rows, provenance = [], {}
    with threadpool_limits(limits=2):
        for mode, source, names in (
            ("worlds", OUTPUT, ("design.json", "worlds.json", "packets")),
            (
                "natural",
                BASE / "navigation_v1/NAV-04/natural_evidence_r1",
                ("private_design.json", "private_worlds.json", "private_packets"),
            ),
        ):
            packet = BASE / f"navigation_v1/NAV-10/state_{mode}_r1"
            if read_json(packet / "run_status.json")["status"] != "COMPLETE":
                raise ValueError("Finish state evaluation first")
            provenance[str(packet.relative_to(ROOT))] = digest(
                packet / "predictions.csv"
            )
            world = {w["world_id"]: w for w in read_json(source / names[1])}
            for d in read_json(source / names[0]):
                ident = ("index", "query", "world_id")
                sm, states = checked_arrays(
                    packet / "private_units" / f"unit_{d['index']:04d}.json"
                )
                tm, teacher = checked_arrays(
                    source / names[2] / f"unit_{d['index']:04d}.json"
                )
                if any(sm["identity"][k] != d[k] or tm[k] != d[k] for k in ident):
                    raise ValueError("Geometry identity mismatch")
                q = d["query"]
                m, f = models[q], functions[q]
                actual = pack_state(teacher["P"], teacher["Q"])
                early = states["residual_mean"] - m["mean_r"]
                reader, _ = linear_reader(f)
                target = f.parts(teacher["P"], teacher["Q"])["score"]
                for degree in (1, 2):
                    method = f"transport{degree}"
                    estimated = states[method]
                    oracle, outside, inside, energies = decompose_state_error(
                        m, bases[q], early, actual, estimated
                    )
                    estimate_score = f.parts(
                        *unpack_state(estimated, program["state_shapes"][q])
                    )["score"]
                    oracle_score = f.parts(
                        *unpack_state(oracle, program["state_shapes"][q])
                    )["score"]
                    compiled_score = score_compiled_projection(
                        compact[q], early, degree
                    )["score"]
                    outside_action = float((outside @ reader).sum())
                    inside_action = float((inside @ reader).sum())
                    rows.append(
                        {
                            "mode": mode,
                            "query": q,
                            "world_id": d["world_id"],
                            "split": world[d["world_id"]]["split"],
                            "method": method,
                            "rank": len(bases[q]),
                            "target": target,
                            "estimate": estimate_score,
                            "oracle": oracle_score,
                            "oracle_uses_teacher": True,
                            "native": tm["native"]["conditional_score"],
                            **energies,
                            "outside_score_error": outside_action,
                            "inside_score_error": inside_action,
                            "score_accounting_error": abs(
                                target - estimate_score - outside_action - inside_action
                            ),
                            "compiled_score_error": abs(
                                compiled_score - estimate_score
                            ),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_geometry.csv", index=False)
    summaries = []
    for (mode, q, method, split), g in frame.groupby(
        ["mode", "query", "method", "split"]
    ):
        summaries.append(
            {
                "mode": mode,
                "query": q,
                "method": method,
                "split": split,
                "n": len(g),
                "rank": int(g["rank"].iloc[0]),
                "outside_energy_fraction": float(
                    g.outside_squared.sum() / max(g.total_squared.sum(), 1e-20)
                ),
                "estimate_r": quality(g.target, g.estimate)["r"],
                "oracle_r": quality(g.target, g.oracle)["r"],
                "estimate_mae": quality(g.target, g.estimate)["mae"],
                "oracle_mae": quality(g.target, g.oracle)["mae"],
                "outside_action_rms": float(np.sqrt(np.mean(g.outside_score_error**2))),
                "inside_action_rms": float(np.sqrt(np.mean(g.inside_score_error**2))),
            }
        )
    pd.DataFrame(summaries).to_csv(out / "reachability_summary.csv", index=False)
    write_json(
        out / "summary.json",
        {
            "status": "COMPLETE_DIAGNOSTIC_NOT_RUNTIME_ORACLE",
            "program_sha256": digest(DEFAULT / "program.json"),
            "source_sha256": provenance,
            "code_sha256": {
                p: digest(ROOT / p)
                for p in (
                    "scripts/analyze_grain_state_reachability.py",
                    "grain_research/vector_mechanisms/state_geometry.py",
                )
            },
            "compiled_files": compiled_files,
            "max_compiled_A_B_error": float(frame.compiled_score_error.max()),
            "max_signed_score_accounting_error": float(
                frame.score_accounting_error.max()
            ),
            "max_energy_accounting_error": float(
                frame.orthogonality_accounting_error.max()
            ),
            "oracle_never_candidate": True,
            "human_labels_used": False,
        },
    )
    print(
        pd.DataFrame(summaries)
        .query("split in ['held_scene','held_author']")
        .round(4)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    analyze(BASE / "navigation_v1/NAV-02/state_reachability_r1")
