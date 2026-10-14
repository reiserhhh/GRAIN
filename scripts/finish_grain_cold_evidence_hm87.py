#!/usr/bin/env python3
"""Complete HM87 arithmetic after its declared absolute prefix gate failed.

This companion does not change the frozen threshold, source, collector or cold
outputs. It reports both KV realizations and bounds their observed effect on the
mechanism contrasts. No new neural calls or automatic promotion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_cold_evidence as C
from grain_research.construction import math_query_evidence as M
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_author_contrast_hm82 import table, write
from scripts.run_grain_cold_evidence_hm87 import OUT, READOUTS, carrier
from scripts.run_grain_query_evidence_hm86 import check_hashes, seal, unit


def main():
    """Reconcile finite numerical realizations without hiding the failed gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    out = parser.parse_args().out
    followup = out / "03_numerical_reconciliation"
    if followup.exists():
        raise FileExistsError("Keep prior companion outputs")
    manifest, rows = (
        read_json(out / "00/manifest.json"),
        read_json(out / "00/rows.json"),
    )
    check_hashes(ROOT, manifest["code"] | manifest["source"])
    check_hashes(out, manifest["inputs"])
    check_hashes(out, read_json(out / "01/raw_seal.json"))
    check_hashes(out, read_json(out / "02/cold/seal.json"))
    followup.mkdir()
    write(
        followup / "manifest.json",
        {
            "code": {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
            "parent_inputs": manifest["inputs"],
            "cold_seal": digest(out / "02/cold/seal.json"),
            "reason": "Frozen analysis failed absolute body KV tolerance1e-5; no threshold changed",
            "new_neural_calls": 0,
        },
    )
    bundle = read_json(out / "00/bundle.json")
    with np.load(out / "02/cold/observations.npz", allow_pickle=False) as data:
        flat = data["vectors"].copy()
    dirs = np.asarray(bundle["directions"])
    body_rows = [r for r in rows if r["stage"] == "body"]
    body_data = {r["unit"]: carrier(unit(out, r["unit"]), r) for r in body_rows}
    cold = np.zeros((6, 2, 16, 6, 4, 12, 128))
    native = np.zeros((6, 16, 6, 4, 12, 128))
    variant = np.zeros_like(native)
    raw_queries = np.zeros((6, 16, 6, 12, 128))
    errors = {
        "old_body_KV": 0.0,
        "native_body_KV": 0.0,
        "query_rotation": 0.0,
        "cold_projected_KV_sensitivity": 0.0,
        "native_projected_KV_sensitivity": 0.0,
        "native_projected_rotation_sensitivity": 0.0,
    }
    for row, vector in zip(body_rows, flat, strict=True):
        cold[row["context"], row["order"], row["corner"]] = vector
    source = ROOT / manifest["spec"]["source"]
    for row in rows:
        raw = unit(out, row["unit"])
        if row["stage"] == "control":
            old = unit(source, row["old_unit"])
            ix = np.concatenate(row["groups"])
            errors["old_body_KV"] = max(
                errors["old_body_KV"],
                *(float(np.max(np.abs(raw[n][ix] - old[n][ix]))) for n in ("k", "v")),
            )
        if row["stage"] != "native":
            continue
        body = rows[row["body_unit"]]
        k, v, ids, endpoint = body_data[body["unit"]]
        full_k, full_v, _, _ = carrier(raw, {**row, "event_order": [0, 1, 2]})
        errors["native_body_KV"] = max(
            errors["native_body_KV"],
            float(np.max(np.abs(k - full_k))),
            float(np.max(np.abs(v - full_v))),
        )
        qi, context, corner = row["query_index"], body["context"], body["corner"]
        position = endpoint + bundle["gaps"][qi]
        q = C.rotate(
            raw["raw_query"], position, bundle["frequencies"], bundle["scaling"]
        )
        errors["query_rotation"] = max(
            errors["query_rotation"], float(np.max(np.abs(q - raw["q"])))
        )
        observed = C.reading(k, v, q, ids)
        native[context, corner, qi] = observed
        raw_queries[context, corner, qi] = raw["raw_query"]
        ref_q = C.rotate(
            bundle["raw_queries"][qi],
            position,
            bundle["frequencies"],
            bundle["scaling"],
        )
        variant[context, corner, qi] = C.reading(full_k, full_v, ref_q, ids)
        errors["cold_projected_KV_sensitivity"] = max(
            errors["cold_projected_KV_sensitivity"],
            float(
                np.max(
                    np.abs(
                        np.einsum(
                            "rhd,thd->rt",
                            variant[context, corner, qi] - cold[context, 0, corner, qi],
                            dirs,
                        )
                    )
                )
            ),
        )
        for name, other in [
            ("native_projected_KV_sensitivity", C.reading(full_k, full_v, q, ids)),
            ("native_projected_rotation_sensitivity", C.reading(k, v, raw["q"], ids)),
        ]:
            errors[name] = max(
                errors[name],
                float(np.max(np.abs(np.einsum("rhd,thd->rt", other - observed, dirs)))),
            )
    fields = {
        "cold": np.einsum("coaqrhd,thd->coaqrt", cold, dirs),
        "native": np.einsum("caqrhd,thd->caqrt", native, dirs)[:, None],
        "cold_native_KV_realization": np.einsum("caqrhd,thd->caqrt", variant, dirs)[
            :, None
        ],
    }
    points, laws = M.points().astype(int), M.evidence_laws(manifest["spec"]["epsilon"])
    conditions, accounts, actions = [], [], []
    for view, field in fields.items():
        for context in range(6):
            target = (context + 1) % 6
            for order in range(field.shape[1]):
                for ri, name in enumerate(READOUTS):
                    score = field[context, order, :, :, ri, target]

                    def effect(g, polarity, score=score):
                        return score[
                            (points[:, 2] == g) & (points[:, 1] == polarity)
                        ].mean(0)

                    own, foreign = (
                        effect(1, 1) - effect(1, -1),
                        effect(-1, 1) - effect(-1, -1),
                    )
                    conditions.append(
                        {
                            "view": view,
                            "order": order,
                            "target": bundle["targets"][target],
                            "readout": name,
                            "own": float(own[target]),
                            "foreign": float(foreign[target]),
                            "selectivity": float(
                                own[target] - np.delete(own, target).mean()
                            ),
                        }
                    )
                    mask = points[:, 3] == -points[:, 2]
                    for qi, query in enumerate(bundle["targets"]):
                        accounts.append(
                            {
                                "view": view,
                                "order": order,
                                "target": bundle["targets"][target],
                                "query": query,
                                "readout": name,
                                "yg_contrast": float(
                                    (laws[0] - laws[1]) @ score[:, qi]
                                ),
                                "equal_name_count_contrast": float(
                                    2 * (laws[0] - laws[1])[mask] @ score[mask, qi]
                                ),
                            }
                        )
    body_lookup = {
        (r["context"], r["corner"]): body_data[r["unit"]]
        for r in body_rows
        if r["order"] == 0
    }
    action_error = 0.0
    for context in range(6):
        for lo, point in enumerate(points):
            if point[1] != -1:
                continue
            high = point.copy()
            high[1] = 1
            hi = int(np.flatnonzero((points == high).all(1))[0])
            for qi, query in enumerate(bundle["targets"]):

                def reader(data, qi=qi):
                    k, v, ids, endpoint = data
                    return lambda raw_query: C.reading(
                        k,
                        v,
                        C.rotate(
                            raw_query,
                            endpoint + bundle["gaps"][qi],
                            bundle["frequencies"],
                            bundle["scaling"],
                        ),
                        ids,
                    )

                b, q, total = C.polarized_action(
                    reader(body_lookup[context, hi]),
                    reader(body_lookup[context, lo]),
                    raw_queries[context, hi, qi],
                    raw_queries[context, lo, qi],
                )
                action_error = max(action_error, float(np.max(np.abs(b + q - total))))
                for ri, name in enumerate(READOUTS):
                    for ti, target in enumerate(bundle["targets"]):
                        actions.append(
                            {
                                "context": context,
                                "low_corner": lo,
                                "query": query,
                                "direction": target,
                                "actor_A": bool(point[2] == 1),
                                "readout": name,
                                "body": float(np.sum(b[ri] * dirs[ti])),
                                "query_action": float(np.sum(q[ri] * dirs[ti])),
                                "full": float(np.sum(total[ri] * dirs[ti])),
                            }
                        )
    # Independent normalized sums audit the already frozen cold process outputs.
    check_error, checks = 0.0, 0
    for row, actual in zip(body_rows, flat, strict=True):
        k, v, ids, endpoint = body_data[row["unit"]]
        for qi, raw_q in enumerate(bundle["raw_queries"]):
            q = C.rotate(
                raw_q,
                endpoint + bundle["gaps"][qi],
                bundle["frequencies"],
                bundle["scaling"],
            )
            expected = np.zeros_like(actual[qi])
            for h in range(12):
                vh, kh = v[:, h // 6].astype(float), k[:, h // 6].astype(float)
                logits = np.sum(kh * q[h], 1)
                a = np.exp(logits - logits.max())
                a /= a.sum()
                expected[0, h] = np.sum(a[:, None] * vh, 0)
                for event in range(3):
                    ix = ids == event
                    expected[1, h] += a[ix].sum() * vh[ix].sum(0) / ix.sum()
                    expected[2, h] += np.sum(
                        (a[ix] - a[ix].sum() / ix.sum())[:, None] * vh[ix], 0
                    )
                expected[3, h] = vh.sum(0) / len(vh)
            checks += expected.size
            check_error = max(check_error, float(np.max(np.abs(actual[qi] - expected))))
    for row in accounts:
        ti, qi = (
            bundle["targets"].index(row["target"]),
            bundle["targets"].index(row["query"]),
        )
        context, ri = (ti - 1) % 6, READOUTS.index(row["readout"])
        score = fields[row["view"]][context, row["order"], :, qi, ri, ti]
        char = points[:, 1] * points[:, 2]
        mask = points[:, 3] == -points[:, 2]
        check_error = max(
            check_error,
            abs(
                float(row["yg_contrast"])
                - 2 * manifest["spec"]["epsilon"] * np.mean(char * score)
            ),
            abs(
                float(row["equal_name_count_contrast"])
                - 2 * manifest["spec"]["epsilon"] * np.mean(char[mask] * score[mask])
            ),
        )
        checks += 2
    table(followup / "target_selection.csv", conditions)
    table(followup / "author_composition.csv", accounts)
    table(followup / "polarized_actions.csv", actions)
    with (followup / "observations.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            cold=cold,
            native=native,
            variant=variant,
            **{key: value for key, value in fields.items()},
        )
    audit = read_json(out / "02/cold/cold_audit.json")
    if (
        max(check_error, action_error) > manifest["spec"]["arithmetic_tolerance"]
        or audit["forbidden_modules"]
    ):
        raise ValueError("Companion arithmetic/cold contract failed")
    selected = [
        r for r in conditions if r["readout"] == "conditional" and r["order"] == 0
    ]
    by_view = {
        view: {r["target"]: r["selectivity"] for r in selected if r["view"] == view}
        for view in fields
    }
    summary = {
        "status": "VERIFIED_FINITE_HM87_WITH_DECLARED_PREFIX_GATE_FAILURE",
        "frozen_absolute_prefix_gate": "FAIL_preserved_threshold1e-5",
        "original_analysis_failed_before_metric_exports": True,
        "errors": errors,
        "cold_neural_calls": 0,
        "new_companion_neural_calls": 0,
        "captured_calls": len(rows),
        "model_seconds": read_json(out / "01/status.json")["model_seconds"],
        "all_target_selection": conditions,
        "conditional_selectivity_by_view_forward": by_view,
        "observed_KV_sensitivity_selectivity_bound": 4
        * errors["cold_projected_KV_sensitivity"],
        "all_forward_cold_signs_same_in_native_KV_realization": all(
            (by_view["cold"][t] > 0) == (by_view["cold_native_KV_realization"][t] > 0)
            for t in bundle["targets"]
        ),
        "alternate_scalar_checks": checks,
        "alternate_error": check_error,
        "polarization_error": action_error,
        "cold_data_reads": audit["data_reads"],
        "same_agent_not_independent_rebuilder": True,
        "original_scope_complete": False,
    }
    write(followup / "summary.json", summary)
    write(followup / "output_seal.json", seal(out, followup))
    print(
        "HM87 companion complete; failed prefix gate retained",
        errors,
        check_error,
        flush=True,
    )


if __name__ == "__main__":
    main()
