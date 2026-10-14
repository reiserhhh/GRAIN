#!/usr/bin/env python3
"""HM73: cached full-information evidence actions, not teacher precision search."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_target_bound_actions as M
from grain_research.construction.math_heterogeneous_evidence_cube import decode
from scripts.diagnose_grain_event_field_cross_order import export, read, sha, write

OUT = ROOT / "results/grain_math_response_v1/equal_information_target_actions_r1"
SOURCES = {
    "HM68": ROOT / "results/grain_math_response_v1/ownership_composition_r1",
    "HM71": ROOT / "results/grain_math_response_v1/cold_relational_measurement_r1",
}
CODE = (
    "docs/GRAIN_EQUAL_INFORMATION_TARGET_ACTION_PLAN_HM73.md",
    "grain_research/construction/math_target_bound_actions.py",
    "scripts/run_grain_equal_information_target_actions.py",
    "tests/test_grain_target_bound_actions.py",
    "grain_research/construction/math_heterogeneous_evidence_cube.py",
    "grain_research/construction/math_cross_target_operator.py",
    "scripts/diagnose_grain_event_field_cross_order.py",
    "grain_research/construction/math_event_field.py",
)
STAGES = ("body_endpoint", "query_endpoint", "centered_logits", "score", "log_odds")
BANKS = ("source0", "source1", "new_forward", "new_reverse")
PAIRINGS = (
    ("source0", "source1"),
    ("source0", "new_forward"),
    ("source0", "new_reverse"),
    ("new_forward", "new_reverse"),
)
FOCALS = (None, -1, 1)


def table_name(key, stage):
    """Address a complete table without relying on file iteration order."""
    return "__".join((*key, stage))


def prepare():
    """Freeze the result-after question, scripts and existing raw/output identities."""
    if OUT.exists():
        raise FileExistsError("Preserve existing HM73; never overwrite")
    inputs = {}
    for label, source in SOURCES.items():
        if (
            read(source / "verification.json")["status"]
            != f"VERIFIED_FINITE_{label}_NOT_GLOBAL"
        ):
            raise ValueError("Require verified existing " + label)
        for seal in ("01/raw_seal.json", "03/output_seal.json"):
            for name, digest in read(source / seal).items():
                if sha(source / name) != digest:
                    raise ValueError("Existing primary drift: " + name)
        for name in (
            "00/manifest.json",
            "00/rows.json",
            "01/raw_seal.json",
            "03/output_seal.json",
            "verification.json",
        ):
            p = source / name
            inputs[str(p.relative_to(ROOT))] = sha(p)
    (OUT / "00/code").mkdir(parents=True)
    for name in CODE:
        p = OUT / "00/code" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, p)
    write(
        OUT / "00/manifest.json",
        {
            "packet": "HM73",
            "result_after_existing_captures": True,
            "code": {name: sha(ROOT / name) for name in CODE},
            "inputs": inputs,
            "expected_actual_captures_tables": [6336, 132],
            "banks": BANKS,
            "stages": STAGES,
            "pairings": PAIRINGS,
            "targets": M.TARGETS,
            "new_calls_fits": [0, 0],
            "original_scope_complete": False,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    write(OUT / "checkpoint.json", {"status": "PREPARED", "active_processes": []})
    print("HM73 prepared: existing6336 captures; zero new calls/fits", flush=True)


def check():
    """Reject drift in code, sealed source manifests, and frozen copies."""
    m = read(OUT / "00/manifest.json")
    for name, digest in (m["code"] | m["inputs"]).items():
        if sha(ROOT / name) != digest:
            raise ValueError("Code/source drift: " + name)
    for name, digest in m["code"].items():
        if sha(OUT / "00/code" / name) != digest:
            raise ValueError("Frozen code copy drift: " + name)


def load_tables():
    """Read all complete raw tables and prove future queries share actual body input."""
    tables, seen, prefixes, body_captures = {}, set(), {}, {}
    prefix_error, count = 0.0, 0
    for label, source in SOURCES.items():
        for name, digest in read(source / "01/raw_seal.json").items():
            if sha(source / name) != digest:
                raise ValueError("Raw primary drift: " + name)
        for row in read(source / "00/rows.json"):
            if label == "HM71" and row["form"] not in ("paraphrase", "reordered"):
                raise ValueError("Unknown frozen wording/order condition")
            bank = (
                f"source{row['family']}"
                if label == "HM68"
                else "new_forward"
                if row["form"] == "paraphrase"
                else "new_reverse"
            )
            key = (bank, row["focal"], row["companion"])
            corner, query = row["corner"], M.TARGETS.index(row["query"])
            cell = (*key, corner, query)
            if cell in seen:
                raise ValueError("Duplicate cell")
            seen.add(cell)
            doc = (*key, corner)
            prefix = tuple(row["input_ids"][: row["partition"]["body"][-1] + 1])
            if doc in prefixes and prefixes[doc] != prefix:
                raise ValueError("Actual body prefix depends on future query")
            prefixes[doc] = prefix
            path = source / f"01/units/unit_{row['unit']:04d}.json"
            meta = read(path)
            if sha(path.with_suffix(".npz")) != meta["array_sha256"]:
                raise ValueError("Raw array digest mismatch")
            for field in ("unit", "doc_id", "focal", "companion", "corner", "query"):
                if meta[field] != row[field]:
                    raise ValueError("Raw capture identity mismatch")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
                h, qh = a["body_end"].astype(float), a["query_end"].astype(float)
            if h.shape != (1536,) or qh.shape != (1536,):
                raise ValueError("Unexpected frozen cut dimensions")
            if doc in body_captures:
                prefix_error = max(
                    prefix_error, float(np.max(abs(h - body_captures[doc])))
                )
            else:
                body_captures[doc] = h
            group = tables.setdefault(
                key,
                {
                    "body_endpoint": np.full((8, 6, 1536), np.nan),
                    "query_endpoint": np.full((8, 6, 1536), np.nan),
                    "centered_logits": np.full((8, 6, 10), np.nan),
                    "score": np.full((8, 6), np.nan),
                    "log_odds": np.full((8, 6), np.nan),
                },
            )
            z = np.asarray(meta["native"]["candidate_logits"], float)
            native = decode(z)
            if abs(native["score"] - meta["native"]["conditional_score"]) > 1e-10:
                raise ValueError("Score/logit inconsistency")
            for stage, value in {
                "body_endpoint": h,
                "query_endpoint": qh,
                "centered_logits": z - z.mean(),
                **native,
            }.items():
                group[stage][corner, query] = value
            count += 1
    expected = {
        (bank, focal, companion)
        for bank, focal in itertools.product(BANKS, M.TARGETS)
        for companion in (*[k for k in M.TARGETS if k != focal], "neutral")
        if companion != "neutral" or bank.startswith("source")
    }
    if set(tables) != expected or count != 6336 or prefix_error > 1e-5:
        raise ValueError(
            "Missing tables, wrong count or question-dependent body vectors"
        )
    for group in tables.values():
        for value in group.values():
            M.complete(value)
    return tables, {
        "actual_existing_captures": count,
        "complete_tables": len(tables),
        "body_documents_prefix_equal_across_queries": len(prefixes),
        "body_prefix_capture_max_query_drift": prefix_error,
    }


def matrix_rows(coefs):
    """Keep every scalar target matrix and focal condition, without winner selection."""
    rows, diagnostics = [], []
    for bank, stage, focal in itertools.product(BANKS, ("score", "log_odds"), FOCALS):
        matrix = np.zeros((6, 6))
        for k, companion in enumerate(M.TARGETS):
            actions = [
                M.ownership_action(coefs[(bank, j, companion), stage], focal)
                for j in M.TARGETS
                if j != companion
            ]
            matrix[:, k] = np.mean(actions, axis=0)
        tag = "marginal" if focal is None else str(focal)
        diagonal, off = np.diag(matrix), matrix[~np.eye(6, dtype=bool)]
        diagnostics.append(
            {
                "bank": bank,
                "stage": stage,
                "focal": tag,
                "matched_mean": float(diagonal.mean()),
                "other_mean": float(off.mean()),
                "matched_minus_other": float(diagonal.mean() - off.mean()),
                "positive_matched_targets": int(np.sum(diagonal > 0)),
                "matched_is_column_max": int(
                    np.sum(diagonal >= matrix.max(axis=0) - 1e-12)
                ),
            }
        )
        for q, k in itertools.product(range(6), repeat=2):
            rows.append(
                {
                    "bank": bank,
                    "stage": stage,
                    "focal": tag,
                    "query": M.TARGETS[q],
                    "companion": M.TARGETS[k],
                    "action": float(matrix[q, k]),
                    "matched": q == k,
                }
            )
    return rows, diagnostics


def compute(tables):
    """Construct lossless query accounts and qualified target actions on every table."""
    coefs, accounts, actions = {}, [], []
    for key, group in tables.items():
        bank, j, k = key
        for stage, value in group.items():
            c = M.coefficients(value)
            coefs[key, stage] = c
            for index, name in enumerate(M.CHARACTERS[1:], 1):
                part = M.query_account(c[index])
                accounts.append(
                    {
                        "bank": bank,
                        "focal": j,
                        "companion": k,
                        "stage": stage,
                        "character": name,
                    }
                    | {
                        name: v
                        for name, v in part.items()
                        if name not in ("shared", "varying")
                    }
                )
            if stage in ("score", "log_odds"):
                for condition in FOCALS:
                    a = M.ownership_action(c, condition)
                    for q, target in enumerate(M.TARGETS):
                        actions.append(
                            {
                                "bank": bank,
                                "focal": j,
                                "companion": k,
                                "stage": stage,
                                "focal_polarity": "marginal"
                                if condition is None
                                else str(condition),
                                "query": target,
                                "action": float(a[q]),
                                "is_companion_query": target == k,
                                "is_focal_query": target == j,
                            }
                        )
    transport = []
    for left, right in PAIRINGS:
        for j, k in itertools.permutations(M.TARGETS, 2):
            for stage, index in itertools.product(STAGES, range(1, 8)):
                a, b = (
                    coefs[(left, j, k), stage][index],
                    coefs[(right, j, k), stage][index],
                )
                for mode in ("full", "shared", "query_varying"):
                    x, y = a, b
                    if mode == "shared":
                        x, y = a.mean(axis=0), b.mean(axis=0)
                    elif mode == "query_varying":
                        x, y = a - a.mean(axis=0), b - b.mean(axis=0)
                    transport.append(
                        {
                            "left": left,
                            "right": right,
                            "focal": j,
                            "companion": k,
                            "stage": stage,
                            "character": M.CHARACTERS[index],
                            "mode": mode,
                            "cosine": M.cosine(x, y),
                        }
                    )
    matrices, matrix_summary = matrix_rows(coefs)
    mechanism_summary = []
    for bank, stage, character, neutral in itertools.product(
        BANKS, STAGES, ("yg", "xyg"), (False, True)
    ):
        selected = [
            r
            for r in accounts
            if r["bank"] == bank
            and r["stage"] == stage
            and r["character"] == character
            and (r["companion"] == "neutral") == neutral
        ]
        if not selected:
            continue
        shared = sum(r["shared_energy"] for r in selected)
        varying = sum(r["varying_energy"] for r in selected)
        size = np.prod(
            coefs[(bank, selected[0]["focal"], selected[0]["companion"]), stage].shape[
                1:
            ]
        )
        mechanism_summary.append(
            {
                "bank": bank,
                "stage": stage,
                "character": character,
                "neutral": neutral,
                "tables": len(selected),
                "shared_energy": shared,
                "varying_energy": varying,
                "varying_fraction": varying / (shared + varying)
                if shared + varying > 1e-24
                else None,
                "action_coefficient_rms": float(
                    np.sqrt((shared + varying) / (len(selected) * size))
                ),
            }
        )
    context_summary = []
    for bank, stage in itertools.product(BANKS, ("score", "log_odds")):
        contrast, signs = [], []
        for j, k in itertools.permutations(M.TARGETS, 2):
            c = coefs[(bank, j, k), stage]
            low, high = M.ownership_action(c, -1), M.ownership_action(c, 1)
            target = M.TARGETS.index(k)
            signs.append(bool(low[target] * high[target] < 0))
            contrast.append(abs(high[target] - low[target]))
        context_summary.append(
            {
                "bank": bank,
                "stage": stage,
                "directed_pairs": 30,
                "conditional_matched_sign_reversals": sum(signs),
                "conditional_matched_action_gap_mean": float(np.mean(contrast)),
            }
        )
    return (
        coefs,
        accounts,
        actions,
        matrices,
        transport,
        {
            "target_matrix_summary": matrix_summary,
            "query_action_summary": mechanism_summary,
            "context_summary": context_summary,
            "new_calls_fits": [0, 0],
            "original_scope_complete": False,
            "result_after_known_materials_not_independent_replication": True,
            "not_causal_binding_id_or_personality_validity": True,
            "rows_accounts_actions_matrices_transport": [
                len(accounts),
                len(actions),
                len(matrices),
                len(transport),
            ],
        },
    )


def run():
    """Analyze existing observations once, saving all coefficients and signed effects."""
    check()
    if (OUT / "01").exists():
        raise FileExistsError("Preserve analyzed HM73; no overwrite")
    start = time.monotonic()
    tables, provenance = load_tables()
    coefs, accounts, actions, matrices, transport, summary = compute(tables)
    (OUT / "01").mkdir()
    np.savez_compressed(
        OUT / "01/coefficients.npz",
        **{table_name(key, stage): value for (key, stage), value in coefs.items()},
    )
    for name, rows in (
        ("action_accounts", accounts),
        ("target_actions", actions),
        ("target_matrices", matrices),
        ("transport", transport),
    ):
        export(OUT / f"01/{name}.csv", rows)
    summary.update(provenance, analysis_wall_seconds=time.monotonic() - start)
    write(OUT / "01/summary.json", summary)
    write(
        OUT / "01/output_seal.json",
        {p.name: sha(p) for p in sorted((OUT / "01").iterdir())},
    )
    write(
        OUT / "checkpoint.json",
        {"status": "ANALYZED", "active_processes": [], "new_calls_fits": [0, 0]},
    )
    print(
        json.dumps(
            {
                "provenance": provenance,
                "target_matrices": summary["target_matrix_summary"],
            },
            indent=2,
        ),
        flush=True,
    )


def csv_rows(name):
    """Read a sealed table without recasting its scalar units or null directions."""
    with (OUT / f"01/{name}.csv").open() as handle:
        return list(csv.DictReader(handle))


def verify():
    """Replay coefficients, pairwise query variances and cube actions by alternate formulas."""
    check()
    for name, digest in read(OUT / "01/output_seal.json").items():
        if sha(OUT / "01" / name) != digest:
            raise ValueError("Output drift: " + name)
    tables, provenance = load_tables()
    errors, checks = [], 0
    with np.load(OUT / "01/coefficients.npz", allow_pickle=False) as a:
        stored = {name: a[name] for name in a.files}
    for key, group in tables.items():
        for stage, value in group.items():
            c = stored[table_name(key, stage)]
            alternate = np.asarray(
                [
                    np.sum(
                        np.asarray(value, np.longdouble)
                        * np.asarray(sign, np.longdouble).reshape(
                            (8,) + (1,) * (value.ndim - 1)
                        ),
                        axis=0,
                        dtype=np.longdouble,
                    )
                    / 8
                    for sign in M.SIGNS
                ],
                float,
            )
            errors.append(float(np.max(abs(c - alternate))))
            errors.append(
                float(np.max(abs(np.tensordot(M.SIGNS.T, c, axes=(1, 0)) - value)))
            )
            checks += 2 * value.size
            for condition in FOCALS:
                direct = M.direct_ownership_action(value, condition)
                errors.append(
                    float(np.max(abs(M.ownership_action(c, condition) - direct)))
                )
                checks += direct.size
    for row in csv_rows("action_accounts"):
        key = (row["bank"], row["focal"], row["companion"])
        v = stored[table_name(key, row["stage"])][M.CHARACTERS.index(row["character"])]
        varying = (
            sum(
                float(np.square(v[i] - v[j]).sum())
                for i, j in itertools.combinations(range(6), 2)
            )
            / 6
        )
        shared = float(np.square(v.sum(axis=0)).sum() / 6)
        errors.extend(
            [
                abs(varying - float(row["varying_energy"])),
                abs(shared - float(row["shared_energy"])),
            ]
        )
        checks += 2
    for row in csv_rows("target_actions"):
        key = (row["bank"], row["focal"], row["companion"])
        condition = (
            None if row["focal_polarity"] == "marginal" else int(row["focal_polarity"])
        )
        value = M.direct_ownership_action(tables[key][row["stage"]], condition)[
            M.TARGETS.index(row["query"])
        ]
        errors.append(abs(float(value) - float(row["action"])))
        checks += 1
    for row in csv_rows("target_matrices"):
        condition = None if row["focal"] == "marginal" else int(row["focal"])
        value = np.mean(
            [
                M.direct_ownership_action(
                    tables[row["bank"], j, row["companion"]][row["stage"]], condition
                )[M.TARGETS.index(row["query"])]
                for j in M.TARGETS
                if j != row["companion"]
            ]
        )
        errors.append(abs(float(value) - float(row["action"])))
        checks += 1
    for row in csv_rows("transport"):
        i = M.CHARACTERS.index(row["character"])
        v, w = [
            stored[
                table_name((row[bank], row["focal"], row["companion"]), row["stage"])
            ][i]
            for bank in ("left", "right")
        ]
        if row["mode"] == "shared":
            v, w = v.sum(axis=0) / 6, w.sum(axis=0) / 6
        elif row["mode"] == "query_varying":
            v, w = v - v.sum(axis=0) / 6, w - w.sum(axis=0) / 6
        expected = M.cosine(v, w)
        if (row["cosine"] == "") != (expected is None):
            raise ValueError("Null direction was dropped or replaced")
        if expected is not None:
            errors.append(abs(expected - float(row["cosine"])))
        checks += 1
    _, accounts, actions, matrices, transport, summary = compute(tables)
    saved = read(OUT / "01/summary.json")
    for name, value in (summary | provenance).items():
        if saved[name] != value:
            raise ValueError("Summary/provenance drift: " + name)
    if [
        len(csv_rows(n))
        for n in ("action_accounts", "target_actions", "target_matrices", "transport")
    ] != [len(accounts), len(actions), len(matrices), len(transport)]:
        raise ValueError("Incomplete output row scope")
    maximum = max(errors)
    if maximum > 1e-8:
        raise ValueError("Alternate arithmetic discrepancy: " + str(maximum))
    result = {
        "status": "VERIFIED_FINITE_HM73_NOT_GLOBAL",
        "alternate_scalar_checks": checks,
        "max_error": maximum,
        "same_agent_not_third_party": True,
        "new_calls_fits": [0, 0],
        "original_scope_complete": False,
        **provenance,
    }
    write(OUT / "verification.json", result)
    write(OUT / "checkpoint.json", result | {"active_processes": []})
    print(json.dumps(result, indent=2), flush=True)


def main():
    """Run a finite cached-data operation; there is no collector or search command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    globals()[parser.parse_args().command]()


if __name__ == "__main__":
    main()
