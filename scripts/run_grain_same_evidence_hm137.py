#!/usr/bin/env python3
"""One fixed same-evidence geometry comparison; never fit a target predictor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import anonymous_bundle as AB
from grain_research.construction import math_same_evidence_geometry as M
from scripts import run_grain_observation_program_hm136 as PREV

SPEC = ROOT / "configs/grain_same_evidence_hm137.json"
CODE = (
    "configs/grain_same_evidence_hm137.json",
    "docs/GRAIN_SAME_EVIDENCE_GEOMETRY_PLAN_HM137.md",
    "grain_research/construction/math_same_evidence_geometry.py",
    "grain_research/construction/anonymous_bundle.py",
    "scripts/run_grain_same_evidence_hm137.py",
    "tests/test_grain_same_evidence_geometry.py",
)
SECTORS = ("constant", "b", "c", "bc")
VIEWS = (
    "E5_full",
    "GRAIN_all15",
    "author_stability",
    "response_stability",
    "principal_components",
)


def read(path):
    """Read one explicitly declared JSON artifact."""
    return json.loads(Path(path).read_text())


def write(path, value):
    """Save generated numerical state; caller controls immutable versus checkpoint."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    """Stream a hash, including the cached model without a large memory copy."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table(path, rows):
    """Write a complete fixed-field table, preserving every nonwinning case."""
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(out):
    """Authenticate old public units and frozen directions before any E5 call."""
    if out.exists():
        raise FileExistsError("Preserve previous or failed run")
    spec = read(SPEC)
    previous = ROOT / spec["previous"]
    PREV.check(previous)
    PREV.H.verify_hashes(previous, read(previous / "02/output_seal.json"))
    if read(previous / "status.json")["state"] != "COLLECTED":
        raise ValueError("Terminal HM136 required")
    bundle_path = ROOT / spec["grain_bundle"]
    if sha(bundle_path) != spec["grain_bundle_sha256"]:
        raise ValueError("Original anonymous bundle changed")
    bundle = read(bundle_path)
    AB.verify_bundle(bundle)
    if bundle["encoder"]["prefix"] != spec["encoder_prefix"]:
        raise ValueError("Deployed encoder prefix required")
    if np.shape(bundle["directions"]) != (1024, 15):
        raise ValueError("Complete original15 directions required")
    sources = read(previous / "00/sources.json")
    records, addresses = [], set()
    for source, definition in enumerate(sources):
        for row in definition["rows"]:
            address = (source, row["bank"], row["family"], row["b"], row["c"])
            if (
                address in addresses
                or row["b"] not in (-1, 1)
                or row["c"] not in (-1, 1)
            ):
                raise ValueError("Duplicate or invalid public response corner")
            addresses.add(address)
            text = row["history_text"]
            records.append(
                {
                    "id": len(records),
                    "source": source,
                    "corpus": definition["name"],
                    "old_id": row["id"],
                    "bank": row["bank"],
                    "family": row["family"],
                    "b": row["b"],
                    "c": row["c"],
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
    if len(records) != spec["expected_records"] or len(addresses) != 96:
        raise ValueError("Full96 old public histories required")
    model = Path(spec["encoder_snapshot"])
    model_files = [
        model / n
        for n in (
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        )
    ]
    if not all(p.is_file() for p in model_files):
        raise FileNotFoundError("Local pinned E5 cache missing; do not download")
    (out / "00").mkdir(parents=True)
    (out / "01").mkdir()
    write(out / "00/records.json", records)
    inputs = [
        bundle_path,
        previous / "02/observations.npz",
        previous / "00/sources.json",
        previous / "02/output_seal.json",
        previous / "status.json",
        out / "00/records.json",
    ]
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {str(ROOT / p): sha(ROOT / p) for p in CODE},
            "inputs": {str(p.resolve()): sha(p) for p in inputs},
            "model_files": {str(p): sha(p) for p in model_files},
            "grain_families": bundle["families"],
            "previous_targets": read(previous / "00/manifest.json")["spec"]["targets"],
            "view_interpretation": "frozen event coordinates, not author posterior scores",
            "new_fits": 0,
            "private_raw_reads": 0,
            "original_scope_complete": False,
        },
    )
    write(
        out / "status.json",
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "pending": None,
            "model_seconds": 0,
        },
    )
    print("PREPARED HM137:96 same-evidence units", flush=True)


def check(out, model=False):
    """Authenticate fixed code/inputs and, before collection, local model files."""
    m = read(out / "00/manifest.json")
    for p, expected in (
        m["code"] | m["inputs"] | (m["model_files"] if model else {})
    ).items():
        if sha(p) != expected:
            raise ValueError("Changed sealed input: " + p)
    return m


def collect(out):
    """Exactly twelve batches; uncertain or over-budget calls are not replayed."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    m = check(out, model=True)
    spec, state = m["spec"], read(out / "status.json")
    if state["state"] != "PREPARED":
        raise ValueError("Do not retry an existing scientific attempt")
    records = read(out / "00/records.json")
    started = time.monotonic()
    state["state"] = "LOADING"
    write(out / "status.json", state)
    try:
        torch.set_num_threads(4)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(
            spec["encoder_snapshot"], local_files_only=True
        )
        encoder = (
            AutoModel.from_pretrained(
                spec["encoder_snapshot"],
                local_files_only=True,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )
        batches = []
        for start in range(0, len(records), spec["batch_size"]):
            if time.monotonic() - started >= spec["maximum_model_seconds"]:
                raise TimeoutError("Original model cap reached; no next batch")
            part = records[start : start + spec["batch_size"]]
            texts = [spec["encoder_prefix"] + r["text"] for r in part]
            encoded = tok(texts, padding=True, truncation=False, return_tensors="pt")
            if encoded["input_ids"].shape[1] > spec["maximum_length"]:
                raise ValueError("Public history exceeds limit; no silent truncation")
            inputs = {k: v.to(device) for k, v in encoded.items()}
            state.update(state="RUNNING", attempts=state["attempts"] + 1, pending=start)
            write(out / "status.json", state)
            with torch.inference_mode():
                hidden = encoder(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                raw = pooled.cpu().numpy().astype(float)
            normalized = AB.normalize(raw)
            path = out / f"01/b{start:03d}.npz"
            np.savez(
                path,
                raw=raw,
                normalized=normalized,
                ids=np.array([r["id"] for r in part]),
                input_ids=encoded["input_ids"].numpy(),
                attention_mask=encoded["attention_mask"].numpy(),
            )
            batches.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha(path),
                    "start": start,
                    "records": len(part),
                    "max_tokens": int(encoded["input_ids"].shape[1]),
                }
            )
            state.update(
                completed=state["completed"] + len(part),
                pending=None,
                model_seconds=time.monotonic() - started,
            )
            write(out / "01/batches.json", batches)
            write(out / "status.json", state)
            print(
                "HM137",
                state["completed"],
                "/96",
                round(state["model_seconds"], 3),
                flush=True,
            )
        state.update(
            state="COLLECTED",
            pending=None,
            model_seconds=time.monotonic() - started,
            device=device,
            dtype="float32",
            pooling="masked mean then float64 L2",
        )
        write(out / "status.json", state)
    except BaseException as error:
        state.update(
            state="FAILED_NOT_RETRIED",
            model_seconds=time.monotonic() - started,
            error=f"{type(error).__name__}: {error}",
        )
        write(out / "status.json", state)
        raise


def arrays(out):
    """Join exact public corner addresses, not embedding similarity or labels."""
    m = check(out)
    state = read(out / "status.json")
    if (
        state["state"] != "COLLECTED"
        or state["completed"] != 96
        or state["pending"] is not None
    ):
        raise ValueError("Complete terminal same-evidence collection required")
    records = read(out / "00/records.json")
    cube = np.zeros((2, 2, 6, 2, 2, 1024))
    seen, norm_error = set(), 0.0
    for batch in read(out / "01/batches.json"):
        if sha(batch["path"]) != batch["sha256"]:
            raise ValueError("Changed encoder primitive")
        with np.load(batch["path"], allow_pickle=False) as z:
            normalized = AB.normalize(z["raw"])
            norm_error = max(
                norm_error, float(np.max(abs(normalized - z["normalized"])))
            )
            for i, row_id in enumerate(z["ids"]):
                r = records[int(row_id)]
                if int(row_id) in seen:
                    raise ValueError("Duplicate event address")
                seen.add(int(row_id))
                address = (
                    r["source"],
                    r["bank"],
                    r["family"],
                    (r["b"] + 1) // 2,
                    (r["c"] + 1) // 2,
                )
                cube[address] = normalized[i]
    if seen != set(range(96)) or norm_error > m["spec"]["numerical_tolerance"]:
        raise ValueError("Missing unit or normalization mismatch")
    bundle = read(ROOT / m["spec"]["grain_bundle"])
    directions = np.asarray(bundle["directions"], float)
    grain = cube @ directions
    with np.load(
        ROOT / m["spec"]["previous"] / "02/observations.npz", allow_pickle=False
    ) as z:
        native = z["cube"].copy()
    if native.shape != (2, 2, 6, 2, 2, 2, 2, 7, 4):
        raise ValueError("Complete original native field required")
    return m, cube, grain, native, directions


def derive(out):
    """Compute all fixed geometry cases; do not rank models or tune coefficients."""
    m, embed, grain, native, directions = arrays(out)
    views = {
        "E5_full": embed,
        "GRAIN_all15": grain,
        "author_stability": grain[..., :5],
        "response_stability": grain[..., 5:10],
        "principal_components": grain[..., 10:15],
    }
    fields = {
        name: np.array(
            [
                [[M.characters(a[s, w, f]) for f in range(6)] for w in range(2)]
                for s in range(2)
            ]
        )
        for name, a in views.items()
    }
    nf = np.array(
        [
            [[M.characters(native[s, w, f]) for f in range(6)] for w in range(2)]
            for s in range(2)
        ]
    )
    errors = {
        "linear_character_compression": float(
            np.max(abs(fields["E5_full"] @ directions - fields["GRAIN_all15"]))
        )
    }
    strength = m["spec"]["account_interaction_strength"]
    for a, coeff in [(grain, fields["GRAIN_all15"]), (native, nf)]:
        for s, w, f in np.ndindex(2, 2, 6):
            direct = M.account_gap(a[s, w, f], strength)
            err = float(np.max(abs(direct - 2 * strength * coeff[s, w, f, 3])))
            errors["account_identity"] = max(errors.get("account_identity", 0), err)
    rows, lefts, rights, pairing_values = [], [], [], []
    for s, w, sector, route, person, unit, view in itertools.product(
        range(2), range(2), range(1, 4), range(2), range(2), range(2), VIEWS
    ):
        x, y = (
            fields[view][s, w, :, sector],
            nf[s, w, :, sector, route, person, :6, unit],
        )
        comparison, values = M.finite_pairings(x, y)
        rows.append(
            {
                "corpus": ("cause", "opportunity")[s],
                "bank": w,
                "sector": SECTORS[sector],
                "program": ("H", "full")[route],
                "actor": ("A", "B")[person],
                "unit": ("score", "logodds")[unit],
                "view": view,
                "representation_coordinates": x.shape[1],
                "target_coordinates": 6,
                "families": 6,
                "representation_RMS": float(np.sqrt(np.mean(x * x))),
                "target_RMS": float(np.sqrt(np.mean(y * y))),
                **comparison,
            }
        )
        lefts.append(M.centered_gram(x))
        rights.append(M.centered_gram(y))
        pairing_values.append(values)
    if len(rows) != 480 or not all(len(v) == 720 for v in pairing_values):
        raise ValueError("Complete480-case design and all finite pairings required")
    cross = []
    for s, sector in itertools.product(range(2), range(1, 4)):
        channels = [(view, fields[view][s, :, :, sector]) for view in VIEWS]
        channels += [
            (f"native_{route}_{actor}_{unit}", nf[s, :, :, sector, r, p, :6, u])
            for r, route in enumerate(("H", "full"))
            for p, actor in enumerate(("A", "B"))
            for u, unit in enumerate(("score", "logodds"))
        ]
        for name, both in channels:
            result, _ = M.finite_pairings(both[0], both[1])
            scale = float(np.linalg.norm(both[0]) * np.linalg.norm(both[1]))
            cosine = None if scale == 0 else float(np.sum(both[0] * both[1]) / scale)
            cross.append(
                {
                    "corpus": ("cause", "opportunity")[s],
                    "sector": SECTORS[sector],
                    "channel": name,
                    "coordinate_field_cosine": cosine,
                    **result,
                }
            )
    accounts = []
    for s, w, f in np.ndindex(2, 2, 6):
        base = {"corpus": ("cause", "opportunity")[s], "bank": w, "family": f}
        for j, label in enumerate(m["grain_families"]):
            accounts.append(
                {
                    **base,
                    "observer": "GRAIN",
                    "program": "fixed_E5",
                    "actor": "whole_history_not_qualified",
                    "unit": "raw_coordinate",
                    "target": label,
                    "gap": float(2 * strength * fields["GRAIN_all15"][s, w, f, 3, j]),
                }
            )
        for r, p, q, u in np.ndindex(2, 2, 7, 2):
            accounts.append(
                {
                    **base,
                    "observer": "native",
                    "program": ("H", "full")[r],
                    "actor": ("A", "B")[p],
                    "unit": ("score", "logodds")[u],
                    "target": m["previous_targets"][q] if q < 6 else "action_fact",
                    "gap": float(2 * strength * nf[s, w, f, 3, r, p, q, u]),
                }
            )
    summary = {
        "status": "ANALYZED_FINITE_SAME_EVIDENCE_NOT_VALIDITY",
        "records": 96,
        "geometry_cases": len(rows),
        "cross_bank_cases": len(cross),
        "account_rows": len(accounts),
        "fits": 0,
        "new_Qwen_calls": 0,
        "errors": errors,
        "original_scope_complete": False,
        "encoder_status": read(out / "status.json"),
        "case_summaries": [],
    }
    for s, sector, view, route, person, unit in itertools.product(
        ("cause", "opportunity"),
        ("b", "c", "bc"),
        VIEWS,
        ("H", "full"),
        ("A", "B"),
        ("score", "logodds"),
    ):
        selected = [
            r
            for r in rows
            if all(
                r[k] == v
                for k, v in (
                    ("corpus", s),
                    ("sector", sector),
                    ("view", view),
                    ("program", route),
                    ("actor", person),
                    ("unit", unit),
                )
            )
        ]
        summary["case_summaries"].append(
            {
                "corpus": s,
                "sector": sector,
                "view": view,
                "program": route,
                "actor": person,
                "unit": unit,
                "cka_banks": [r["cka"] for r in selected],
                "matched_minus_pairing_mean_banks": [
                    r["matched_minus_pairing_mean"] for r in selected
                ],
                "finite_tail_fractions": [
                    r["finite_pairing_tail_fraction"] for r in selected
                ],
            }
        )
    return summary, rows, cross, accounts, fields, nf, lefts, rights, pairing_values


def analyze(out):
    """Save one immutable analysis of every predeclared representation/program."""
    if (out / "02").exists():
        raise FileExistsError("Do not overwrite an analysis")
    result = derive(out)
    summary, rows, cross, accounts, fields, nf, lefts, rights, values = result
    (out / "02").mkdir()
    table(out / "02/geometry.csv", rows)
    table(out / "02/cross_bank.csv", cross)
    table(out / "02/account_gaps.csv", accounts)
    np.savez(out / "02/fields.npz", native=nf, **fields)
    np.savez(
        out / "02/gram_pairings.npz",
        left=np.array(lefts),
        right=np.array(rights),
        pairing_values=np.array(values),
    )
    write(out / "02/summary.json", summary)
    write(
        out / "02/seal.json", {str(p.resolve()): sha(p) for p in (out / "02").iterdir()}
    )
    print("ANALYZED HM137", len(rows), "cases; no fits", summary["errors"], flush=True)


def verify(out):
    """Read-only replay plus alternative character and coordinate CKA formulas."""
    m, _, grain, native, _ = arrays(out)
    for p, expected in read(out / "02/seal.json").items():
        if sha(p) != expected:
            raise ValueError("Changed analysis artifact")
    summary, rows, cross, accounts, fields, nf, _, _, values = derive(out)
    if summary != read(out / "02/summary.json"):
        raise ValueError("Summary replay mismatch")
    for filename, expected in (
        ("geometry", rows),
        ("cross_bank", cross),
        ("account_gaps", accounts),
    ):
        with (out / f"02/{filename}.csv").open(newline="") as f:
            saved = list(csv.DictReader(f))
        if saved != [
            {k: "" if v is None else str(v) for k, v in row.items()} for row in expected
        ]:
            raise ValueError("Table replay mismatch: " + filename)
    errors = dict(summary["errors"])
    for cube, coeff in ((grain, fields["GRAIN_all15"]), (native, nf)):
        for s, w, f in np.ndindex(2, 2, 6):
            direct = []
            for j in range(4):
                terms = []
                for b, c in itertools.product((-1, 1), repeat=2):
                    weight = (1, b, c, b * c)[j]
                    terms.append(weight * cube[s, w, f, (b + 1) // 2, (c + 1) // 2] / 4)
                direct.append(sum(terms))
            err = float(np.max(abs(np.array(direct) - coeff[s, w, f])))
            errors["alternative_characters"] = max(
                errors.get("alternative_characters", 0), err
            )
            err = float(np.max(abs(M.reconstruct(coeff[s, w, f]) - cube[s, w, f])))
            errors["all_corner_reconstruction"] = max(
                errors.get("all_corner_reconstruction", 0), err
            )
    alternate = 0.0
    for row in rows:
        s, w = ("cause", "opportunity").index(row["corpus"]), row["bank"]
        j = SECTORS.index(row["sector"])
        r, p, u = (
            ("H", "full").index(row["program"]),
            ("A", "B").index(row["actor"]),
            ("score", "logodds").index(row["unit"]),
        )
        x, y = fields[row["view"]][s, w, :, j], nf[s, w, :, j, r, p, :6, u]
        xc, yc = x - x.mean(0), y - y.mean(0)
        direct = np.linalg.norm(xc.T @ yc) ** 2 / (
            np.linalg.norm(xc.T @ xc) * np.linalg.norm(yc.T @ yc)
        )
        alternate = max(alternate, abs(float(direct) - row["cka"]))
    errors["alternative_coordinate_CKA"] = alternate
    with np.load(out / "02/gram_pairings.npz", allow_pickle=False) as z:
        errors["all_pairing_replay"] = float(
            np.max(abs(np.array(values) - z["pairing_values"]))
        )
    if max(errors.values()) > m["spec"]["numerical_tolerance"]:
        raise ValueError("Predeclared numerical tolerance failed; no rescue")
    result = {
        "status": "VERIFIED_FINITE_HM137_NOT_GLOBAL",
        "errors": errors,
        "geometry_cases": len(rows),
        "finite_pairing_values": len(rows) * 720,
        "cross_bank_cases": len(cross),
        "account_rows": len(accounts),
        "same_agent_numerical_audit": True,
        "original_scope_complete": False,
    }
    path = out / "verification.json"
    if path.exists():
        if read(path) != result:
            raise ValueError("Previous verification differs")
    else:
        write(path, result)
    print(json.dumps(result), flush=True)


def main():
    """Run one phase; cold collection does not resume uncertain calls."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("prepare", "collect", "analyze", "verify"), required=True
    )
    args = parser.parse_args()
    out = ROOT / read(SPEC)["out"]
    globals()[args.phase](out)


if __name__ == "__main__":
    main()
