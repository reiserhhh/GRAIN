#!/usr/bin/env python3
"""HM106: exact joint observation construction on actual frozen text carriers."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_reading as M
from scripts import analyze_grain_qualified_response_sectors_hm105 as S
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_qualified_transport_hm105 as Q
from scripts import run_grain_text_response_hm104 as T

SPEC = ROOT / "configs/grain_joint_reading_hm106.json"
CODE = (
    "configs/grain_joint_reading_hm106.json",
    "docs/GRAIN_JOINT_READING_PLAN_HM106.md",
    "grain_research/construction/math_joint_reading.py",
    "scripts/run_grain_joint_reading_hm106.py",
    "tests/test_grain_math_joint_reading.py",
    "scripts/analyze_grain_qualified_response_sectors_hm105.py",
)
LAWS = {
    "uniform": np.ones(4) / 4,
    "unequal": np.array([0.1, 0.2, 0.3, 0.4]),
    "joint_plus": (1 + 0.5 * M.BASIS[3]) / 4,
    "joint_minus": (1 - 0.5 * M.BASIS[3]) / 4,
}


def source_check(source):
    """Authenticate actual public captures and already-opened prior results."""
    Q.check(source)
    for seal in ("01/raw_seal.json", "output_seal.json", "response_sector_seal.json"):
        H.verify_hashes(source, H.load(source / seal))
    if (
        H.load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM105_NOT_GLOBAL"
    ):
        raise ValueError("Verified actual HM105 source required")


def prepare(out):
    """Freeze this new account without rerunning or rewriting inherited captures."""
    if out.exists():
        raise FileExistsError("Preserve previous and failed packets")
    spec = H.load(SPEC)
    source = ROOT / spec["source"]
    source_check(source)
    rows = H.load(source / "00/rows.json")
    if len(rows) != 468:
        raise ValueError("Full frozen source battery required")
    (out / "00").mkdir(parents=True)
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "classification": spec["classification"],
            "code": {name: H.sha(ROOT / name) for name in CODE},
            "source": {
                name: H.sha(source / name)
                for name in (
                    "00/manifest.json",
                    "00/rows.json",
                    "00/reader.json",
                    "01/raw_seal.json",
                    "output_seal.json",
                    "verification.json",
                    "response_sector_seal.json",
                )
            },
            "all_source_captures_authenticated": len(rows),
            "used_actual_corner_captures": 288,
            "complete_corner_groups": 72,
            "positive_author_laws": {name: p.tolist() for name, p in LAWS.items()},
            "new_neural_calls": 0,
            "new_predictor_fits": 0,
        },
    )
    print(
        "HM106 PREPARED:72 complete groups,288 actual corners,no new inference",
        flush=True,
    )


def check(out):
    """Fail if a source or sealed implementation changed between phases."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"])
    source = ROOT / manifest["spec"]["source"]
    H.verify_hashes(source, manifest["source"])
    source_check(source)
    return manifest, source


def cases(source, spec):
    """Yield actual aligned role/content corners in fixed complete design order."""
    rows = H.load(source / "00/rows.json")
    reader = H.load(source / "00/reader.json")
    lookup = {
        tuple(
            row[k] for k in ("bank", "bridge", "carrier", "axis", "level", "owner")
        ): row
        for row in rows
    }
    for bank, bridge, carrier, action in itertools.product(
        spec["banks"], spec["bridges"], spec["carriers"], spec["actions"]
    ):
        picked = [
            lookup[bank, bridge, carrier, action, level, owner]
            for level, owner in ((2, 2), (0, 2), (2, 0), (0, 0))
        ]
        coordinates = [
            (r["current"], [r["input_ids"][i] for i in r["current"]]) for r in picked
        ]
        if any(x != coordinates[0] for x in coordinates):
            raise ValueError("Current token coordinates change within a corner group")
        raw = [T.raw_unit(source, r) for r in picked]
        keys, values = (np.stack([r[name] for r in raw]) for name in ("k", "v"))
        q = np.asarray(reader["positions"][str(max(picked[0]["current"]))], float)
        directions = np.asarray(reader["directions"], float)
        yield (bank, bridge, carrier, action), keys, values, q, directions


def cosine(a, b):
    """Leave zero-vector alignments undefined rather than treating them as failure."""
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.vdot(a, b) / scale) if scale > 1e-24 else None


def finite_stats(values):
    """Summarize all defined comparisons, explicitly retaining undefined counts."""
    present = [x for x in values if x is not None]
    return {
        **(T.stat(present) if present else {"n": 0}),
        "undefined": len(values) - len(present),
    }


def account(component, total):
    """Signed alignment is not an additive variance fraction or psychological r."""
    denom = float(np.sum(total**2))
    return {
        "rms": float(np.sqrt(np.mean(component**2))),
        "signed_alignment": float(np.vdot(component, total) / denom) if denom else None,
        "norm_ratio": float(np.linalg.norm(component) / np.sqrt(denom))
        if denom
        else None,
    }


def analyze(out):
    """Construct all signed products and author scores without fitting anything."""
    start = time.monotonic()
    manifest, source = check(out)
    spec, dest = manifest["spec"], out / "01"
    dest.mkdir()
    targets = H.load(source / "00/reader.json")["targets"]
    labels, coefficients, products, query_products = [], [], [], []
    character_rows, joint_rows, author_rows = [], [], []
    errors = {"character_rebuild": 0.0, "query_rebuild": 0.0, "author_direct": 0.0}
    for case, k, v, q, d in cases(source, spec):
        p, c = M.primitives(k, v, q, d)
        fields = {
            "native": p,
            "target_shared": np.broadcast_to(p.mean(axis=2, keepdims=True), p.shape),
            "uniform": np.full_like(p, 1 / p.shape[1]),
        }
        native_products = M.product_law(p, c)
        query_parts = M.query_product_law(p, c)
        errors["query_rebuild"] = max(
            errors["query_rebuild"],
            float(np.max(np.abs(query_parts.sum(0) - native_products))),
        )
        modes = []
        context = dict(zip(("bank", "bridge", "carrier", "action"), case, strict=True))
        for mode, w in fields.items():
            a = M.product_law(w, c).sum(1)
            f = (w * c).sum(axis=(1, 3))
            errors["character_rebuild"] = max(
                errors["character_rebuild"], float(np.max(np.abs(M.reconstruct(a) - f)))
            )
            modes.append(a)
            for char, name in enumerate(M.CHARACTERS):
                for target, label in enumerate(targets):
                    character_rows.append(
                        {
                            **context,
                            "weight_mode": mode,
                            "character": name,
                            "target": label,
                            "coefficient": float(a[char, target]),
                        }
                    )
            for law_name, law in LAWS.items():
                score = M.author_score(a, law)
                errors["author_direct"] = max(
                    errors["author_direct"], float(np.max(np.abs(score - law @ f)))
                )
                for target, label in enumerate(targets):
                    author_rows.append(
                        {
                            **context,
                            "weight_mode": mode,
                            "author_law": law_name,
                            "target": label,
                            "score": float(score[target]),
                        }
                    )
        for channel, name in enumerate(M.QUERY_CHANNELS):
            for r, product in enumerate(M.JOINT_PRODUCTS):
                for target, label in enumerate(targets):
                    joint_rows.append(
                        {
                            **context,
                            "query_channel": name,
                            "product": product,
                            "target": label,
                            "interaction": float(
                                4 * query_parts[channel, 3, r, target]
                            ),
                        }
                    )
        labels.append(case)
        coefficients.append(modes)
        products.append(native_products)
        query_products.append(query_parts)
    labels, coefficients, products, query_products = map(
        np.asarray, (labels, coefficients, products, query_products)
    )
    with (dest / "laws.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            labels=labels,
            coefficients=coefficients,
            products=products,
            query_products=query_products,
        )
    H.table(dest / "characters.csv", character_rows)
    H.table(dest / "joint_products.csv", joint_rows)
    H.table(dest / "author_scores.csv", author_rows)

    def table(bank, bridge, carrier, mode=0):
        ix = np.all(labels[:, :3] == (bank, bridge, carrier), axis=1)
        return 4 * coefficients[ix, mode, 3]

    summary = {
        "packet": "HM106",
        "status": "COMPUTED_EXPLORATORY_NOT_GLOBAL",
        "classification": spec["classification"],
        "new_neural_calls": 0,
        "new_fits": 0,
        "complete_corner_groups": len(labels),
        "actual_corner_captures_used": 4 * len(labels),
        "seconds": time.monotonic() - start,
        "errors": errors,
        "banks": {},
        "original_scope_complete": False,
    }
    transport, carrier_rows = [], []
    for bank in spec["banks"]:
        ix = labels[:, 0] == bank
        total = 4 * coefficients[ix, 0, 3]
        summary["banks"][str(bank)] = {
            "native_interaction": account(total, total),
            "joint_products": {
                name: account(4 * products[ix, 3, r], total)
                for r, name in enumerate(M.JOINT_PRODUCTS)
            },
            "query_channels": {
                name: account(4 * query_products[ix, channel, 3].sum(1), total)
                for channel, name in enumerate(M.QUERY_CHANNELS)
            },
            "reading_modes": {},
        }
        for mode, name in enumerate(spec["weight_modes"]):
            current = 4 * coefficients[ix, mode, 3]
            record = account(current, total)
            record["coupled_transport_cosine"] = finite_stats(
                [
                    cosine(
                        S.sectors(table(0, b, c, mode))["coupled"],
                        S.sectors(table(bank, b, c, mode))["coupled"],
                    )
                    for b, c in itertools.product(spec["bridges"], spec["carriers"])
                ]
            )
            record["full_transport_cosine"] = finite_stats(
                [
                    cosine(table(0, b, c, mode), table(bank, b, c, mode))
                    for b, c in itertools.product(spec["bridges"], spec["carriers"])
                ]
            )
            summary["banks"][str(bank)]["reading_modes"][name] = record
            for bridge, carrier in itertools.product(spec["bridges"], spec["carriers"]):
                a, s = (
                    table(bank, bridge, carrier, mode),
                    table(0, bridge, carrier, mode),
                )
                transport.append(
                    {
                        "bank": bank,
                        "bridge": bridge,
                        "carrier": carrier,
                        "mode": name,
                        "full_cosine": cosine(a, s),
                        "coupled_cosine": cosine(
                            S.sectors(a)["coupled"], S.sectors(s)["coupled"]
                        ),
                        "rms": float(np.sqrt(np.mean(a**2))),
                    }
                )
            for bridge in spec["bridges"]:
                a, b = table(bank, bridge, 0, mode), table(bank, bridge, 1, mode)
                carrier_rows.append(
                    {
                        "bank": bank,
                        "bridge": bridge,
                        "mode": name,
                        "interaction_difference_rms": float(
                            np.sqrt(np.mean((a - b) ** 2))
                        ),
                        "coupled_carrier_cosine": cosine(
                            S.sectors(a)["coupled"], S.sectors(b)["coupled"]
                        ),
                        "boundary": "Carrier tokens,length,and registered query position may all change",
                    }
                )
    H.table(dest / "transport.csv", transport)
    H.table(dest / "carrier_visibility.csv", carrier_rows)
    H.write(out / "summary.json", summary)
    H.write(
        out / "output_seal.json",
        {**H.seals(out, dest), "summary.json": H.sha(out / "summary.json")},
    )
    print("HM106 ANALYZED", summary, flush=True)


def verify(out):
    """Replay direct per-head scores and individual products by alternate arithmetic."""
    manifest, source = check(out)
    H.verify_hashes(out, H.load(out / "output_seal.json"))
    with np.load(out / "01/laws.npz", allow_pickle=False) as z:
        saved = {name: np.array(z[name]) for name in z.files}
    with (source / "03/interactions.csv").open() as stream:
        old = {
            (
                int(r["bank"]),
                int(r["bridge"]),
                int(r["carrier"]),
                int(r["axis"]),
                r["target"],
            ): float(r["actual"])
            for r in csv.DictReader(stream)
            if r["source_span"] == "s7"
        }
    targets = H.load(source / "00/reader.json")["targets"]
    basis = np.array(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], float
    )
    errors = {
        name: 0.0
        for name in (
            "direct_read",
            "all_products",
            "query_products",
            "old_native",
            "parseval",
            "author",
            "uniform_joint_value",
        )
    }
    for index, (case, k, v, q, d) in enumerate(cases(source, manifest["spec"])):
        p, c = (
            np.empty((4, len(k[0]), len(q), q.shape[1])),
            np.empty((4, len(k[0]), len(q), q.shape[1])),
        )
        for corner, target, head in itertools.product(
            range(4), range(len(q)), range(q.shape[1])
        ):
            kh = head // (q.shape[1] // k.shape[2])
            l = k[corner, :, kh] @ q[target, head]
            weight = np.exp(l - l.max())
            p[corner, :, target, head] = weight / weight.sum()
            c[corner, :, target, head] = v[corner, :, kh] @ d[target, head]
        weights = [
            p,
            np.broadcast_to(p.mean(axis=2, keepdims=True), p.shape),
            np.full_like(p, 1 / p.shape[1]),
        ]
        for mode, weight in enumerate(weights):
            f = np.sum(weight * c, axis=(1, 3))
            coeff = basis @ f / 4
            errors["direct_read"] = max(
                errors["direct_read"],
                float(np.max(np.abs(coeff - saved["coefficients"][index, mode]))),
            )
            errors["parseval"] = max(
                errors["parseval"],
                float(np.max(np.abs(np.mean(f**2, axis=0) - np.sum(coeff**2, axis=0)))),
            )
            for law in LAWS.values():
                errors["author"] = max(
                    errors["author"],
                    float(np.max(np.abs(law @ f - (basis @ law) @ coeff))),
                )
        ph, ch = [
            np.stack([np.sum(a * b[:, None, None, None], axis=0) / 4 for b in basis])
            for a in (p, c)
        ]
        for j, r in itertools.product(range(4), range(4)):
            term = np.sum(ph[r] * ch[r ^ j], axis=(0, 2))
            errors["all_products"] = max(
                errors["all_products"],
                float(np.max(np.abs(term - saved["products"][index, j, r]))),
            )
        pm, cm = p.mean(axis=2, keepdims=True), c.mean(axis=2, keepdims=True)
        for channel, (wp, vc) in enumerate(
            ((pm, cm), (pm, c - cm), (p - pm, cm), (p - pm, c - cm))
        ):
            for j, r in itertools.product(range(4), range(4)):
                a = np.sum(wp * basis[r, :, None, None, None], axis=0) / 4
                b = np.sum(vc * basis[r ^ j, :, None, None, None], axis=0) / 4
                term = np.sum(a * b, axis=(0, 2))
                errors["query_products"] = max(
                    errors["query_products"],
                    float(
                        np.max(
                            np.abs(term - saved["query_products"][index, channel, j, r])
                        )
                    ),
                )
        actual = 4 * saved["coefficients"][index, 0, 3]
        inherited = np.array([old[*case, label] for label in targets])
        errors["old_native"] = max(
            errors["old_native"], float(np.max(np.abs(actual - inherited)))
        )
        errors["uniform_joint_value"] = max(
            errors["uniform_joint_value"],
            float(
                np.max(
                    np.abs(
                        saved["coefficients"][index, 2, 3]
                        - ch[3].mean(axis=0).sum(axis=-1)
                    )
                )
            ),
        )
    if max(errors.values()) > manifest["spec"]["tolerance"]:
        H.write(out / "verification_failure.json", errors)
        raise ValueError("Keep failed account;do not tune it into passing")
    result = {
        "status": "VERIFIED_FINITE_HM106_NOT_GLOBAL",
        "cases": len(saved["labels"]),
        "errors": errors,
        "same_agent_alternate_arithmetic": True,
        "new_neural_calls": 0,
        "new_predictor_fits": 0,
        "original_scope_complete": False,
    }
    H.write(out / "verification.json", result)
    print("HM106 VERIFIED", result, flush=True)


def main():
    """Run separately sealed finite phases, never overwrite frozen results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "analyze", "verify"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/joint_reading_law_r1",
    )
    args = parser.parse_args()
    globals()[args.phase](args.output.resolve())


if __name__ == "__main__":
    main()
