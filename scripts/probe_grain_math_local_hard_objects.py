#!/usr/bin/env python3
"""Frozen-program hard object controls; result-guided, not a refitting search."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from run_grain_math_local import DEFAULT_OUT as SOURCE
from run_grain_math_local import TARGETS, load_program, material_text
from run_grain_math_response import MODELS, ROOT, encode, sha, write_csv, write_json
from sklearn.metrics import roc_auc_score

from grain_research.construction import math_local as L

CORES = [
    [
        "organize",
        "A group meeting was described in a personal notebook.",
        "organize the people into a shared plan for their work",
        "organize my notebook pages describing the people in the group",
        "check the reported meeting dates against the signed attendance log",
        "draw a check mark beside a picture of the attendance log",
    ],
    [
        "coordinate",
        "A poster depicted a team preparing a demonstration.",
        "coordinate the team members and agree on their responsibilities",
        "coordinate the colors on a poster depicting the team members",
        "compare the reported test figures against the original measurements",
        "compare the decorative colors on covers of the measurement reports",
    ],
    [
        "arrange",
        "Photographs showed a group preparing a public event.",
        "arrange a joint planning session with the group members",
        "arrange printed photographs of the group members on my shelf",
        "verify the reported event expenses against the supplier invoices",
        "write the word verify in calligraphy on an invoice folder",
    ],
    [
        "bring",
        "A collection of images showed several volunteers at work.",
        "bring the volunteers together and allocate their tasks",
        "bring images of the volunteers together in a collage",
        "cross-check the claimed participant totals using the registration records",
        "paint a cross-shaped pattern on a box containing registration records",
    ],
    [
        "set_up",
        "A software screen displayed an illustration of a team meeting.",
        "set up a shared discussion with the team to divide the work",
        "set up my screen layout around an illustration of the team discussion",
        "audit the reported item counts against the signed inventory sheets",
        "design an audit logo for the cover of the inventory sheets",
    ],
    [
        "plan",
        "A book described several people working on a project.",
        "plan a joint work session with the project participants",
        "plan the layout of my personal drawing of the project participants",
        "confirm the project dates using two independent archive records",
        "stamp the word confirmed onto a decorative bookmark shaped like an archive record",
    ],
    [
        "allocate",
        "My desktop contained pictures of a team preparing a shipment.",
        "allocate the shipment responsibilities through a team discussion",
        "allocate space on my desktop for pictures of the shipment team",
        "validate the reported shipment quantities against the original dispatch logs",
        "decorate a dispatch folder with the word validation",
    ],
    [
        "schedule",
        "A personal scrapbook described workers preparing an exhibition.",
        "schedule a joint planning meeting with the exhibition workers",
        "schedule a private hour to draw pictures of the exhibition workers",
        "reconcile the reported visitor total with the original ticket records",
        "draw a reconciliation symbol on an ornamental ticket record cover",
    ],
]


def probe_metrics(score, z, bits):
    """Report only estimands present in this smaller, all-own/actual diagnostic."""
    q = bits[:, 0] == 1
    decision = L.qualify(z)
    result = {
        "unit_rmse": float(np.sqrt(np.mean((score - q * bits[:, 1]) ** 2))),
        "qualified_rmse": float(np.sqrt(np.mean((score[q] - bits[q, 1]) ** 2))),
        "unqualified_rms": float(np.sqrt(np.mean(score[~q] ** 2))),
        "qualification_tp": int(np.sum(decision & q)),
        "qualification_fp": int(np.sum(decision & ~q)),
        "qualification_fn": int(np.sum(~decision & q)),
        "qualification_tn": int(np.sum(~decision & ~q)),
        "relevant_polarity_gain": float(2 * np.mean(score[q] * bits[q, 1])),
        "irrelevant_polarity_gain": float(2 * np.mean(score[~q] * bits[~q, 1])),
        "other_polarity_gain": None,
        "hypothetical_polarity_gain": None,
    }
    for j, name in enumerate(L.FACTOR_NAMES):
        result[name + "_auc"] = (
            float(roc_auc_score(bits[:, j] == 1, z[:, j]))
            if len(np.unique(bits[:, j])) == 2
            else None
        )
        result[name + "_rmse"] = float(np.sqrt(np.mean((z[:, j] - bits[:, j]) ** 2)))
    return result


def run(out, source, device):
    """Encode fresh fixed controls, then evaluate source-frozen observers unchanged."""
    paths = [Path(__file__), source / "02/model_freeze.json"]
    signature = {str(p): sha(p) for p in paths}
    dest = out / "00"
    contract = dest / "protocol_before_run.json"
    bank = []
    for target, family, k, p in product(range(2), range(8), (-1, 1), (-1, 1)):
        bits = [k, p, 1, 1, 1]
        bank.append(
            {
                "target": TARGETS[target],
                "family": family,
                "bits": bits,
                "text": material_text(CORES[family], target, bits),
            }
        )
    if contract.exists():
        old = json.loads(contract.read_text())
        if (
            old["signature"] != signature
            or sha(dest / "material_bank.jsonl") != old["bank_sha256"]
        ):
            raise ValueError("hard-control protocol changed; choose new output")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "material_bank.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in bank)
        )
        write_json(
            dest / "reader_registry.json",
            json.loads((source / "00/reader_registry.json").read_text()),
        )
        write_json(
            contract,
            {
                "signature": signature,
                "bank_sha256": sha(dest / "material_bank.jsonl"),
                "result_guided": True,
                "refitting": False,
                "new_texts": len(bank),
                "new_encodings": 2 * len(bank),
                "qualification": "same action word, different object; authored semantic cases, not human gold",
                "source_bundle_hashes": {
                    p.name: sha(p)
                    for p in (source / "02/bundles").glob("*.npz")
                    if not p.name.startswith("._")
                },
            },
        )
    metric_rows, object_rows = [], []
    for reader in MODELS:
        encode(out, reader, device)
        x = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        for target in TARGETS:
            m = np.array([r["target"] == target for r in bank])
            rows = [r for r, keep in zip(bank, m, strict=True) if keep]
            bits = np.array([r["bits"] for r in rows])
            model = load_program(source / f"02/bundles/{reader}_{target}.npz")
            scores, z = L.score_programs(model, x[m])
            for arm in ("joint", "declared_product", "direct_ridge", "rbf"):
                metric_rows.append(
                    {
                        "reader": reader,
                        "target": target,
                        "arm": arm,
                        **probe_metrics(scores[arm], z, bits),
                    }
                )
            for i, r in enumerate(rows):
                object_rows.append(
                    {
                        "reader": reader,
                        "target": target,
                        "family": r["family"],
                        "relevance_expected": r["bits"][0],
                        "polarity": r["bits"][1],
                        "relevance_readout": float(z[i, 0]),
                        "qualifies": bool(L.qualify(z[i : i + 1])[0]),
                        "joint_score": float(scores["joint"][i]),
                        "product_score": float(scores["declared_product"][i]),
                    }
                )
    write_csv(out / "02/metrics.csv", metric_rows)
    write_csv(out / "02/object_scores.csv", object_rows)
    write_json(out / "02/summary.json", {"metrics": metric_rows, "refitted_models": 0})
    print(f"hard-object diagnostic complete: {len(bank)} public controls; no refit")


def main():
    """Run a single cold diagnostic without API, generation or neural fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/local_hard_objects_r2",
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    run(args.out, args.source, args.device)


if __name__ == "__main__":
    main()
