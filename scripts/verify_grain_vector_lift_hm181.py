#!/usr/bin/env python3
"""Alternate actual-vector accounts and literal cold replay;not a blind audit."""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_configuration_kernel_hm144 as K
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_vector_lift_hm181 as S


def main():
    """Verify every source event and full footprint without selecting outcomes."""
    out = S.DEFAULT
    spec = R.check(out, ("compiled_seal.json", "scored_seal.json"))["config"]
    z = R.prepared(out)
    with np.load(out / "source_footprints.npz", allow_pickle=False) as file:
        saved = {k: file[k] for k in file.files}
    parent = ROOT / spec["parent"]
    b = R.read(parent / "source_bundle.json")
    with np.load(parent / "source_trade_components.npz", allow_pickle=False) as file:
        changes = file["changes"]
    profiles = {tuple(p): j for j, p in enumerate(b["profiles"])}
    h, direction, _, _, _, _ = K.load_numeric()
    coordinate_error = float(
        np.max(
            np.abs(
                (h @ direction)[z["source"], :24].reshape(386, 2, 12, 15)
                - z["source_coordinates"]
            )
        )
    )
    m, e, coarse = (
        np.zeros_like(saved["moments"]),
        np.zeros_like(saved["e5"]),
        np.zeros_like(saved["coarse"]),
    )
    weight_error, pushed_error = 0.0, 0.0
    for v in range(2):
        pushed = np.zeros((32, 8, 8, 8))
        for u in range(386):
            g = profiles[tuple(z["targets"][u])]
            w = np.zeros((4, 4, 4, 5))
            for aa, bb, cc in itertools.product(range(4), repeat=3):
                ids = (aa, 4 + bb, 8 + cc)
                a, bbcell, c = z["cells"][u, v, list(ids)]
                f = b["profile_tables"][v][1][g][a][bbcell][c]
                d = changes[v, :, g, a, bbcell, c] / (386 * 64 * f)
                w[aa, bb, cc] = d
                pushed[g, a, bbcell, c] += 1 / (386 * 64) - d.sum()
                weight_error = max(
                    weight_error,
                    float(
                        np.max(
                            np.abs(d - saved["weights"][v, u, aa * 16 + bb * 4 + cc])
                        )
                    ),
                )
                sign = np.r_[1, 2 * z["targets"][u].astype(float) - 1]
                coarse[v, :, :, (a * 8 + bbcell) * 8 + c] += d[:, None] * sign[None, :]
            aug = np.concatenate(
                (np.ones((12, 1)), z["source_coordinates"][u, v]), axis=1
            ).reshape(3, 4, 16)
            vphi = np.einsum("abcq,ai,bj,ck->qijk", w, *aug, optimize=True).reshape(
                5, 4096
            )
            sign = np.r_[1, 2 * z["targets"][u].astype(float) - 1]
            m[v] += vphi[:, None, :] * sign[None, :, None]
            for role, axes in enumerate(((1, 2), (0, 2), (0, 1))):
                local = (
                    w.sum(axis=axes).T
                    @ h[z["source"][u], 12 * v + 4 * role : 12 * v + 4 * role + 4]
                )
                e[v, :, :, role] += local[:, None, :] * sign[None, :, None]
        pushed_error = max(
            pushed_error,
            float(np.max(np.abs(pushed - np.asarray(b["profile_tables"])[v, 0]))),
        )
    witnesses = np.asarray(R.read(out / "vector_bundle.json")["witnesses"])
    result = {
        r["frame_key"]: r
        for r in map(json.loads, (out / "cold_output.jsonl").read_text().splitlines())
    }
    cold_error, triples = 0.0, 0
    for line in (out / "cold_input.jsonl").read_text().splitlines():
        row = json.loads(line)
        roles = [
            [e for e in row["events"] if e["slot"] // 4 == role] for role in range(3)
        ]
        total = np.zeros((2, 5))
        for a, bb, c in itertools.product(*roles):
            phi = np.kron(
                np.kron(np.r_[1, a["coordinates"]], np.r_[1, bb["coordinates"]]),
                np.r_[1, c["coordinates"]],
            )
            total += np.einsum("vqk,k->vq", witnesses, phi) / 64
            triples += 1
        expected = result[row["frame_key"]]
        cold_error = max(
            cold_error, float(np.max(np.abs(total - expected["retained_readings"])))
        )
        if expected["triples"] != int(np.prod([len(r) for r in roles])):
            raise ValueError("Cold opportunity changed")
        if not expected["triples"] and expected["conditional_readings"] is not None:
            raise ValueError("No-opportunity reading assigned a personality value")
    receipt = {
        "source_triples": 49408,
        "cold_records": len(result),
        "cold_triples": triples,
        "actual_coordinate_error": coordinate_error,
        "weight_replay_error": weight_error,
        "H_pushforward_error": pushed_error,
        "tensor_replay_error": float(np.max(np.abs(m - saved["moments"]))),
        "E5_replay_error": float(np.max(np.abs(e - saved["e5"]))),
        "coarse_replay_error": float(np.max(np.abs(coarse - saved["coarse"]))),
        "cold_replay_error": cold_error,
        "same_agent_not_blind": True,
    }
    if (
        max(v for k, v in receipt.items() if k.endswith("error"))
        > spec["comparison_tolerance"]
    ):
        raise ValueError("Alternate fixed observation audit failed")
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
