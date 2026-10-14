#!/usr/bin/env python3
"""Independent reconstruction of frozen identity results and observer sanity."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_grain_predecessor_reference import digest, write

OUT = ROOT / "results/grain_vector_mechanisms/mechanism_todo_v1/evidence_identity_r1"
MODES = ["intention", "action_report", "preference", "ability", "experience", "request"]


def observer_sanity(out):
    """One post-result diagnostic, no query selection or main-result replacement."""
    from scripts.run_grain_evidence_calibration import infer_pairs

    destination = out / "observer_sanity.csv"
    if destination.exists():
        raise FileExistsError("Do not repeat the diagnostic")
    cases = [
        (
            "A dog is running in a park.",
            "An animal is running in a park.",
            "entailment",
        ),
        (
            "A dog is running in a park.",
            "There is no animal running in the park.",
            "contradiction",
        ),
        ("A dog is running in a park.", "The dog's owner is a teacher.", "neutral"),
        (
            "Respondent: I asked you for help yesterday.",
            "The respondent asked for help yesterday.",
            "entailment",
        ),
        (
            "Respondent: I asked you for help yesterday.",
            "The respondent did not ask for help yesterday.",
            "contradiction",
        ),
        (
            "Respondent: I asked you for help yesterday.",
            "The respondent reports an action that they themselves already performed.",
            "entailment",
        ),
    ]
    frame = pd.DataFrame(
        [
            {"audit_id": str(i), "sentence": p, "item_text": h, "expected": e}
            for i, (p, h, e) in enumerate(cases)
        ]
    )
    # Written before inference: a diagnostic control, not a new rescue experiment.
    frame.to_csv(out / "observer_sanity_inputs.csv", index=False)
    start = time.monotonic()
    result = infer_pairs(frame)
    frame.merge(result, on="audit_id", validate="one_to_one").to_csv(
        destination, index=False
    )
    write(
        out / "observer_sanity_manifest.json",
        {
            "pairs": len(frame),
            "seconds": time.monotonic() - start,
            "inputs_sha256": digest(out / "observer_sanity_inputs.csv"),
            "output_sha256": digest(destination),
            "post_result": True,
            "changes_main_outputs": False,
        },
    )


def audit(out):
    """Replay numeric accounts without using the production decoder functions."""
    freeze = json.loads((out / "freeze.json").read_text())
    for base, key in ((ROOT, "files"), (out, "inputs")):
        for path, sha in freeze[key].items():
            assert digest(base / path) == sha, path
    observations = json.loads((out / "observations.json").read_text())
    for name, sha in observations["files"].items():
        assert digest(out / name) == sha
    public = pd.read_csv(out / "public_bank.csv").fillna("")
    private = pd.read_csv(out / "private_natural_bank.csv").fillna("")
    pairs = pd.read_csv(out / "private_nli_pairs.csv")
    assert pairs.audit_id.is_unique and len(pairs) == 800
    probabilities = pairs[["p_entailment", "p_neutral", "p_contradiction"]].to_numpy()
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=2e-6)
    assert (probabilities >= 0).all()
    raw = pd.concat([public, private]).set_index("record_id")
    for r in pairs.itertuples():
        p = raw.loc[r.record_id]
        expected = "Respondent: " + p.reply
        if r.view == "context" and p.partner:
            expected = "Partner: " + p.partner + "\n" + expected
        assert r.sentence == expected
    from scripts.run_grain_multiview_address import load_material

    _, _, natural_raw, _, _ = load_material()
    for r in private.itertuples():
        assert (
            natural_raw[int(r.source_row)]["body"][
                int(r.source_start) : int(r.source_end)
            ]
            == r.reply
        )
    main = public[public.part.isin(["source", "held"])]
    source = main[main.part == "source"]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit(source.reply)
    reading = pairs.pivot(
        index=["record_id", "view"], columns="query", values="p_entailment"
    )
    arrays = {
        "word_tfidf": tfidf.transform(main.reply).toarray(),
        "typed_queries": np.stack(
            [
                reading.loc[(r.record_id, "context"), MODES].to_numpy()
                for r in main.itertuples()
            ]
        ),
        "content_query": np.array(
            [
                [reading.loc[(r.record_id, "context"), "content"]]
                for r in main.itertuples()
            ]
        ),
    }
    stored = json.loads((out / "source_decoder.json").read_text())
    bases = {}
    for method, arr in arrays.items():
        basis = np.stack(
            [arr[(main.part == "source") & (main.expected == m)].mean(0) for m in MODES]
        )
        np.testing.assert_allclose(basis, stored[method]["source_means"], atol=1e-14)
        bases[method] = basis
    config = json.loads((out / "config.json").read_text())
    rng = np.random.default_rng(config["seed"])
    truth = rng.dirichlet(np.full(6, 1.5), config["simulated_authors"])
    preds = pd.read_csv(out / "simulated_composition_predictions.csv")
    max_kkt = 0
    checks = 0
    for person, pi in enumerate(truth):
        identities = rng.choice(6, config["utterances_per_author"], p=pi)
        words = rng.integers(0, 2, len(identities))
        for part in ("source", "held"):
            ix = [
                main.index[(main.expected == MODES[m]) & (main.part == part)][w]
                for m, w in zip(identities, words, strict=True)
            ]
            for method, arr in arrays.items():
                selected = (
                    preds[
                        (preds.person == person)
                        & (preds.part == part)
                        & (preds.method == method)
                    ]
                    .set_index("mode")
                    .loc[MODES]
                )
                x = selected.estimate.to_numpy()
                b = bases[method]
                y = arr[ix].mean(0)
                np.testing.assert_allclose(selected.target, pi, atol=1e-14)
                np.testing.assert_allclose(
                    selected.observed_fraction,
                    np.bincount(identities, minlength=6) / len(identities),
                    atol=1e-14,
                )
                assert abs(x.sum() - 1) < 1e-8 and x.min() > -1e-8
                gradient = 2 * b @ (x @ b - y)
                # Convex simplex KKT: all occupied components have minimum gradient.
                violation = max(
                    float(np.ptp(gradient[x > 1e-7])),
                    float(np.max(gradient[x > 1e-7]) - np.min(gradient)),
                )
                max_kkt = max(max_kkt, violation)
                assert violation < 1e-5
                np.testing.assert_allclose(
                    selected.reconstruction_residual,
                    np.linalg.norm(x @ b - y),
                    atol=1e-12,
                )
                checks += 1
    metrics = pd.read_csv(out / "composition_metrics.csv")
    for r in metrics.itertuples():
        rows = preds[
            (preds.part == r.part)
            & (preds.method == r.method)
            & (preds["mode"] == r.mode)
        ]
        assert np.isclose(
            r.mse, np.mean((rows.estimate - rows.target) ** 2), atol=1e-13
        )
        assert np.isclose(
            r.r, np.corrcoef(rows.estimate, rows.target)[0, 1], atol=1e-12
        )
    # Exact measured content-observation collision, not a fabricated constant reader.
    b = bases["content_query"]
    a = np.vstack([b.T, np.ones(6)])
    q = np.array([-1, 1, 0, 0, 0, 0], float)
    null = q - a.T @ np.linalg.solve(a @ a.T, a @ q)
    direction = null / np.max(np.abs(null)) / 12
    plus = np.full(6, 1 / 6) + direction
    minus = np.full(6, 1 / 6) - direction
    assert min(plus.min(), minus.min()) >= 0
    assert abs((plus - minus) @ b[:, 0]) < 1e-12
    collision = {
        "plus": plus.tolist(),
        "minus": minus.tolist(),
        "query_values": [float(plus @ b[:, 0]), float(minus @ b[:, 0])],
        "action_minus_intention_targets": [float(plus @ q), float(minus @ q)],
        "typed_vector_separation": float(
            np.linalg.norm((plus - minus) @ bases["typed_queries"])
        ),
    }
    write(out / "content_collision.json", collision)
    rank = json.loads((out / "identifiability.json").read_text())
    for method, b in bases.items():
        a = np.vstack([b.T, np.ones(6)])
        assert np.linalg.matrix_rank(a) == rank[method]["rank"]
    result = {
        "status": "PASS",
        "frozen_files": len(freeze["files"]),
        "probability_pairs": len(pairs),
        "raw_premise_checks": len(pairs),
        "natural_original_address_checks": len(private),
        "source_basis_checks": 3,
        "convex_optimality_checks": checks,
        "max_kkt_violation": max_kkt,
        "metric_checks": len(metrics),
        "content_collision": collision,
        "not_independent_execution_team": True,
    }
    write(out / "independent_numerical_audit.json", result)
    print(json.dumps(result), flush=True)


def main():
    """Run the bounded diagnostic or independent numerical audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sanity", "audit"])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    (observer_sanity if args.stage == "sanity" else audit)(args.output)


if __name__ == "__main__":
    main()
