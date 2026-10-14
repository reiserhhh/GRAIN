#!/usr/bin/env python3
"""Direct finite likelihood replay and process nonidentification certificates."""

import argparse
import csv
import gzip
import itertools
import json
from pathlib import Path

import numpy as np


def law(z, f, b):
    """Independent observed marginal of each declared mechanism."""
    sigmoid = lambda v: 1 / (1 + np.exp(-v))
    if f in ("M1", "M2", "M3"):
        return sigmoid(b * z[:, int(f[-1]) - 1])
    if f == "audience":
        return sigmoid(b * z[:, 1])
    if f == "belief":
        return sigmoid(b * z[:, 0]) * sigmoid(-1.2) + (
            1 - sigmoid(b * z[:, 0])
        ) * sigmoid(1.2)
    p = sigmoid(b) * sigmoid(-1.2) + (1 - sigmoid(b)) * sigmoid(1.2)
    return sigmoid(np.log(p / (1 - p)) * z[:, 0])


def main():
    """Infer equivalence classes from probability signatures, not saved class labels."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    even = np.repeat(corners[np.prod(corners, axis=1) == 1], 3, axis=0)
    odd = np.repeat(corners[np.prod(corners, axis=1) == -1], 3, axis=0)
    maximum = 0.0
    authors = 0
    targets = 0
    with gzip.open(args.input / "author_packets.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            authors += 1
            families = (
                ("M1", "M2", "M3")
                if row["panel"] == "main"
                else ("M1", "M2", "M3", "belief", "audience", "negative_M1")
            )
            atoms = [(f, b) for f in families for b in (0.5, 1, 1.5)]
            signatures = []
            classes = []
            for f, b in atoms:
                signature = law(corners, f, b)
                matches = [
                    i
                    for i, s in enumerate(signatures)
                    if np.max(np.abs(s - signature)) < 1e-12
                ]
                if matches:
                    classes.append(matches[0])
                else:
                    classes.append(len(signatures))
                    signatures.append(signature)
            classes = np.array(classes)
            prior = np.array(
                [1 / (len(signatures) * sum(classes == c)) for c in classes]
            )
            dev = np.array(row["development"])
            future = np.array(row["future"])
            for saved, z, y, test, observed in (
                (row["scores"][0], even, dev, odd, future),
                (row["scores"][1], odd[:6], future[:6], odd[6:], future[6:]),
            ):
                probability = np.array([law(z, f, b) for f, b in atoms])
                mass = prior * np.prod(
                    np.where(y, probability, 1 - probability), axis=1
                )
                post = mass / mass.sum()
                prediction = post @ np.array([law(test, f, b) for f, b in atoms])
                cp = np.bincount(classes, weights=post)
                for actual, stored in (
                    (post, saved["posterior"]),
                    (prediction, saved["prediction"]),
                    (cp, saved["class_posterior"]),
                ):
                    maximum = max(maximum, float(np.max(np.abs(actual - stored))))
                    np.testing.assert_allclose(actual, stored, atol=1e-12)
                np.testing.assert_allclose(
                    np.mean((prediction - observed) ** 2), saved["brier"], atol=1e-12
                )
                np.testing.assert_allclose(
                    -np.log(np.where(observed, prediction, 1 - prediction)).mean(),
                    saved["log_loss"],
                    atol=1e-12,
                )
                targets += 1
    certificates = []
    for beta in (0.5, 1, 1.5):
        for x in (-1.0, -1 / 3, 1 / 3, 1.0):
            z = np.array([[x, 1.0, -1.0]])
            first = float(law(z, "belief", beta)[0])
            second = float(law(z, "negative_M1", beta)[0])
            p_b = 1 / (1 + np.exp(-beta * x))
            joint = np.zeros((2, 2))
            for b, y in itertools.product((0, 1), repeat=2):
                response = 1 / (1 + np.exp(-1.2 * (1 - 2 * b)))
                joint[b, y] = (p_b if b else 1 - p_b) * (
                    response if y else 1 - response
                )
            independent = np.outer(joint.sum(axis=1), joint.sum(axis=0))
            certificates.append(
                {
                    "beta": beta,
                    "z1": x,
                    "belief_y": first,
                    "negative_logistic_y": second,
                    "marginal_difference": first - second,
                    "joint_B_Y_vs_independent_KL": float(
                        np.sum(joint * np.log(joint / independent))
                    ),
                    "joint_comparator": "same B and Y marginals but independent;extra B measurement adds a stated assumption",
                }
            )
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "process_certificates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(certificates[0]))
        writer.writeheader()
        writer.writerows(certificates)
    result = {
        "status": "DIRECT_POSTERIOR_AND_PREDICTION_REPLAY_PASS",
        "authors": authors,
        "targets": targets,
        "max_difference": maximum,
        "certificates": len(certificates),
        "scope": "process class discovery by signature;extraB comparison is declared independence alternative,not universal psychological identification",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
