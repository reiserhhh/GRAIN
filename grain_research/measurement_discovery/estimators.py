"""Small finite-history estimators used by the GP02 competition."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

STATES = tuple((a, b) for a in (0, 1) for b in (0, 1))


def ordered_rows(records):
    """Group observation records by author and order each history by t."""
    grouped = {}
    for row in records:
        grouped.setdefault(row["author_id"], []).append(row)
    return {
        author: sorted(rows, key=lambda row: (row["t"], row["event_id"]))
        for author, rows in grouped.items()
    }


def transitions(rows):
    """Return design rows for all observations with two preceding outcomes."""
    if len(rows) < 3:
        return np.empty((0, 4)), np.empty(0, dtype=float)
    x, y = [], []
    values = [int(row["observation"]) for row in rows]
    for index in range(2, len(values)):
        a, b = values[index - 2 : index]
        x.append([1.0, b, a, a * b])
        y.append(values[index])
    return np.asarray(x), np.asarray(y, dtype=float)


def fit_b1(source_rows):
    """Fit a source-only four-state conditional frequency table."""
    counts = np.zeros((4, 2), dtype=float)
    for rows in source_rows.values():
        values = [int(row["observation"]) for row in rows]
        for index in range(2, len(values)):
            state = 2 * values[index - 2] + values[index - 1]
            counts[state, values[index]] += 1
    source_p = (counts[:, 1] + 1) / (counts.sum(axis=1) + 2)
    return {"counts": counts, "source_p": source_p}


def fit_b2(source_rows, alpha=1.0):
    """Fit the pooled second-order interaction regression on source rows only."""
    matrices = [transitions(rows) for rows in source_rows.values()]
    x = np.concatenate([part[0] for part in matrices if len(part[0])], axis=0)
    y = np.concatenate([part[1] for part in matrices if len(part[1])], axis=0)
    model = Ridge(alpha=alpha, fit_intercept=False).fit(x, y)
    return {"coef": model.coef_, "alpha": float(alpha)}


def fit_m(source_rows, rank=2):
    """Fit the source-only rank-rank compression of author transition tables."""
    tables = []
    for rows in source_rows.values():
        counts = np.zeros((4, 2), dtype=float)
        values = [int(row["observation"]) for row in rows]
        for index in range(2, len(values)):
            state = 2 * values[index - 2] + values[index - 1]
            counts[state, values[index]] += 1
        tables.append((counts[:, 1] + 1) / (counts.sum(axis=1) + 2))
    table = np.asarray(tables)
    mean = table.mean(axis=0)
    _, singular, right = np.linalg.svd(table - mean, full_matrices=False)
    return {
        "mean": mean,
        "right": right[:rank],
        "singular_values": singular,
        "rank": rank,
    }


def author_table(rows):
    """Estimate one author's four-state probabilities from an observed prefix."""
    counts = np.zeros((4, 2), dtype=float)
    values = [int(row["observation"]) for row in rows]
    for index in range(2, len(values)):
        state = 2 * values[index - 2] + values[index - 1]
        counts[state, values[index]] += 1
    return (counts[:, 1] + 1) / (counts.sum(axis=1) + 2)


def predict_one(candidate, state, prefix, fitted, alpha=1.0):
    """Predict one next binary observation using only the supplied prefix."""
    a, b = state
    if candidate == "b1":
        p = fitted["b1"]["source_p"][2 * a + b]
    elif candidate == "b2":
        x, _ = transitions(prefix)
        if len(x):
            y = np.asarray([row["observation"] for row in prefix[2:]], float)
            source = np.asarray(fitted["b2"]["coef"])
            coef = np.linalg.solve(
                x.T @ x + alpha * np.eye(x.shape[1]),
                x.T @ y + alpha * source,
            )
        else:
            coef = fitted["b2"]["coef"]
        p = np.dot([1.0, b, a, a * b], coef)
    elif candidate == "m":
        raw = author_table(prefix)
        model = fitted["m"]
        p = (
            model["mean"]
            + (raw - model["mean"])
            @ np.asarray(model["right"]).T
            @ np.asarray(model["right"])
        )[2 * a + b]
    else:
        raise ValueError("unknown candidate")
    return float(np.clip(p, 1e-6, 1 - 1e-6))


def evaluate_author(rows, fitted, prefix_len, alpha):
    """Run a strict next-observation protocol and return candidate predictions."""
    if len(rows) <= prefix_len:
        return []
    prefix = list(rows[:prefix_len])
    values = [int(row["observation"]) for row in prefix[-2:]]
    output = []
    for row in rows[prefix_len:]:
        state = tuple(values[-2:])
        for candidate in ("b1", "b2", "m"):
            probability = predict_one(candidate, state, prefix, fitted, alpha)
            output.append({
                "author_id": row["author_id"],
                "event_id": row["event_id"],
                "t": row["t"],
                "candidate": candidate,
                "y": int(row["observation"]),
                "p": probability,
            })
        prefix.append(row)
        values.append(int(row["observation"]))
    return output


def metric_rows(predictions):
    """Compute Brier and log loss globally and by author."""
    rows = []
    for candidate in ("b1", "b2", "m"):
        subset = [row for row in predictions if row["candidate"] == candidate]
        if not subset:
            continue
        authors = sorted({row["author_id"] for row in subset})
        brier = np.mean([(row["p"] - row["y"]) ** 2 for row in subset])
        log_loss = np.mean([
            -(row["y"] * np.log(row["p"]) + (1 - row["y"]) * np.log1p(-row["p"]))
            for row in subset
        ])
        rows.append({"candidate": candidate, "brier": float(brier), "log_loss": float(log_loss), "authors": len(authors), "events": len(subset)})
    return rows
