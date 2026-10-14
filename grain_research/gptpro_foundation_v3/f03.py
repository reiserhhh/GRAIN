"""Finite opportunity, selection, and response worlds for F03."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

WORLD_VALUES = {
    "W3-A": (0.0, 0.0),
    "W3-B": (0.8, 0.0),
    "W3-C": (0.0, 1.0),
    "W3-D": (0.8, 1.0),
}


def sigmoid(value: float) -> float:
    """Return a numerically safe logistic probability."""
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))


def response_probability(r: float, x: int) -> float:
    """Return P(Y=1 | z,x) for the finite response world."""
    return float(np.clip(r + 0.15 * (2 * int(x) - 1), 0.05, 0.95))


def selection_probability(
    kappa: float, lam: float, gamma: float, z: int, x: int, y: int
) -> float:
    """Return P(S=1 | O=1,z,x,Y=y)."""
    return sigmoid(kappa + lam * z + 0.8 * (2 * int(x) - 1) + gamma * (2 * int(y) - 1))


def cell_probability(
    r: float, pi: float, kappa: float, lam: float, gamma: float, z: int, x: int, y: int
) -> dict[str, float]:
    """Return the exact joint probability for one z/x/Y cell."""
    p_y = response_probability(r, x) if y else 1.0 - response_probability(r, x)
    p_s = selection_probability(kappa, lam, gamma, z, x, y)
    p_joint = 0.5 * p_y
    return {
        "p_x": 0.5,
        "p_response": response_probability(r, x),
        "p_selection_given_opportunity": p_s,
        "p_opportunity_and_y": pi * p_joint,
        "p_published_y": pi * p_joint * p_s,
        "p_published": pi * p_s,
    }


def population_cells(params: dict) -> list[dict]:
    """Enumerate exact cells for both conditions and response values."""
    rows: list[dict] = []
    for z in (-1, 1):
        key = "-" if z == -1 else "+"
        for x in (0, 1):
            for y in (0, 1):
                rows.append(
                    {
                        "z": z,
                        "x": x,
                        "y": y,
                        **cell_probability(
                            params["r"][key],
                            params["pi"][key],
                            params["kappa"],
                            params["lambda"],
                            params["gamma"],
                            z,
                            x,
                            y,
                        ),
                    }
                )
    return rows


def exact_cells(params: dict) -> list[dict]:
    """Backward-compatible alias for exact population cell enumeration."""
    return population_cells(params)


def response_bound(g: float, f: float, pi: float | None = None) -> tuple[float, float]:
    """Return the partial-identification interval for E[Y|z]."""
    if pi is None:
        return (float(g), float(g + 1.0 - f))
    return (float(g / pi), float(g / pi + 1.0 - f / pi))


def opaque_author(seed: int, world: str, design: str, role: str, index: int) -> str:
    """Identify the same simulated author across the three paired designs."""
    return (
        "u_"
        + hashlib.sha256(f"{seed}|{world}|{role}|{index}".encode()).hexdigest()[:16]
    )


def _author_truth(rng: np.random.Generator, world: str) -> dict:
    """Sample author-level parameters; keep them out of records."""
    lam, gamma = WORLD_VALUES[world]
    return {
        "r": {
            "-": float(rng.choice([0.2, 0.5, 0.8])),
            "+": float(rng.choice([0.2, 0.5, 0.8])),
        },
        "pi": {"-": float(rng.choice([0.4, 0.8])), "+": float(rng.choice([0.4, 0.8]))},
        "kappa": float(rng.choice([-0.5, 0.5])),
        "lambda": lam,
        "gamma": gamma,
    }


def generate_records(
    world: str,
    design: str,
    seed: int,
    *,
    events: int = 32,
    counts: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Generate one design with a nested event stream per author.

    ``fixed`` sets O=S=1, ``common`` sets O=1 but leaves selection active,
    and ``free`` samples both opportunity and selection.  The same latent Y
    draw is used for selection and observation, which preserves the MNAR arm.
    Truth is returned separately and never inserted in records.
    """
    if world not in WORLD_VALUES:
        raise ValueError(f"unknown world {world}")
    if design not in {"free", "fixed", "common"}:
        raise ValueError(f"unknown design {design}")
    counts = counts or {"source": 300, "validation": 100, "application": 200}
    records: list[dict] = []
    truth: list[dict] = []
    index = 0
    world_number = list(WORLD_VALUES).index(world)
    for role_number, role in enumerate(("source", "validation", "application")):
        streams = np.random.SeedSequence([seed, 3, world_number, role_number]).spawn(4)
        author_seeds = [stream.spawn(counts[role]) for stream in streams]
        for number in range(counts[role]):
            author = opaque_author(seed, world, design, role, number)
            params = _author_truth(
                np.random.default_rng(author_seeds[0][number]), world
            )
            material = author_seeds[1][number].spawn(2)
            noise = author_seeds[2][number].spawn(3)
            z_values = np.random.default_rng(material[0]).choice([-1, 1], size=events)
            x_values = np.random.default_rng(material[1]).integers(2, size=events)
            y_noise, opportunity_noise, selection_noise = [
                np.random.default_rng(stream).random(events) for stream in noise
            ]
            truth.append(
                {
                    "author_id": author,
                    "world": world,
                    "design": design,
                    "role": role,
                    **params,
                }
            )
            for t in range(events):
                z = int(z_values[t])
                x = int(x_values[t])
                key = "-" if z == -1 else "+"
                latent_y = int(y_noise[t] < response_probability(params["r"][key], x))
                if design == "fixed":
                    opportunity, selected, status = 1, 1, "assigned"
                else:
                    opportunity = (
                        1
                        if design == "common"
                        else int(opportunity_noise[t] < params["pi"][key])
                    )
                    selected = 0
                    if opportunity:
                        selected = int(
                            selection_noise[t]
                            < selection_probability(
                                params["kappa"],
                                params["lambda"],
                                params["gamma"],
                                z,
                                x,
                                latent_y,
                            )
                        )
                    status = "observed"
                records.append(
                    {
                        "schema_version": "grain-gptpro-record-v3",
                        "author_id": author,
                        "event_id": f"{world}_{design}_{seed}_{index:08d}",
                        "episode_id": f"{author}_episode",
                        "t": t,
                        "condition": {"z": z, "x": x},
                        "opportunity": opportunity,
                        "opportunity_status": status,
                        "selected": selected,
                        "observation": latent_y if selected else None,
                        "text_ref": f"synthetic://{world}/{design}/{index}",
                        "reader_id": "synthetic_identity_reader",
                        "role": role,
                    }
                )
                index += 1
    return records, truth


def grouped(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by opaque author and preserve time order."""
    output: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        output[row["author_id"]].append(row)
    return {key: sorted(rows, key=lambda row: row["t"]) for key, rows in output.items()}


def naive_selected_mean(rows: list[dict], z: int) -> float | None:
    """Estimate E[Y | selected,z] from observed selected responses."""
    values = [
        row["observation"]
        for row in rows
        if row["condition"]["z"] == z
        and row["selected"] == 1
        and row["observation"] is not None
    ]
    return float(np.mean(values)) if values else None


def beta_condition_mean(
    rows: list[dict], z: int, alpha: float = 1.0, beta: float = 1.0
) -> float | None:
    """Return a bounded Beta posterior mean for the observed response view."""
    values = [
        row["observation"]
        for row in rows
        if row["condition"]["z"] == z
        and row["selected"] == 1
        and row["observation"] is not None
    ]
    return (
        float((alpha + sum(values)) / (alpha + beta + len(values))) if values else None
    )


def known_opportunity_bounds(
    rows: list[dict], z: int
) -> tuple[float | None, float | None, float, float]:
    """Return bounds and publication/opportunity rates when O is known."""
    subset = [row for row in rows if row["condition"]["z"] == z]
    if not subset:
        return None, None, 0.0, 0.0
    published = [row for row in subset if row["selected"] == 1]
    f = float(len(published) / len(subset))
    g = float(
        sum(row["observation"] for row in published if row["observation"] is not None)
        / len(subset)
    )
    pi = float(np.mean([row["opportunity"] for row in subset]))
    if pi <= 0:
        return 0.0, 1.0, f, pi
    lo, hi = response_bound(g, f, pi)
    return max(0.0, lo), min(1.0, hi), f, pi


def ipw_condition_mean(rows: list[dict], z: int) -> tuple[float | None, int]:
    """Fit observed-covariate propensity and return a finite IPW estimate.

    In MNAR worlds Y also drives selection, so this intentionally remains
    biased under the wrong assumption; it is not an oracle repair.
    """
    opportunity_rows = [row for row in rows if row["opportunity"] == 1]
    if not opportunity_rows:
        return None, 0
    xmat = np.array(
        [[row["condition"]["z"], row["condition"]["x"]] for row in opportunity_rows],
        dtype=float,
    )
    labels = np.array([row["selected"] for row in opportunity_rows], dtype=float)
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
        model.fit(xmat, labels)
        q = np.clip(model.predict_proba(xmat)[:, 1], 0.01, 0.99)
        selected_rows = [
            row
            for row in opportunity_rows
            if row["selected"] == 1
            and row["observation"] is not None
            and row["condition"]["z"] == z
        ]
        if not selected_rows:
            return None, int(np.sum((q <= 0.01) | (q >= 0.99)))
        features = np.array(
            [[row["condition"]["z"], row["condition"]["x"]] for row in selected_rows],
            dtype=float,
        )
        q_selected = np.clip(model.predict_proba(features)[:, 1], 0.01, 0.99)
    except Exception:
        q = np.full(len(labels), float(np.mean(labels)))
        selected_rows = [
            row
            for row in opportunity_rows
            if row["selected"] == 1
            and row["observation"] is not None
            and row["condition"]["z"] == z
        ]
        if not selected_rows:
            return None, 0
        q_selected = np.full(len(selected_rows), float(np.mean(labels)))
    weights = 1.0 / q_selected
    estimate = float(
        np.sum(
            weights
            * np.array([row["observation"] for row in selected_rows], dtype=float)
        )
        / np.sum(weights)
    )
    return estimate, int(np.sum((q < 0.01) | (q > 0.99)))
