#!/usr/bin/env python3
"""Source-estimated chart transitions and path consistency diagnostics.

These are finite maps between observed coordinate systems. The experiments
test prerequisites of an atlas; they do not assert that embeddings form a
smooth psychometric manifold or that a graph residual is a sheaf cohomology.
"""

import numpy as np

from .math_local import affine_fit, affine_score


def polynomial_coordinates(x: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate a fixed degree-one or degree-two chart, without hidden truth."""
    if degree == 1:
        return x
    if degree != 2:
        raise ValueError("only the frozen two finite degrees are supported")
    return np.column_stack(
        [x] + [x[:, i] * x[:, j] for i in range(x.shape[1]) for j in range(i + 1)]
    )


def fit_transition(x: np.ndarray, y: np.ndarray, degree: int) -> dict:
    """Fit a transition on paired source observations only, not evaluation states."""
    return {
        **affine_fit(polynomial_coordinates(x, degree), y, 1e-4),
        "degree": np.array(degree),
    }


def transition(model: dict, x: np.ndarray) -> np.ndarray:
    """Apply a source-frozen chart transition to observed application coordinates."""
    return affine_score(model, polynomial_coordinates(x, int(model["degree"])))


def atlas_statistics(
    source: list[np.ndarray], application: list[np.ndarray], degree: int
) -> tuple[list[dict], dict, dict]:
    """Compare direct, composite and cycle transports with independent pair fits."""
    models, predictions, rows = {}, {}, []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            key = f"{i}_to_{j}"
            model = fit_transition(source[i], source[j], degree)
            models[key] = model
            predictions[key] = transition(model, application[i])
            rows.append(
                {
                    "degree": degree,
                    "path": key,
                    "type": "direct",
                    "observed_rmse": float(
                        np.sqrt(np.mean((predictions[key] - application[j]) ** 2))
                    ),
                }
            )
    for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 1, 0)):
        first = transition(models[f"{i}_to_{j}"], application[i])
        composite = transition(models[f"{j}_to_{k}"], first)
        direct = predictions[f"{i}_to_{k}"]
        cycle = transition(models[f"{k}_to_{i}"], composite)
        name = f"{i}_to_{j}_to_{k}"
        predictions[name] = composite
        predictions[name + "_cycle"] = cycle
        rows.append(
            {
                "degree": degree,
                "path": name,
                "type": "composite",
                "observed_rmse": float(
                    np.sqrt(np.mean((composite - application[k]) ** 2))
                ),
                "composition_defect": float(
                    np.sqrt(np.mean((composite - direct) ** 2))
                ),
                "cycle_defect": float(np.sqrt(np.mean((cycle - application[i]) ** 2))),
            }
        )
    return rows, models, predictions


def simulated_chart_world(seed: int, world: str) -> tuple[list, list, dict]:
    """Generate paired source/application charts; keep latent states evaluator-only."""
    if world not in ("affine", "curved", "collision"):
        raise ValueError(world)
    rng = np.random.default_rng(seed)
    groups, truth = [], {}
    for role in ("source", "application"):
        theta = rng.uniform(-1, 1, (240, 2))
        first = theta.copy()
        second = theta @ np.array([[1.4, 0.3], [-0.2, 0.9]]).T + [0.2, -0.1]
        third = theta @ np.array([[0.8, -0.1], [0.4, 1.2]]).T + [-0.1, 0.3]
        if world == "curved":
            third[:, 1] += 0.9 * third[:, 0] ** 2
        elif world == "collision":
            # Exact sign collision in the noiseless chart; not merely sparse data.
            third = np.column_stack([theta[:, 0] ** 2, theta[:, 1]])
        noise = [rng.normal(0, 0.005, first.shape) for _ in range(3)]
        groups.append(
            [
                value + error
                for value, error in zip((first, second, third), noise, strict=True)
            ]
        )
        truth[role] = theta
    return groups[0], groups[1], truth


def collision_witness() -> dict:
    """Produce identical observed chart states with distinct hidden signed states."""
    u, v = np.array([0.6, 0.2]), np.array([-0.6, 0.2])
    chart_u, chart_v = [u[0] ** 2, u[1]], [v[0] ** 2, v[1]]
    return {
        "chart_u": chart_u,
        "chart_v": chart_v,
        "observation_difference": float(np.linalg.norm(np.array(chart_u) - chart_v)),
        "signed_target_difference": float(u[0] - v[0]),
        "scope": "this chart cannot identify the signed target, even with infinite repeats",
    }
