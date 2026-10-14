#!/usr/bin/env python3
"""Source-identified relational indices and a separate numerical evidence ledger."""

from __future__ import annotations

import itertools
import re

import numpy as np


def material_bank():
    """Return fixed source/cold worlds with evaluation roles kept separate."""
    permutations = list(itertools.permutations(range(3)))
    rows, truths = [], []

    def append(role, names, fields, events, order, meta):
        header = "Participants: " + ", ".join(names) + ".\n"
        header += "Fields: " + "; ".join(fields) + ".\nRecords:\n"
        body = header + "\n".join(events[i][0] for i in order)
        rows.append({"id": len(rows), "role": role, "body": body})
        truths.append(
            {"id": len(rows) - 1, "roles": [events[i][1] for i in order], **meta}
        )

    for role, names0, countries, jobs, layouts in (
        (
            "source",
            ("John", "Mary", "Alex"),
            ("Italy", "Japan", "Canada"),
            ("nurse", "baker", "pilot"),
            range(2),
        ),
        (
            "cold_fact",
            ("David", "Emma", "Ryan"),
            ("France", "Turkey", "Spain"),
            ("teacher", "driver", "artist"),
            range(1),
        ),
    ):
        for pi, permutation in enumerate(permutations):
            names = [names0[i] for i in permutation]
            for reverse, layout in itertools.product(range(2), layouts):
                rng = np.random.default_rng(129000 + len(rows))
                c, j = rng.permutation(countries), rng.permutation(jobs)
                fields = ["residence", "occupation"][:: -1 if reverse else 1]
                events = []
                for person, name in enumerate(names):
                    for relation, field in enumerate(fields):
                        action = (
                            f"lives in {c[person]}"
                            if field == "residence"
                            else f"works as a {j[person]}"
                        )
                        events.append((f"{name} {action}.", [person, relation]))
                base = (
                    np.arange(6)[:: -1 if layout else 1]
                    if role == "source"
                    else np.array([2, 5, 0, 4, 1, 3])
                )
                order = np.roll(base, pi).tolist()
                append(
                    role,
                    names,
                    fields,
                    events,
                    order,
                    {"permutation": pi, "header_reverse": reverse, "layout": layout},
                )

    behaviours = (
        (
            (
                "welcomed strangers and kept their conversation going",
                "sat apart and avoided starting a conversation",
            ),
            (
                "introduced everyone and asked several friendly questions",
                "chose a quiet seat and spoke only when addressed",
            ),
        ),
        (
            (
                "remained worried and expected things to go wrong",
                "stayed calm and did not expect a disaster",
            ),
            (
                "kept fretting about mistakes long after the problem ended",
                "felt relaxed after checking what actually happened",
            ),
        ),
        (
            (
                "offered help and considered what other people needed",
                "dismissed requests and ignored what other people needed",
            ),
            (
                "listened kindly and made room for another person's needs",
                "refused to listen and insisted only personal needs mattered",
            ),
        ),
        (
            (
                "planned each step and finished the agreed work",
                "left tasks unfinished and ignored the agreed schedule",
            ),
            (
                "checked the schedule and completed each promised task",
                "forgot the plan and abandoned the promised tasks",
            ),
        ),
        (
            (
                "tried unfamiliar ideas and explored an unusual design",
                "rejected unfamiliar ideas and repeated the old design",
            ),
            (
                "experimented with a new approach out of curiosity",
                "insisted on the familiar approach and avoided experimenting",
            ),
        ),
        (
            (
                "said the chatbot felt lonely and wanted companionship",
                "said the chatbot had no feelings or personal wishes",
            ),
            (
                "talked about the software as a friend with its own emotions",
                "described the software as a tool without human emotions",
            ),
        ),
    )
    for family, wording, reverse, assignment in itertools.product(
        range(6), range(2), range(2), range(2)
    ):
        names = [("David", "Emma", "Ryan")[i] for i in permutations[family]]
        fields = ["community dinner", "project review"][:: -1 if reverse else 1]
        events = []
        for person, name in enumerate(names):
            for relation, field in enumerate(fields):
                condition = ("At the " if wording == 0 else "During the ") + field
                action = (
                    "read the printed agenda and then took a seat"
                    if person == 2
                    else behaviours[family][wording][person ^ assignment ^ relation]
                )
                events.append((f"{condition}, {name} {action}.", [person, relation]))
        order = np.roll(np.array([4, 1, 2, 5, 0, 3]), family + reverse).tolist()
        append(
            "cold_behaviour",
            names,
            fields,
            events,
            order,
            {
                "family": family,
                "wording": wording,
                "header_reverse": reverse,
                "assignment": assignment,
            },
        )
    return rows, truths


def record_intervals(body):
    """Extract declared record endpoints without using owner or condition labels."""
    marker = "Records:\n"
    if body.count(marker) != 1:
        raise ValueError("One declared record section required")
    begin = body.index(marker) + len(marker)
    result = []
    for match in re.finditer(r"[^\n]+", body[begin:]):
        record = match.group()
        endpoint = len(record.rstrip(". "))
        if not endpoint:
            raise ValueError("Empty record has no observation")
        result.append([begin + match.start(), begin + match.start() + endpoint])
    if len(result) != 6:
        raise ValueError("Fixed finite protocol requires six complete records")
    return result


def fit_program(states, labels, components=2):
    """Fit fixed-component source PLS and export an exact NumPy affine reader."""
    from sklearn.cross_decomposition import PLSRegression

    x, y = np.asarray(states, float), np.asarray(labels, float)
    if (
        x.ndim != 2
        or y.shape != (len(x), 2)
        or not all(np.isfinite(a).all() for a in (x, y))
    ):
        raise ValueError("Finite source states and two structural indices required")
    center, scale = x.mean(0), x.std(0, ddof=1)
    scale[scale == 0] = 1
    scaled = (x - center) / scale
    model = PLSRegression(n_components=components, scale=False, max_iter=500, tol=1e-6)
    model.fit(scaled, y)
    coefficients = model.coef_.T
    if coefficients.shape != (x.shape[1], 2):
        raise ValueError("Unexpected installed PLS coefficient convention")
    program = {
        "center": center,
        "scale": scale,
        "coefficients": coefficients,
        "intercept": model.intercept_ - model._x_mean @ coefficients,
        "components": np.array(components),
        "iterations": np.array(model.n_iter_),
    }
    predicted = predict(program, x)
    if not np.allclose(predicted, model.predict(scaled), rtol=1e-10, atol=1e-10):
        raise RuntimeError("Export does not reproduce source PLS")
    return program


def predict(program, states):
    """Apply a frozen affine structural reader; no labels or text are accepted."""
    x = np.asarray(states, float)
    if x.ndim != 2 or x.shape[1] != len(program["center"]) or not np.isfinite(x).all():
        raise ValueError("Compatible finite cold numeric states required")
    return ((x - program["center"]) / program["scale"]) @ program[
        "coefficients"
    ] + program["intercept"]


def assign(predictions):
    """Map continuous structural coordinates to the fixed legal index grid."""
    p = np.asarray(predictions, float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("Two finite structural coordinates required")
    return np.stack(
        (
            np.clip(np.floor(p[:, 0] + 0.5), 0, 2),
            np.clip(np.floor(p[:, 1] + 0.5), 0, 1),
        ),
        axis=1,
    ).astype(int)


def target_values(values, directions):
    """Use the existing uniform value observation, keeping all targets/heads."""
    v, d = np.asarray(values, float), np.asarray(directions, float)
    if (
        v.ndim != 3
        or d.ndim != 3
        or d.shape[1] % v.shape[1]
        or v.shape[2] != d.shape[2]
    ):
        raise ValueError("Compatible V carriers and frozen target covectors required")
    if not np.isfinite(v).all() or not np.isfinite(d).all():
        raise ValueError("Finite value observations required")
    kh = np.arange(d.shape[1]) // (d.shape[1] // v.shape[1])
    return np.einsum("ehd,qhd->eq", v[:, kh], d)


def cell_ledger(roles, readings):
    """Keep conditional means, counts and explicit empty cells, not fake zeros."""
    role, y = np.asarray(roles), np.asarray(readings, float)
    if role.shape != (len(y), 2) or y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("Aligned assigned cells and finite target readings required")
    if (
        not np.issubdtype(role.dtype, np.integer)
        or np.any(role < 0)
        or np.any(role >= [3, 2])
    ):
        raise ValueError("Legal integer structural roles required")
    count = np.zeros((3, 2), int)
    totals = np.zeros((3, 2, y.shape[1]), float)
    for cell, score in zip(role, y, strict=True):
        count[tuple(cell)] += 1
        totals[tuple(cell)] += score
    mean = np.full_like(totals, np.nan)
    np.divide(totals, count[..., None], out=mean, where=count[..., None] > 0)
    return mean, count
