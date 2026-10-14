"""Finite predictive-state worlds for F02."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np


def c1_probability(kind: int, z: int) -> float:
    """Return the exact C1 binary response probability."""
    return 0.5 + 0.25 * kind * z


def c2_next_probability(a: int, b: int, next_value: int, world: str = "xor") -> float:
    """Return a second-order transition probability for XOR or iid worlds."""
    if world == "iid":
        return 0.5
    if world != "xor":
        raise ValueError("world must be xor or iid")
    expected = a ^ b
    return 0.75 if next_value == expected else 0.25


def future_distribution(
    history: tuple[int, int], horizon: int, world: str = "xor"
) -> dict[str, float]:
    """Enumerate the exact finite future law conditional on the last two outcomes."""
    distribution = {"": 1.0}
    for _ in range(horizon):
        next_distribution: dict[str, float] = defaultdict(float)
        for suffix, probability in distribution.items():
            prefix = history + tuple(int(value) for value in suffix)
            a, b = prefix[-2:]
            for value in (0, 1):
                next_distribution[suffix + str(value)] += (
                    probability * c2_next_probability(a, b, value, world)
                )
        distribution = dict(next_distribution)
    return distribution


def c3_identified_sets() -> list[dict]:
    """Return exact exposure/product ambiguity certificates."""
    return [
        {
            "pi": 0.8,
            "q": 0.5,
            "published_rate": 0.4,
            "q_set_given_published": [0.4, 1.0],
        },
        {
            "pi": 0.4,
            "q": 1.0,
            "published_rate": 0.4,
            "q_set_given_published": [0.4, 1.0],
        },
    ]


def opaque_author(seed: int, world: str, role: str, index: int) -> str:
    """Create an opaque author key that does not encode role or latent type."""
    raw = f"{seed}|{world}|{role}|{index}".encode()
    return "u_" + hashlib.sha256(raw).hexdigest()[:16]


def generate_world(
    world: str, seed: int, *, events: int = 48, counts: dict[str, int] | None = None
) -> tuple[list[dict], list[dict]]:
    """Generate records and a separate truth table for C1/C2/C3."""
    counts = counts or {"source": 300, "validation": 100, "application": 200}
    world_number = {"c1": 1, "c2_xor": 2, "c2_iid": 3, "c3": 4}.get(world)
    if world_number is None:
        raise ValueError(f"unknown world {world}")
    records, truth = [], []
    index = 0
    for role_number, role in enumerate(("source", "validation", "application")):
        streams = np.random.SeedSequence([seed, 2, world_number, role_number]).spawn(4)
        author_rng, material_rng, noise_rng, render_rng = [
            np.random.default_rng(s) for s in streams
        ]
        kinds = np.resize(np.array([-1, 1]), counts[role])
        author_rng.shuffle(kinds)
        for author_number in range(counts[role]):
            author_id = opaque_author(seed, world, role, author_number)
            kind = int(kinds[author_number])
            values = [int(material_rng.integers(2)), int(material_rng.integers(2))]
            pi, q = (0.8, 0.5) if kind == -1 else (0.4, 1.0)
            target = {
                "author_id": author_id,
                "role": role,
                "world": world,
                "kind": kind,
            }
            if world == "c3":
                target.update({"pi": pi, "q": q, "published_rate": pi * q})
            truth.append(target)
            for t in range(events):
                z = -1 if t % 2 == 0 else 1
                opportunity = 1
                selected = 1
                if world == "c1":
                    observation = int(noise_rng.random() < c1_probability(kind, z))
                    opportunity_status = "assigned"
                elif world in ("c2_xor", "c2_iid"):
                    transition_world = "xor" if world == "c2_xor" else "iid"
                    p = c2_next_probability(values[-2], values[-1], 1, transition_world)
                    observation = int(noise_rng.random() < p)
                    values.append(observation)
                    opportunity_status = "assigned"
                elif world == "c3":
                    opportunity = int(material_rng.random() < pi)
                    selected = opportunity * int(noise_rng.random() < q)
                    # Content is ancillary here; the target is opportunity and
                    # selection. Both types have exactly the same publication law.
                    observation = int(render_rng.random() < 0.5) if selected else None
                    opportunity_status = "observed"
                else:
                    raise ValueError(f"unknown world {world}")
                records.append(
                    {
                        "schema_version": "grain-gptpro-record-v3",
                        "author_id": author_id,
                        "event_id": f"{world}_{seed}_{index:08d}",
                        "episode_id": f"{author_id}_episode",
                        "t": t,
                        "condition": {"z": z},
                        "opportunity": opportunity,
                        "opportunity_status": opportunity_status,
                        "selected": selected,
                        "observation": observation,
                        "text_ref": f"synthetic://{world}/{index}",
                        "reader_id": "synthetic_identity_reader",
                        "role": role,
                    }
                )
                index += 1
    return records, truth
