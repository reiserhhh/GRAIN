#!/usr/bin/env python3
"""Lossless history observation, declared contrasts and strict qualification binding."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .context import QUESTIONS
from .controls import material_hash
from .external_material import ARMS
from .ready_comparison import digest, verified_packet
from .source_history import capture_aligned

SITES = ("input_embedding", "block_04", "block_10", "block_16", "final_norm")
HISTORIES = ("own", "matched", "invisible")
KINDS = (
    "current_own",
    "current_context",
    "current_visibility",
    "decision_own_behavior",
    "decision_own_disposition",
    "decision_context_behavior",
    "decision_context_disposition",
    "decision_task_interaction",
)


def qualification_packet(folder, material, audit, root):
    """Reject a token preflight or any changed instrument, not just a green status."""
    m, s = verified_packet(folder)
    if (
        m["protocol"] != "VM05_HISTORY_INSTRUMENT_V1"
        or m["preflight_only"]
        or not s["real_neural_qualification_passed"]
        or not s["all_instrument_errors_zero"]
        or s["forwards"] != 648
        or s["token_layouts_replayed"] != 324
        or m["material_identity"] != material["identity"]
        or m["audit_identity"] != audit["identity"]
        or m["labels_read"]
        or tuple(m["sites"]) != SITES
        or m["pad_to_length"] != 3072
    ):
        raise ValueError("Require complete648-forward real history qualification")
    for name, sha in m["code_sha256"].items():
        if digest(Path(root) / name) != sha:
            raise ValueError("Qualified instrument implementation changed")
    model = json.loads(
        (Path(folder) / "actual_observer.json").read_text(encoding="utf-8")
    )
    if (
        model["dtype"] != "torch.bfloat16"
        or not model["device"].startswith("cuda")
        or model["revision"] != material["qwen_revision"]
        or tuple(model["sites"]) != SITES
        or model["training"]
        or model["pad_to_length"] != 3072
        or model["attention"] != "eager"
    ):
        raise ValueError("Qualification model is not the pinned CUDA instrument")
    return m, s, model


def observation_signature(model):
    """Bind scored observer identity but not wall-clock metadata."""
    return material_hash({k: v for k, v in model.items() if k != "unix_time"})


def collect_history_author(layouts, observer, *, deadline=None, repeat=False):
    """Capture all54 logical layouts, caching only identical full model inputs."""
    expected = {
        (v, a, t, h)
        for v in range(3)
        for a in ARMS
        for t in QUESTIONS
        for h in HISTORIES
    }
    if set(layouts) != expected or tuple(observer.sites) != SITES:
        raise ValueError("Incomplete layout or observation-site protocol")
    candidates = [
        observer.tokenizer.encode(t, add_special_tokens=False) for t in ("Yes", "No")
    ]
    if any(len(c) != 1 for c in candidates) or candidates[0] == candidates[1]:
        raise ValueError("Frozen diagnostic response candidates changed")
    yes, no = (c[0] for c in candidates)
    before = observer.calls
    cache, arrays, rows = {}, {}, []
    prefix_error = embedding_error = replay_error = 0.0
    for vi in range(3):
        for arm in ARMS:
            earlier = None
            for task in QUESTIONS:
                states_by_history = {}
                for history in HISTORIES:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Preserve partial shard; do not promote incomplete author"
                        )
                    row = layouts[vi, arm, task, history]
                    key = material_hash(row)
                    if key not in cache:
                        states, logits = capture_aligned(observer, row)
                        if set(states) != set(SITES) or any(
                            v.shape[0] != 3 for v in states.values()
                        ):
                            raise ValueError("Wrong current mean/std/decision schema")
                        if repeat and not cache:
                            again, other = capture_aligned(observer, row)
                            replay_error = max(
                                float(np.max(abs(logits - other))),
                                max(
                                    float(np.max(abs(states[k] - again[k])))
                                    for k in SITES
                                ),
                            )
                            if replay_error != 0:
                                raise ValueError(
                                    "First author native history replay failed"
                                )
                        scores = np.asarray([logits[yes], logits[no]], dtype=np.float32)
                        cache[key] = states, scores
                    states, scores = cache[key]
                    states_by_history[history] = states
                    for site in SITES:
                        name = f"{arm}__{task}__{history}__{site}"
                        arrays.setdefault(
                            name, np.empty((3, *states[site].shape), dtype=np.float32)
                        )[vi] = states[site]
                    rows.append(
                        {
                            "view": vi,
                            "text_arm": arm,
                            "task": task,
                            "history_arm": history,
                            "layout_sha256": key,
                            "current_tokens": len(row["current_positions"]),
                            "current_position_sha256": material_hash(
                                row["current_positions"]
                            ),
                            "current_token_sha256": row["current_token_sha256"],
                            "prompt_tokens": len(row["ids"]),
                            "visible_tokens": row["visible_tokens"],
                            "yes_logit": float(scores[0]),
                            "no_logit": float(scores[1]),
                            "margin": float(
                                np.float64(scores[0]) - np.float64(scores[1])
                            ),
                        }
                    )
                embedding_error = max(
                    embedding_error,
                    max(
                        float(
                            np.max(
                                abs(
                                    states_by_history[h]["input_embedding"]
                                    - states_by_history["own"]["input_embedding"]
                                )
                            )
                        )
                        for h in HISTORIES
                    ),
                )
                if earlier is not None:
                    prefix_error = max(
                        prefix_error,
                        max(
                            float(
                                np.max(
                                    abs(states_by_history[h][s][1:] - earlier[h][s][1:])
                                )
                            )
                            for h in HISTORIES
                            for s in SITES
                        ),
                    )
                earlier = states_by_history
    if prefix_error != 0 or embedding_error != 0:
        raise ValueError("Current causal prefix or fixed-token embedding changed")
    if len(arrays) != 90 or any(not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("Incomplete/nonfinite author features")
    return arrays, {
        "logical_layouts": 54,
        "unique_layouts": len(cache),
        "forwards": observer.calls - before,
        "replay_max": replay_error,
        "causal_prefix_max": prefix_error,
        "input_embedding_max": embedding_error,
        "readouts": rows,
    }


def history_channel(features, arm, site, kind):
    """Equal-view mean/std of current states or declared differences, never trait scores.

    Dispersion of three view-level differences is not the standard deviation of
    token-level differences, whose cross-history covariance was not captured.
    """
    if arm not in ARMS or site not in SITES or kind not in KINDS:
        raise ValueError("Unknown frozen history readout")
    get = lambda t, h: np.asarray(features[f"{arm}__{t}__{h}__{site}"], dtype=float)
    for task in QUESTIONS:
        for h in HISTORIES:
            x = get(task, h)
            if x.ndim != 3 or x.shape[:2] != (3, 3) or not np.isfinite(x).all():
                raise ValueError("Wrong history feature shape")
    own = get("behavior", "own")
    other = get("behavior", "matched")
    if kind == "current_own":
        x = own[:, 1]
    elif kind == "current_context":
        x = own[:, 1] - other[:, 1]
    elif kind == "current_visibility":
        x = own[:, 1] - get("behavior", "invisible")[:, 1]
    elif kind == "decision_task_interaction":
        x = (
            get("disposition", "own")[:, 0]
            - get("disposition", "matched")[:, 0]
            - (own[:, 0] - other[:, 0])
        )
    else:
        task = kind.rsplit("_", 1)[-1]
        x = get(task, "own")[:, 0]
        if kind.startswith("decision_context"):
            x = x - get(task, "matched")[:, 0]
    return np.concatenate([x.mean(0), x.std(0)])
