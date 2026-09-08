#!/usr/bin/env python3
"""Controlled relational worlds and frozen-reference navigation bookkeeping.

World records specify actions and opportunities, never human personality truth.
No private author text or psychological labels are needed to build the packet.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/grain_vector_mechanisms"
MODEL = BASE / "local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
OUTPUT = BASE / "navigation_v1/NAV-01/controlled_r1"
QUERIES = ("extraversion", "TF_cont", "adapted_dejected5")
KEYS = ("Z", "h", "G", "g", "b")
SCENES = (
    (
        "construction",
        "social",
        "at a neighborhood gathering",
        "introduced myself to people I had not met and invited them to join a conversation",
    ),
    (
        "construction",
        "social",
        "at a club meeting",
        "sat away from the group and said I would rather spend the evening alone",
    ),
    (
        "construction",
        "decision",
        "during a project review",
        "compared the numerical evidence and said the same decision rule should apply even if someone felt disappointed",
    ),
    (
        "construction",
        "decision",
        "while allocating team duties",
        "asked how each person would feel and changed the plan to avoid hurting a teammate",
    ),
    (
        "construction",
        "setback",
        "after an unsuccessful application",
        "said I felt lonely and discouraged and could not find the motivation to try again",
    ),
    (
        "construction",
        "setback",
        "after missing a personal goal",
        "said I felt hopeful and ready to try another approach despite the setback",
    ),
    (
        "development",
        "social",
        "at a workshop lunch",
        "started a conversation with the table and later asked for a quiet break on my own",
    ),
    (
        "development",
        "decision",
        "while choosing a charity project",
        "checked the cost figures and then asked whether the people affected would feel respected",
    ),
    (
        "development",
        "setback",
        "after a cancelled trip",
        "said I felt let down but still looked forward to making another plan",
    ),
    (
        "held_scene",
        "social",
        "at an unfamiliar sports event",
        "welcomed several strangers and kept the conversation going when the group fell silent",
    ),
    (
        "held_scene",
        "decision",
        "while settling a housing dispute",
        "used a consistent written rule and declined to change it for a personal appeal",
    ),
    (
        "held_scene",
        "setback",
        "after a damaged exhibition",
        "said I felt downhearted and wanted to stop working because the effort seemed pointless",
    ),
)


def digest(path):
    """Hash source or artifact bytes without printing their contents."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    """Read a strict local JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Atomically commit a finite JSON unit; partial files are not completed units."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def render_world(scene, actor, opportunity, realization):
    """Render the same event facts with two surface forms and explicit ownership."""
    if (
        actor not in ("self", "other")
        or opportunity not in ("voluntary", "assigned")
        or realization not in (0, 1)
    ):
        raise ValueError("Unknown controlled-world condition")
    _, _, location, action = SCENES[scene]
    subject = "I" if actor == "self" else "My colleague"
    if actor == "other":
        action = (
            action.replace("myself", "themselves")
            .replace(" I ", " they ")
            .replace(" my ", " their ")
        )
    condition = (
        "Nobody required a particular action or statement; each person could choose freely."
        if opportunity == "voluntary"
        else "The coordinator assigned that action and supplied those exact words as a script to perform."
    )
    statement = f"{subject} {action}."
    if realization == 0:
        event = f"This happened {location}. {statement} {condition}"
    else:
        event = f'In my notes about what happened {location}, I recorded: "{statement}" {condition}'
    neutral = (
        "Afterwards I noted the time and put the remaining materials on the table."
    )
    return [event, neutral]


def build_worlds():
    """Build96 scenarios without assigning a personality score to any simulated person."""
    worlds = []
    for scene, (split, family, location, action) in enumerate(SCENES):
        for actor, opportunity, realization in product(
            ("self", "other"), ("voluntary", "assigned"), (0, 1)
        ):
            worlds.append(
                {
                    "world_id": f"s{scene:02d}_{actor}_{opportunity}_r{realization}",
                    "scene_id": f"s{scene:02d}",
                    "split": split,
                    "semantic_family": family,
                    "actor": actor,
                    "recipient": "people in the described event",
                    "action": action,
                    "opportunity": opportunity,
                    "negation": False,
                    "quotation_owner": actor,
                    "event_time": "recent_unspecified",
                    "query_referent": "writer",
                    "evidence_reliability": "reported_event_not_verified",
                    "realization": realization,
                    "location": location,
                    "texts": render_world(scene, actor, opportunity, realization),
                    "world_facts_are_personality_labels": False,
                }
            )
    validate_worlds(worlds)
    return worlds


def validate_worlds(worlds):
    """Reject duplicate worlds, scene leakage and accidental personality ground truth."""
    if len(worlds) != 96 or len({w["world_id"] for w in worlds}) != 96:
        raise ValueError("Expected96 unique worlds")
    roles = {}
    for world in worlds:
        roles.setdefault(world["scene_id"], set()).add(world["split"])
        if world.get("world_facts_are_personality_labels") is not False:
            raise ValueError("Synthetic psychological truth is forbidden")
    if any(len(values) != 1 for values in roles.values()):
        raise ValueError("A scene crosses the construction/evaluation boundary")
    if {
        s: sum(s in v for v in roles.values())
        for s in ("construction", "development", "held_scene")
    } != {"construction": 6, "development": 3, "held_scene": 3}:
        raise ValueError("Unexpected scene split")


def tokenize_world(tokenizer, texts, query):
    """Keep original query/scaffold bytes and separate the P region from final Q."""
    from .context_carrier import query_region_mask
    from .direct_traits import SYSTEM, candidates, render

    if query == "adapted_dejected5":
        from scripts.run_grain_narrow_functions import reference_view

        definition = read_json(BASE / "narrow_functions_m30b_v1/queries.json")[query]
        text, spans, encoded, mask = reference_view(
            tokenizer, texts, definition["question"]
        )
        alphabet = candidates("extraversion", tokenizer)
    else:
        user, spans = render(texts, query)
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if text.count(user) != 1:
            raise ValueError("Ambiguous prompt scaffold")
        offset = text.index(user)
        spans = [(lo + offset, hi + offset) for lo, hi in spans]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        mask = query_region_mask(
            encoded["offset_mapping"], offset + user.rindex("\nQuestion: ") + 1
        )
        alphabet = candidates(query, tokenizer)
    if len(encoded["input_ids"]) > 3056:
        raise ValueError("No silent truncation in navigation")
    return text, spans, encoded, mask, alphabet


def real_reference_sources():
    """Identify immutable completed core and narrow reference archives."""
    return {
        "extraversion": BASE
        / "all_factor_function_analysis_m28_v2/nine_query_functions.npz",
        "TF_cont": BASE
        / "all_factor_function_analysis_m28_v2/nine_query_functions.npz",
        "adapted_dejected5": BASE / "narrow_functions_m30b_v1/reference.npz",
    }


def load_functions(path, queries=QUERIES):
    """Load fixed readers, rejecting missing or incompatible coefficient arrays."""
    from .functional_profiles import FunctionalQuery

    with np.load(path, allow_pickle=False) as z:
        return {q: FunctionalQuery(*(z[f"{q}__{k}"] for k in KEYS)) for q in queries}


def quality(actual, predicted):
    """Descriptive model fidelity, not pseudo-replicated human inference statistics."""
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isfinite([y, p]).all():
        raise ValueError("Matched finite observations required")
    r = (
        float(pearsonr(y, p).statistic)
        if len(y) >= 3 and min(np.ptp(y), np.ptp(p)) > 1e-12
        else None
    )
    return {
        "n": len(y),
        "r": r,
        "mae": float(np.mean(abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
    }


def reference_world_ids():
    """Fix two synthetic construction references by event coverage, not model outcomes."""
    return {
        "extraversion": ["s00_self_voluntary_r0", "s01_self_voluntary_r0"],
        "TF_cont": ["s02_self_voluntary_r0", "s03_self_voluntary_r0"],
        "adapted_dejected5": ["s04_self_voluntary_r0", "s05_self_voluntary_r0"],
    }
