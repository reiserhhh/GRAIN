#!/usr/bin/env python3
"""Persona-conditioned generation, blinded readout and displacement statistics.

Materials are authored and synthetic. Persona statements come in alternates so
that the two poles of an axis can be chosen with identical token counts (needed
for position-aligned patching). Nothing here reads human records or labels.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict

import numpy as np

AXES = ("extraversion", "neuroticism")
POLES = {"high": 1, "low": -1}
FAMILIES = ("named", "described", "anchored")
SHARED_SUFFIX = (
    " Write naturally in the first person. Do not mention or describe your "
    "personality traits."
)
PERSONAS = {
    ("extraversion", "high", "named"): [
        "You are a highly extraverted person: outgoing, talkative, and energized by company.",
        "You are a very extraverted person: outgoing, talkative, and energized by company.",
        "You are an extraverted person: outgoing, talkative, and energized by other people.",
        "You are a highly extraverted person: outgoing, talkative, and energized by people.",
    ],
    ("extraversion", "low", "named"): [
        "You are a highly introverted person: reserved, quiet, and drained by company.",
        "You are a highly introverted person: reserved, quiet, and quickly drained by company.",
        "You are a highly introverted person: reserved, quiet, and easily drained by other people.",
        "You are a very introverted person: reserved, quiet, and soon drained by company.",
        "You are a highly introverted person: reserved, quiet, and drained by other people's company.",
        "You are a highly introverted person: reserved, quiet, and easily drained by other people around you.",
    ],
    ("extraversion", "high", "described"): [
        "You feel most alive around other people, you start conversations easily, and you speak up without hesitation.",
        "You feel most alive around people, you start conversations easily, and you speak up without hesitation.",
        "You come alive around other people, you start conversations easily, and you speak up freely.",
        "You feel most alive around other people, you start conversations easily, and you speak up freely.",
    ],
    ("extraversion", "low", "described"): [
        "You feel most at ease on your own, you rarely start conversations, and you speak up only when necessary.",
        "You feel most at ease alone, you rarely start conversations, and you speak up only when necessary.",
        "You feel most at ease on your own, you rarely start conversations, and you speak up only when needed.",
        "You feel most at ease alone, you rarely start conversations, and you speak up only when needed.",
    ],
    ("extraversion", "high", "anchored"): [
        "When you enter a room of strangers you walk over and introduce yourself first.",
        "When you enter a room of strangers you go over and introduce yourself first.",
        "When you enter a room of strangers you cross the room and greet people first.",
        "When you enter a room full of strangers you walk over and introduce yourself first.",
    ],
    ("extraversion", "low", "anchored"): [
        "When you enter a room of strangers you hang back until someone approaches you.",
        "When you enter a room of strangers you stay quiet until someone approaches you.",
        "When you enter a room of strangers you wait quietly until someone greets you.",
        "When you enter a room of strangers you keep near the wall and wait to be approached.",
        "When you enter a room of strangers you stay near the wall and wait to be spoken to.",
    ],
    ("neuroticism", "high", "named"): [
        "You are a highly neurotic person: anxious, easily upset, and prone to worry.",
        "You are a very neurotic person: anxious, easily upset, and prone to worry.",
        "You are a highly neurotic person: anxious, easily upset, and given to worry.",
        "You are a highly neurotic person: anxious, easily upset, and full of worry.",
    ],
    ("neuroticism", "low", "named"): [
        "You are an emotionally stable person: calm, rarely upset, and free of worry.",
        "You are a highly emotionally stable person: calm, rarely upset, and free of worry.",
        "You are an emotionally stable person: calm, rarely upset, and free from worry.",
        "You are a very emotionally stable person: calm, rarely upset, and free of worry.",
    ],
    ("neuroticism", "high", "described"): [
        "You worry a great deal, small setbacks unsettle you for days, and tension is never far away.",
        "You worry a great deal, small setbacks unsettle you for days, and tension never feels far away.",
        "You worry constantly, small setbacks unsettle you for days, and tension is never far away.",
    ],
    ("neuroticism", "low", "described"): [
        "You take setbacks in stride, you rarely feel unsettled for long, and calm comes easily to you.",
        "You take setbacks in stride, you rarely stay unsettled for long, and calm comes easily to you.",
        "You take setbacks calmly, you rarely feel unsettled for long, and calm comes easily to you.",
    ],
    ("neuroticism", "high", "anchored"): [
        "When plans change at the last minute your stomach knots and you replay it for hours.",
        "When plans change at the last minute your stomach tightens and you replay it for hours.",
        "When plans change at the last minute your stomach knots and you replay it all evening.",
        "When plans change at the last minute your stomach knots and you keep replaying it for hours.",
        "When plans change at the last moment your stomach knots and you replay it for hours.",
    ],
    ("neuroticism", "low", "anchored"): [
        "When plans change at the last minute you shrug, adjust, and forget about it within minutes.",
        "When plans change at the last minute you shrug, adapt, and forget about it within minutes.",
        "When plans change at the last minute you shrug and forget about it within minutes.",
        "When plans change at the last minute you simply shrug and move on within minutes.",
        "When plans change at the last minute you shrug, adjust, and move on quickly.",
        "When plans change at the last minute you shrug it off and move on within minutes.",
        "When plans change at the last minute you shrug and move on within minutes.",
    ],
}
NEUTRAL = "You are a writer."
NONPERSONA = {
    "spelling": "You always use British spelling and avoid abbreviations.",
    "newsletter": "You are writing for a community newsletter that prefers plain language.",
}
SITUATIONS = [
    ("diary", "Write a diary entry of about 150 words about a community event you attended last weekend."),
    ("email", "Write an email of about 150 words to a coworker about a setback in a shared project."),
    ("opinion", "Write a short opinion piece of about 150 words on a new policy that closes your street to cars on Sundays."),
    ("advice", "Write a reply of about 150 words to a friend who is asking for advice about planning a solo trip."),
    ("review", "Write a product review of about 150 words of the headphones you bought last month."),
    ("neighbor", "Write a note of about 150 words to a neighbor about late-night noise."),
]
DESIGNS = {
    "pilot": {"families": ("named", "described"), "situations": 4, "seeds": (11, 23)},
    "expanded": {"families": ("named", "described", "anchored"), "situations": 6, "seeds": (11, 23, 37)},
}
LEAK = re.compile(
    r"\b(?:extra|extro|intro)ver\w*|\bneurotic\w*|\bpersonalit\w*|"
    r"\bemotionally stable\b|\bemotional stability\b",
    re.IGNORECASE,
)
REFUSAL = re.compile(r"\bAs an AI\b|\bI am an AI\b|\blanguage model\b", re.IGNORECASE)
TOPIC = {
    "diary": ("event", "weekend", "community"),
    "email": ("project", "setback", "team"),
    "opinion": ("street", "cars", "sunday"),
    "advice": ("trip", "travel", "solo"),
    "review": ("headphones", "sound", "bought"),
    "neighbor": ("noise", "night", "neighbor"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def choose_aligned(tokenizer, axis, family):
    """Pick the first high/low alternate pair with identical token counts."""
    highs = PERSONAS[(axis, "high", family)]
    lows = PERSONAS[(axis, "low", family)]
    length = {}
    for text in highs + lows:
        length[text] = len(tokenizer.encode(text, add_special_tokens=False))
    for high in highs:
        for low in lows:
            if length[high] == length[low]:
                return high, low, length[high]
    raise ValueError(f"No token-aligned pair for {axis}/{family}: {[(t, length[t]) for t in highs + lows]}")


def build_materials(tokenizer, design="pilot"):
    """Return the complete factorial as rows; no condition is filtered by outcome."""
    spec = DESIGNS[design]
    situations = SITUATIONS[: spec["situations"]]
    rows, aligned = [], {}
    for axis in AXES:
        for family in spec["families"]:
            high, low, n = choose_aligned(tokenizer, axis, family)
            aligned[f"{axis}:{family}"] = {"high": high, "low": low, "span_tokens": n}
    for sit_id, user in situations:
        for seed in spec["seeds"]:
            for axis in AXES:
                for family in spec["families"]:
                    pair = aligned[f"{axis}:{family}"]
                    for pole in ("high", "low"):
                        rows.append(_row(f"{axis}_{pole}", axis, POLES[pole], family, sit_id, seed, pair[pole], user))
            rows.append(_row("neutral", "none", 0, "none", sit_id, seed, NEUTRAL, user))
            for key, statement in NONPERSONA.items():
                rows.append(_row(f"nonpersona_{key}", "none", 0, "none", sit_id, seed, statement, user))
    return rows, aligned


def _row(condition, axis, pole, family, situation, seed, persona, user):
    system = persona + SHARED_SUFFIX
    text_id = f"{condition}|{family}|{situation}|s{seed}"
    return {
        "text_id": text_id,
        "condition": condition,
        "axis": axis,
        "pole": pole,
        "family": family,
        "situation": situation,
        "seed": seed,
        "persona": persona,
        "system": system,
        "user": user,
        "block": f"{situation}|{family}|s{seed}",
    }


def split_roles(design="pilot"):
    """Situation roles: discovery (fit/site selection), development, evaluation."""
    spec = DESIGNS[design]
    ids = [s for s, _ in SITUATIONS[: spec["situations"]]]
    if design == "pilot":
        return {"discovery": ids[:2], "development": ids[2:3], "evaluation": ids[3:4]}
    return {"discovery": ids[:3], "development": ids[3:4], "evaluation": ids[4:6]}


def audit_text(text, situation, finish_reason):
    words = re.findall(r"[A-Za-z']+", text)
    lower = text.lower()
    first_person = sum(w.lower() in {"i", "me", "my", "mine", "myself"} for w in words)
    return {
        "words": len(words),
        "leak_hits": len(LEAK.findall(text)),
        "refusal": bool(REFUSAL.search(text)),
        "first_person_share": first_person / max(len(words), 1),
        "topic_hits": sum(k in lower for k in TOPIC.get(situation, ())),
        "truncated": finish_reason == "length",
        "empty": len(words) < 20,
    }


def strip_leak(text):
    return LEAK.sub("[X]", text)


def auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels)
    pos, neg = s[y == 1], s[y == -1]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def balanced_accuracy(pred, labels):
    p, y = np.asarray(pred), np.asarray(labels)
    if not (y == 1).any() or not (y == -1).any():
        return None
    return float(0.5 * (((p == 1) & (y == 1)).sum() / (y == 1).sum() + ((p == -1) & (y == -1)).sum() / (y == -1).sum()))


def paired_signflip_null(scores, labels, blocks, n=2000, seed=0):
    """Null AUC by swapping high/low labels within each matched block."""
    rng = np.random.default_rng(seed)
    s, y, b = np.asarray(scores, float), np.asarray(labels), np.asarray(blocks)
    groups = defaultdict(list)
    for i, key in enumerate(b):
        groups[key].append(i)
    null = []
    for _ in range(n):
        yy = y.copy()
        for idx in groups.values():
            if rng.random() < 0.5:
                yy[idx] = -yy[idx]
        null.append(auc(s, yy))
    observed = auc(s, y)
    null = np.asarray([v for v in null if v is not None])
    return {"observed": observed, "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "p_greater": float(np.mean(null >= observed)) if observed is not None else None, "n": len(null)}


def ledoit_wolf(X):
    """Shrinkage covariance (Ledoit-Wolf) without relying on scikit availability."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(np.asarray(X, float))
    return lw.covariance_, float(lw.shrinkage_)


def mahalanobis2(delta, cov):
    return float(np.asarray(delta) @ np.linalg.solve(cov, np.asarray(delta)))


def within_condition_covariance(features, rows):
    """Ledoit-Wolf shrinkage covariance of residuals around condition x family x situation cells."""
    cells = defaultdict(list)
    for r in rows:
        if r["text_id"] in features:
            cells[(r["condition"], r["family"], r["situation"])].append(np.asarray(features[r["text_id"]], float))
    residuals = []
    for vals in cells.values():
        if len(vals) > 1:
            mean = np.mean(vals, axis=0)
            residuals.extend(v - mean for v in vals)
    if len(residuals) < 3:
        raise ValueError("Not enough replicated cells for a within-condition covariance")
    cov, shrinkage = ledoit_wolf(np.asarray(residuals))
    return cov, shrinkage, len(residuals)


def displacement_summary(features, rows, axis, cov=None):
    """Between-pole displacement per block against within-condition seed spread.

    The metric is the within-condition (seed) covariance with Ledoit-Wolf
    shrinkage, never the pooled covariance that already contains the pole effect.
    """
    f = {r["text_id"]: np.asarray(features[r["text_id"]]) for r in rows if r["text_id"] in features}
    if cov is None:
        cov, _, _ = within_condition_covariance(features, rows)
    by = defaultdict(dict)
    for r in rows:
        if r["axis"] == axis and r["text_id"] in f:
            by[(r["family"], r["situation"], r["seed"])][r["pole"]] = f[r["text_id"]]
    between = [v[1] - v[-1] for v in by.values() if 1 in v and -1 in v]
    within = defaultdict(list)
    for r in rows:
        if r["axis"] == axis and r["text_id"] in f:
            within[(r["condition"], r["family"], r["situation"])].append(f[r["text_id"]])
    within_d = []
    for vals in within.values():
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                within_d.append(vals[i] - vals[j])
    mean_between = np.mean(between, axis=0) if between else None
    return {
        "pairs": len(between),
        "mean_displacement": mean_between.tolist() if mean_between is not None else None,
        "D2_mean_displacement": mahalanobis2(mean_between, cov) if mean_between is not None else None,
        "D2_per_pair_median": float(np.median([mahalanobis2(d, cov) for d in between])) if between else None,
        "D2_within_seed_median": float(np.median([mahalanobis2(d, cov) for d in within_d])) if within_d else None,
        "within_pairs": len(within_d),
    }
