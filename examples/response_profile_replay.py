#!/usr/bin/env python3
"""Replay behavior profiles on two public corpora. Publish aggregates, never raw turns."""

import argparse
import itertools
import json
import re
import zipfile
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from grain_core.response_profile import measure


def ref(behavior, reader, condition):
    return {
        "behavior": behavior,
        "reader": reader,
        "condition_rule": condition,
        "frame": "observed_replies",
        "unit": "next adjacent other-speaker utterance",
        "weights": {"cue": 0.5, "other": 0.5},
        "contrast": ["cue", "other"],
    }


def row(person, dialog, i, cue, outcome):
    return {
        "focal": person,
        "dialogue": str(dialog),
        "id": str(i - 1),
        "response_id": str(i),
        "condition": "cue" if cue else "other",
        "status": "support" if outcome else "refute",
    }


def _hash(label, name):
    return sha256((label + ":" + name).encode()).digest()[0]


def rpc(path):
    reference = ref(
        "question mark in focal reply",
        "unicode-question-mark-v1",
        "question mark in immediately preceding partner message",
    )
    panels = [[], []]
    partners = defaultdict(set)
    blocks = defaultdict(Counter)
    dialogues = 0
    with zipfile.ZipFile(path) as z:
        for name in sorted(
            n for n in z.namelist() if "/dialogues/" in n and n.endswith(".json")
        ):
            turns = json.loads(z.read(name))["utterances"]
            speakers = {t["interlocutor_id"] for t in turns}
            focal = [s for s in speakers if _hash("focal", s) < 128]
            if len(speakers) != 2 or len(focal) != 1:
                continue
            who = focal[0]
            peer = (speakers - {who}).pop()
            k = int(_hash("partner", peer) >= 128)
            dialogues += 1
            for i, (a, b) in enumerate(itertools.pairwise(turns), 1):
                if (
                    a["interlocutor_id"] != peer
                    or b["interlocutor_id"] != who
                    or not a["text"].strip()
                    or not b["text"].strip()
                ):
                    continue
                cue = bool(re.search("[?？]", a["text"]))
                yes = bool(re.search("[?？]", b["text"]))
                panels[k].append({**row(who, name, i, cue, yes), "partner": peer})
                partners[who, k].add(peer)
                blocks[who, k, name][int(cue), int(yes)] += 1
    # Null expectation when the marginal reply and cue frequencies are held
    # fixed INSIDE each dialogue. This removes between-dialogue mixture.
    residual = defaultdict(float)
    for (who, k, _), tab in blocks.items():
        nr = tab[0, 1] + tab[1, 1]
        nc = tab[1, 0] + tab[1, 1]
        residual[who, k] += tab[1, 1] - nc * nr / tab.total()
    scores = []
    standard_errors = {}
    for k, part in enumerate(panels):
        score = {}
        for c in measure(reference, part, sampling_cluster="partner")["cards"]:
            who = c["focal"]
            standard_errors[who, k] = c["sampling_precision"]["delete_one_cluster_se"][
                "response_contrast"
            ]
            n0, n1 = (c["conditions"][key]["n"] for key in ("other", "cue"))
            if min(n0, n1) < 5 or len(partners[who, k]) < 2:
                continue
            score[who] = [
                c["natural"][0],
                c["standardized"][0],
                c["response_contrast"][0],
                residual[who, k] * (1 / n0 + 1 / n1),
                n1 / (n0 + n1),
            ]
        scores.append(score)
    keep = sorted(scores[0].keys() & scores[1].keys())
    if len(keep) < 4:
        raise ValueError("insufficient repeated people in disjoint partner panels")
    a, b = (np.array([s[k] for k in keep]) for s in scores)

    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if np.std(x) * np.std(y) > 0 else None

    rng = np.random.default_rng(518)
    null = np.stack([b[rng.permutation(len(b))] for _ in range(200)])
    # Fit on other focal speakers in A only. Account for a person's usual punctuation,
    # its bounded scale, and how often that person was addressed with a question mark.
    X = np.column_stack([np.ones(len(keep)), a[:, 0], a[:, 0] ** 2, a[:, 4]])
    baseline = np.empty(len(keep))
    for i in range(len(keep)):
        train = np.arange(len(keep)) != i
        baseline[i] = X[i] @ np.linalg.lstsq(X[train], a[train, 2], rcond=None)[0]
    err0 = (baseline - b[:, 2]) ** 2
    err1 = (a[:, 2] - b[:, 2]) ** 2
    return {
        "dyads": dialogues,
        "reply_counts": [len(x) for x in panels],
        "eligible_people": len(keep),
        "partner_precision": [
            {
                "estimable": len(ses),
                "median_contrast_se": float(np.median(ses)) if ses else None,
            }
            for k in (0, 1)
            for ses in [
                [
                    standard_errors[u, k]
                    for u in keep
                    if standard_errors[u, k] is not None
                ]
            ]
        ],
        "held_partner_contrast_check": {
            "usual_rate_and_cue_mix_mse": float(err0.mean()),
            "own_panel_A_contrast_mse": float(err1.mean()),
            "people_lower_error_using_own": int(sum(err1 < err0)),
            "marginal_baseline_inputs": "panel A only; leave focal out of coefficient fit; linear + squared natural rate + cue share",
        },
        "inclusion": "at least 2 distinct partners and 5 readings per condition in each panel",
        "same_person_across_disjoint_partners": {
            key: {
                "pearson_r": corr(a[:, j], b[:, j]),
                "shuffled_person_median_r": float(
                    np.median([corr(a[:, j], u[:, j]) for u in null])
                ),
            }
            for j, key in enumerate(
                [
                    "natural",
                    "standardized",
                    "cue_minus_other",
                    "within_dialogue_excess_contrast",
                ]
            )
        },
        "reference": reference,
    }


def casino(path):
    docs = json.loads(path.read_text(encoding="utf-8"))
    reference = ref(
        "focal reply annotated self-need or other-need",
        "CaSiNo strategy annotations",
        "preceding partner utterance annotated elicit-pref",
    )
    rows = []
    confusion = defaultdict(Counter)
    marked_prompts = []
    pairs = Counter()
    matched = 0
    for d in docs:
        ann = d["annotations"]
        if not ann:
            continue
        turns = d["chat_logs"]
        speech = [(i, t) for i, t in enumerate(turns) if not t["task_data"]]
        if len(speech) != len(ann) or any(
            t["text"] != v[0] for (_, t), v in zip(speech, ann)
        ):
            continue
        matched += 1
        tags = {
            i: set(v[1].split(","))
            for (i, _), v in zip(speech, ann, strict=True)
            if v[1].strip()
        }
        for i, (a, b) in enumerate(itertools.pairwise(turns), 1):
            if i - 1 not in tags or i not in tags or a["id"] == b["id"]:
                continue
            cue = "elicit-pref" in tags[i - 1]
            yes = bool(tags[i] & {"self-need", "other-need"})
            r = row("all observed replies", d["dialogue_id"], i, cue, yes)
            rows.append(r)
            if "?" in a["text"] or "？" in a["text"]:
                marked_prompts.append(r)
            pairs[str(d["dialogue_id"]), b["id"], r["condition"], yes] += 1
            lexical = bool(re.search(r"\bneeds?\b", b["text"], re.IGNORECASE))
            confusion[r["condition"]][int(yes), int(lexical)] += 1
    c = measure(reference, rows)["cards"][0]
    within = []
    for did, role in sorted({(k[0], k[1]) for k in pairs}):

        def rate(cond, did=did, role=role):
            y = pairs[did, role, cond, True]
            n = y + pairs[did, role, cond, False]
            return y / n if n else None

        h, l = rate("cue"), rate("other")
        if h is not None and l is not None:
            within.append(h - l)
    return {
        "dialogues": len(docs),
        "annotated_dialogues": sum(bool(d["annotations"]) for d in docs),
        "exactly_aligned": matched,
        "profile": c,
        "question_mark_prompts_only": measure(reference, marked_prompts)["cards"][0],
        "reference": reference,
        "within_dialogue_role_pairs": len(within),
        "equal_role_mean_contrast": float(np.mean(within)),
        "need_word_vs_annotation": {
            cond: {
                "tp": t[1, 1],
                "fn": t[1, 0],
                "fp": t[0, 1],
                "tn": t[0, 0],
                "sensitivity": t[1, 1] / (t[1, 0] + t[1, 1]),
                "false_positive_rate": t[0, 1] / (t[0, 1] + t[0, 0]),
            }
            for cond, t in confusion.items()
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("rpc", "casino", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    result = {"rpc": rpc(a.rpc), "casino": casino(a.casino)}
    with a.out.open("x", encoding="utf-8") as f:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), file=f)


if __name__ == "__main__":
    main()
