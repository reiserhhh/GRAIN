#!/usr/bin/env python3
"""Fixed evidence and instrument interventions, independent of human criterion scores."""

from __future__ import annotations

import re

import numpy as np


def binary_query_factorial():
    """Cross semantic framing with identical two-code alphabets and aligned polarity."""
    frames = {
        "big5_binary": "Judge the writer's Big Five Extraversion, meaning sociability, assertiveness, and energetic engagement with other people. Choose between a relatively low and a relatively high level.",
        "mbti_binary": "Judge which MBTI preference is more strongly suggested for this writer. Choose between Introversion and Extraversion.",
    }
    records = {}
    for framing, body in frames.items():
        low, high = (
            ("a relatively low level", "a relatively high level")
            if framing == "big5_binary"
            else ("Introversion", "Extraversion")
        )
        for coding, alphabet in (("digits", ("0", "1")), ("letters", ("A", "B"))):
            negative, positive = alphabet
            key = framing + "_" + coding
            records[key] = {
                "framing": framing,
                "coding": coding,
                "labels": list(alphabet),
                "values": [0.0, 1.0],
                "style": "core",
                "question": body
                + f" Return only {negative} for {low}, or {positive} for {high}.",
            }
    return records


def tokenize_question(tokenizer, texts, definition):
    """Tokenize a new frozen question with the existing core or narrow scaffold."""
    from .context_carrier import query_region_mask
    from .direct_traits import SYSTEM

    if definition["style"] == "narrow":
        from scripts.run_grain_narrow_functions import reference_view

        text, spans, encoded, mask = reference_view(
            tokenizer, texts, definition["question"]
        )
    else:
        user = "The excerpts below were written by one anonymous writer.\n\n"
        spans = []
        for i, body in enumerate(texts):
            if not isinstance(body, str) or not body:
                raise ValueError("Nonempty complete excerpts required")
            user += ("\n\n" if i else "") + f"Excerpt {i + 1}:\n"
            spans.append((len(user), len(user) + len(body)))
            user += body
        user += (
            "\nEvidence ends here.\n\nQuestion: " + definition["question"] + "\nAnswer:"
        )
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if text.count(user) != 1:
            raise ValueError("Ambiguous instrument scaffold")
        offset = text.index(user)
        spans = [(a + offset, b + offset) for a, b in spans]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        mask = query_region_mask(
            encoded["offset_mapping"], offset + user.rindex("\nQuestion: ") + 1
        )
    labels = definition["labels"]
    alphabet_ids = [tokenizer.encode(v, add_special_tokens=False) for v in labels]
    if any(len(v) != 1 for v in alphabet_ids) or len(
        {v[0] for v in alphabet_ids}
    ) != len(labels):
        raise ValueError("Distinct single-token response codes required")
    if len(encoded["input_ids"]) > 3056:
        raise ValueError("No silent truncation in instrument packet")
    alphabet = {
        "labels": labels,
        "values": definition["values"],
        "token_ids": [v[0] for v in alphabet_ids],
    }
    return text, spans, encoded, mask, alphabet


def construct_view_reference(observer, full_view, edited_view):
    """Apply the existing two-context, three-node construction to a new question."""
    from .context_carrier import cached_suffix_kwargs, split_context_state
    from .factor_functions import joint_gradient
    from .full_state_edit import align_excerpts, align_states, capture_boundary
    from .query_functional import integrated_matrix, path_state, region_gradient

    ft, fs, fe, fm, alphabet = full_view
    et, es, ee, em, other = edited_view
    if alphabet != other or not np.array_equal(
        np.asarray(fe["input_ids"])[fm], np.asarray(ee["input_ids"])[em]
    ):
        raise ValueError("Reference removal changed query or answer coordinates")
    kept = list(range(1, len(fs)))
    mapping, audit = align_excerpts(ft, et, fs, es, kept, fe, ee)
    full, kw, native = capture_boundary(observer, fe["input_ids"], alphabet)
    _, _, g = joint_gradient(observer, full, fm, kw, alphabet)
    edited, _, _ = capture_boundary(observer, ee["input_ids"], alphabet)
    aligned = align_states(full, edited, mapping)
    gradients = []
    kwargs = cached_suffix_kwargs(observer.model, edited)
    for e in (0, 1):
        context = aligned if e == 0 else split_context_state(aligned, edited, em, "E")
        nodes = []
        for t in (0.0, 0.5, 1.0):
            _, value = region_gradient(
                observer,
                context,
                path_state(aligned[:, em], edited[:, em], t),
                em,
                kwargs,
                alphabet,
            )
            nodes.append(value)
        gradients.append(nodes)
    return {
        "Z": full[0, fm].cpu().numpy(),
        "h": full[0, -1].cpu().numpy(),
        "G": integrated_matrix(gradients),
        "g": g,
        "b": native["conditional_score"],
    }, audit


def natural_evidence_edits(texts):
    """Select the first explicit first-person proposition, then alter only its use.

    Wrappers are experimental counterfactual context, not claims about the author.
    The original body stays byte-identical inside each wrapper. Other excerpts
    stay unmodified. No model attribution or human label chooses the excerpt.
    """
    if len(texts) < 2 or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("At least two natural excerpts required")
    pattern = re.compile(
        r"\bI\s+(?:am|feel|think|believe|like|love|hate|want|prefer|enjoy|have|was|do)\b|\bI['\u2019]m\b",
        re.IGNORECASE,
    )
    chosen = next((i for i, t in enumerate(texts) if pattern.search(t)), None)
    if chosen is None:
        return None
    body = texts[chosen]
    wrappers = {
        "self_report": 'This is my own account of my experience: "' + body + '"',
        "other_report": "This is a friend's account of their experience, not my own: \""
        + body
        + '"',
        "endorsed": 'I endorse this description of my experience: "' + body + '"',
        "denied": 'I do not endorse this description of my experience: "' + body + '"',
        "today": 'Earlier today I wrote this description of my experience: "'
        + body
        + '"',
        "past": 'Several years ago I wrote this description of my experience: "'
        + body
        + '"',
    }
    variants = {"original": list(texts)}
    for name, new_body in wrappers.items():
        variants[name] = list(texts)
        variants[name][chosen] = new_body
    variants["remove"] = [t for i, t in enumerate(texts) if i != chosen]
    return {
        "selected_excerpt": chosen,
        "selection_rule": "first explicit first-person proposition",
        "variants": variants,
        "matched_pairs": [
            ("self_report", "other_report", "actor"),
            ("endorsed", "denied", "endorsement"),
            ("today", "past", "time"),
            ("original", "remove", "removal"),
        ],
    }


TIME_EVENTS = (
    (
        "I felt sad and could not get started on anything",
        "I felt cheerful and ready to start things",
    ),
    (
        "I felt lonely even when others were nearby",
        "I felt connected to the people around me",
    ),
    (
        "I felt discouraged after small setbacks",
        "I felt hopeful even after small setbacks",
    ),
    (
        "I thought nothing I did was worthy of praise",
        "I was able to appreciate what I had done",
    ),
    (
        "I felt downhearted and had little motivation",
        "I felt encouraged and had enough motivation",
    ),
    (
        "I felt alone and saw little to look forward to",
        "I felt supported and had things to look forward to",
    ),
)


def temporal_worlds():
    """Cross event-time assignment and display order while retaining both event contents."""
    rows = []
    for i, (negative, positive) in enumerate(TIME_EVENTS):
        for negative_now in (False, True):
            past, current = (
                (positive, negative) if negative_now else (negative, positive)
            )
            earlier = "Six months ago, " + past[0].lower() + past[1:] + "."
            # Keep first-person I capitalized after the time clause.
            earlier = earlier.replace(", i ", ", I ")
            today = "Today, " + current + "."
            for reverse in (False, True):
                rows.append(
                    {
                        "world_id": f"t{i:02d}_n{int(negative_now)}_r{int(reverse)}",
                        "scene_id": f"t{i:02d}",
                        "split": "construction" if i < 2 else "held_scene",
                        "negative_now": negative_now,
                        "reverse_display": reverse,
                        "texts": [today, earlier] if reverse else [earlier, today],
                        "event_labels_are_human_personality_truth": False,
                    }
                )
    return rows
