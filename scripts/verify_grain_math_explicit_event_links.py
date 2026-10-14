#!/usr/bin/env python3
"""HM59 raw-binding and graph/finite-Fourier replay; no model calls or fitting."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import verify_grain_math_event_relation_renaming as V

OUT = ROOT / "results/grain_math_response_v1/explicit_event_links_r1"


def parse_relations(text):
    """Independently parse the two declared fixed syntax families, without scores."""
    heading, first, second = text.splitlines()
    actor = re.fullmatch(
        r"The following event identity statements apply only to Person ([AB])'s occasions\.",
        heading,
    )
    if actor is None:
        raise ValueError("Missing explicit event scope")
    edges = []
    for line in (first, second):
        a = re.fullmatch(
            r"Occasion (K[1-4]) is (the same event as|a different event from) occasion (K[1-4])\.",
            line,
        )
        b = re.fullmatch(
            r"(K[1-4]) and (K[1-4]) refer to (one event|separate events)\.", line
        )
        if a:
            left, relation, right = a.groups()
            same = relation == "the same event as"
        elif b:
            left, right, relation = b.groups()
            same = relation == "one event"
        else:
            raise ValueError("Undeclared relation syntax")
        edges.append({"actor": actor[1], "a": left, "b": right, "same": same})
    if {(e["a"], e["b"]) for e in edges} != {("K3", "K1"), ("K4", "K2")} or sum(
        e["same"] for e in edges
    ) != 1:
        raise ValueError("Exactly one of the two fixed links must be an alias")
    return edges


def graph_weights(row):
    """Use connected components, rather than the production incidence formula."""
    relations = parse_relations(row["declaration"])
    if relations != row["declared_relations"]:
        raise ValueError("Text-to-edge metadata mismatch")
    records = row["history"]
    nodes = {(r["actor"], r["occasion_code"]) for r in records}
    graph = {node: set() for node in nodes}
    for edge in relations:
        x, y = (edge["actor"], edge["a"]), (edge["actor"], edge["b"])
        if edge["same"] and x in nodes and y in nodes:
            graph[x].add(y)
            graph[y].add(x)
    components, unseen = [], set(nodes)
    while unseen:
        queue, component = [min(unseen)], set()
        while queue:
            node = queue.pop()
            if node in component:
                continue
            component.add(node)
            queue.extend(graph[node] - component)
        components.append(component)
        unseen -= component
    roots = {node: min(component) for component in components for node in component}
    sizes = {node: len(component) for component in components for node in component}
    weights = []
    for r in records:
        node, root = (
            (r["actor"], r["occasion_code"]),
            roots[r["actor"], r["occasion_code"]],
        )
        expected = {
            "event_id": root[0] + ":" + root[1],
            "kind": "new" if root == node else "alias",
        }
        if root != node:
            expected["parent_id"] = "r" + root[1][1]
        for key, value in expected.items():
            if r.get(key) != value:
                raise ValueError("Canonical event/parent binding mismatch")
        if root == node and "parent_id" in r:
            raise ValueError("Original claim must not acquire a parent")
        weights.append(1 / (len(components) * sizes[node]))
    return np.asarray(weights), len(components)


def stats(rows):
    """Recompute descriptive means and RMS with scalar dot products."""
    result = {"cases": len(rows)}
    for key in ("score", "oriented_score", "log_odds", "probability_L1_half"):
        x = [r[key] for r in rows]
        result[key + "_mean"] = sum(x) / len(x)
        result[key + "_rms"] = float(np.sqrt(sum(v * v for v in x) / len(x)))
    result["oriented_positive"] = sum(r["oriented_score"] > 0 for r in rows)
    return result


def verify(out):
    """Replay every raw law, reference weight, contrast and group coefficient."""
    from transformers import AutoTokenizer

    V.MAX_ERROR = 0.0
    m, state = V.read(out / "00/manifest.json"), V.read(out / "run_status.json")
    tol, spec = m["spec"]["arithmetic_tolerance"], m["spec"]
    if (
        m["packet"] != "HM59"
        or state["status"] != "COLLECTED"
        or state["in_flight"] is not None
    ):
        raise ValueError("Complete terminal HM59 required")
    if (
        any(
            state[k] != 384
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        )
        or state["model_seconds"] > 900
    ):
        raise ValueError("Actual acquisition accounting mismatch")
    for name, sha in m["code"].items():
        if V.sha(ROOT / name) != sha or V.sha(out / "00/code" / name) != sha:
            raise ValueError("Frozen code drift:" + name)
    for mapping, root in ((m["assets"], out), (m["source_hashes"], ROOT)):
        for name, sha in mapping.items():
            if V.sha(root / name) != sha:
                raise ValueError("Frozen material/source drift:" + name)
    for name, sha in V.read(out / "03/artifact_hashes.json").items():
        if V.sha(out / name) != sha:
            raise ValueError("Scientific output drift:" + name)
    if V.sha(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Local model config drift")
    if V.sha(ROOT / m["observer_reference"]) != m["observer_reference_sha256"]:
        raise ValueError("Observer reference drift")
    actual, prior = (
        V.read(out / "01/observer.json"),
        V.read(ROOT / m["observer_reference"]),
    )
    for key in (
        "model",
        "torch",
        "transformers",
        "dtype",
        "device",
        "hidden_size",
        "layers",
        "max_tokens",
        "attention",
        "training",
        "template_sha256",
    ):
        if actual[key] != prior[key]:
            raise ValueError("Actual observer mismatch:" + key)
    atoms = {}
    source = ROOT / spec["source"]
    sm = V.read(source / "00/manifest.json")
    for r in V.read(source / "00/rows.json"):
        if r["role"] != "atom":
            continue
        path = (
            ROOT / sm["reused"][str(r["unit"])]["meta"]
            if r["reused"]
            else source / f"01/units/unit_{r['unit']:04d}.json"
        )
        raw = V.read(path)
        if V.sha(path.with_suffix(".npz")) != raw["array_sha256"]:
            raise ValueError("Atom array drift")
        atoms[
            (r["content"], r["family"], r["polarity"], r["history"][0]["event_id"])
        ] = softmax(raw["native"]["candidate_logits"])
    saved = V.read(out / "00/atoms.json")
    if (
        len(atoms) != 96
        or len(saved) != 96
        or len({tuple(r["key"]) for r in saved}) != 96
    ):
        raise ValueError("Full actual-code atom bank required")
    for r in saved:
        for a, b in zip(r["probabilities"], atoms[tuple(r["key"])], strict=True):
            V.compare(a, b, tol)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    rows, pairs = V.read(out / "00/rows.json"), V.read(out / "00/pairs.json")
    behaviours = V.read(ROOT / spec["behaviour_reference"])["spec"]["behaviours"]
    cells = {
        (
            r["content"],
            r["family"],
            r["polarity"],
            r["order"],
            r["syntax"],
            r["declaration_actor"],
            r["scheme"],
        )
        for r in rows
    }
    if len(rows) != 384 or cells != set(
        itertools.product(
            m["targets"],
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            ("A", "B"),
            ("focus_alias", "companion_alias"),
        )
    ):
        raise ValueError("Incomplete material design")
    native, laws, refs, weight_rows, observations = [], [], [], [], {}
    forward_seconds = 0.0
    for r in rows:
        sequence = (
            ["K1", "K2", "K3", "K4"] if r["order"] == 0 else ["K4", "K3", "K2", "K1"]
        )
        lines = []
        for code in sequence:
            focal = code in ("K1", "K3")
            family = r["family"] if focal else 1 - r["family"]
            polarity = r["polarity"] if focal else 1 - r["polarity"]
            lines.append(
                f"On occasion {code}, Person A {behaviours[r['content']][family][polarity]}."
            )
        if len(r["history"]) != 4:
            raise ValueError("Four material-bound mention records required")
        for record, code in zip(r["history"], sequence, strict=True):
            focal = code in ("K1", "K3")
            family = r["family"] if focal else 1 - r["family"]
            polarity = r["polarity"] if focal else 1 - r["polarity"]
            bound = {
                "record_id": "r" + code[1],
                "occasion_code": code,
                "actor": "A",
                "family": family,
                "polarity": polarity,
                "condition": 1 if family == 0 else -1,
            }
            if any(record.get(key) != value for key, value in bound.items()):
                raise ValueError("Text-to-behavior/history identity mismatch")
        if (
            r["action_block"] != "\n".join(lines)
            or r["body"] != r["action_block"] + "\n\n" + r["declaration"]
        ):
            raise ValueError("Action/whole-body identity drift")
        edges = parse_relations(r["declaration"])
        if edges[0]["actor"] != r["declaration_actor"] or edges[0]["same"] != (
            r["scheme"] == "focus_alias"
        ):
            raise ValueError("Declaration condition mismatch")
        text, ids, part, cues, actor = render(
            r["body"], r["content"], "A", 1, m["system"], tokenizer
        )
        if (text, ids, part, cues, actor) != (
            r["rendered"],
            r["input_ids"],
            r["partition"],
            r["cue_slots"],
            r["actor_slot"],
        ):
            raise ValueError("Actual tokenizer/chat-input mismatch")
        offsets = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        start = text.index(r["body"])
        for key, lo, hi in (
            ("action_positions", start, start + len(r["action_block"])),
            (
                "declaration_positions",
                start + len(r["action_block"]) + 2,
                start + len(r["body"]),
            ),
        ):
            expected = [
                i for i, (a, b) in enumerate(offsets) if a < hi and b > lo and b > a
            ]
            if r[key] != expected:
                raise ValueError("Token address mismatch:" + key)
        path = out / f"01/units/unit_{r['unit']:04d}.json"
        raw = V.read(path)
        if raw["row"] != r or V.sha(path.with_suffix(".npz")) != raw["array_sha256"]:
            raise ValueError("Raw input/array binding mismatch")
        forward_seconds += raw["forward_seconds"]
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
            width = actual["hidden_size"]
            for key, length in (
                ("declaration", len(r["declaration_positions"])),
                ("suffix", len(part["suffix"])),
            ):
                if (
                    arrays[key].shape != (length, width)
                    or not np.isfinite(arrays[key]).all()
                ):
                    raise ValueError("Capture dimensions/finiteness mismatch")
            if arrays["endpoint"].shape != (width,) or not np.array_equal(
                arrays["endpoint"], arrays["suffix"][-1]
            ):
                raise ValueError("Endpoint binding mismatch")
        logits = np.asarray(raw["native"]["candidate_logits"])
        p = softmax(logits)
        v = np.arange(10) / 9
        score = float(sum(x * y for x, y in zip(p, v, strict=True)))
        odds = float(logsumexp(logits[5:]) - logsumexp(logits[:5]))
        V.compare(raw["native"]["conditional_score"], score, tol)
        for a, b in zip(raw["native"]["conditional_probabilities"], p, strict=True):
            V.compare(a, b, tol)
        w, count = graph_weights(r)
        x = np.array(
            [
                atoms[(r["content"], h["family"], h["polarity"], h["occasion_code"])]
                for h in r["history"]
            ]
        )
        z = np.array(
            [
                sum(
                    atoms[(r["content"], h["family"], h["polarity"], code)]
                    for code in ("K1", "K2", "K3", "K4")
                )
                / 4
                for h in r["history"]
            ]
        )
        profiles = {
            "event_actual_code": np.sum(w[:, None] * x, axis=0),
            "event_code_neutral": np.sum(w[:, None] * z, axis=0),
            "mention_actual_code": np.sum(x, axis=0) / 4,
        }
        ref_scores = {k: float(sum(p * v)) for k, p in profiles.items()}
        observations[r["unit"]] = {
            "score": score,
            "log_odds": odds,
            "p": p,
            "profiles": profiles,
            "ref_scores": ref_scores,
        }
        keys = {
            k: r[k]
            for k in (
                "unit",
                "content",
                "query",
                "family",
                "polarity",
                "order",
                "syntax",
                "declaration_actor",
                "scheme",
            )
        }
        native.append(
            {
                **keys,
                "score": score,
                "log_odds": odds,
                "candidate_mass": raw["native"]["candidate_mass"],
                "events": count,
                "mentions": 4,
                **{k + "_score": s for k, s in ref_scores.items()},
            }
        )
        laws.extend(
            {"unit": r["unit"], "candidate": j, "probability": float(x)}
            for j, x in enumerate(p)
        )
        for basis, distribution in profiles.items():
            refs.extend(
                {
                    "unit": r["unit"],
                    "basis": basis,
                    "candidate": j,
                    "probability": float(x),
                }
                for j, x in enumerate(distribution)
            )
        for position, h in enumerate(r["history"]):
            weight_rows.append(
                {
                    "unit": r["unit"],
                    "position": position,
                    "record_id": h["record_id"],
                    "occasion_code": h["occasion_code"],
                    "event_id": h["event_id"],
                    "event_weight": float(w[position]),
                    "mention_weight": 0.25,
                }
            )
    V.compare(state["native_forward_seconds"], forward_seconds, tol)
    if len(pairs) != 192 or len({(p["lo"], p["hi"]) for p in pairs}) != 192:
        raise ValueError("Unique complete scheme pairs required")
    proofs = V.read(out / "00/token_proofs.json")
    if len(proofs) != 192:
        raise ValueError("Complete token proofs required")
    paired, lookup = [], {}
    for pair, proof in zip(pairs, proofs, strict=True):
        low, high = rows[pair["lo"]], rows[pair["hi"]]
        if any(
            low[k] != high[k] or low[k] != pair[k]
            for k in (
                "content",
                "family",
                "polarity",
                "order",
                "syntax",
                "declaration_actor",
            )
        ):
            raise ValueError("Unmatched pair conditions")
        if (low["scheme"], high["scheme"]) != (
            "focus_alias",
            "companion_alias",
        ) or pair["orientation"] != 2 * low["polarity"] - 1:
            raise ValueError("Scheme/orientation binding mismatch")
        a, b = low["input_ids"], high["input_ids"]
        if (
            Counter(a) != Counter(b)
            or len(a) != len(b)
            or low["action_block"] != high["action_block"]
        ):
            raise ValueError("Actual matched token/content proof failed")
        if Counter(re.findall(r"\w+|[^\w\s]", low["body"])) != Counter(
            re.findall(r"\w+|[^\w\s]", high["body"])
        ):
            raise ValueError("Raw word bag mismatch")
        changed = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
        if (
            a[: max(low["action_positions"]) + 1]
            != b[: max(high["action_positions"]) + 1]
            or low["partition"]["suffix"] != high["partition"]["suffix"]
        ):
            raise ValueError("Action prefix/suffix positions mismatch")
        if [a[i] for i in low["partition"]["suffix"]] != [
            b[i] for i in high["partition"]["suffix"]
        ] or not set(changed) <= set(low["declaration_positions"]) | set(
            high["declaration_positions"]
        ):
            raise ValueError("Changes outside relation declarations")
        expected_proof = {
            "lo": low["unit"],
            "hi": high["unit"],
            "full_length": len(a),
            "changed_positions": changed,
            "word_bag_equal": True,
            "full_token_bag_equal": True,
            "action_prefix_equal": True,
            "query_suffix_equal": True,
            "not_digit_only": True,
        }
        if proof != expected_proof:
            raise ValueError("Saved token proof drift")
        x, y = observations[pair["lo"]], observations[pair["hi"]]
        delta, odds = y["score"] - x["score"], y["log_odds"] - x["log_odds"]
        result = {
            **pair,
            "score": delta,
            "oriented_score": delta * pair["orientation"],
            "log_odds": odds,
            "oriented_log_odds": odds * pair["orientation"],
            "probability_L1_half": float(sum(abs(y["p"] - x["p"])) / 2),
        }
        for basis in x["profiles"]:
            result[basis + "_score"] = y["ref_scores"][basis] - x["ref_scores"][basis]
            result[basis + "_tv"] = float(
                sum(abs(y["profiles"][basis] - x["profiles"][basis])) / 2
            )
        paired.append(result)
        lookup[
            tuple(
                pair[k]
                for k in (
                    "content",
                    "family",
                    "polarity",
                    "order",
                    "syntax",
                    "declaration_actor",
                )
            )
        ] = result
    interactions, chars = [], []
    for c, f, p, order, syntax in itertools.product(
        m["targets"], (0, 1), (0, 1), (0, 1), (0, 1)
    ):
        a, b = (
            lookup[(c, f, p, order, syntax, "A")],
            lookup[(c, f, p, order, syntax, "B")],
        )
        delta = a["score"] - b["score"]
        signed = (
            observations[a["hi"]]["p"]
            - observations[a["lo"]]["p"]
            - observations[b["hi"]]["p"]
            + observations[b["lo"]]["p"]
        )
        interactions.append(
            {
                "interaction": len(interactions),
                "content": c,
                "family": f,
                "polarity": p,
                "order": order,
                "syntax": syntax,
                "score": delta,
                "oriented_score": delta * (2 * p - 1),
                "log_odds": a["log_odds"] - b["log_odds"],
                "probability_L1_half": float(sum(abs(signed)) / 2),
                "A_pair": a["pair"],
                "B_pair": b["pair"],
                "event_code_neutral_score": a["event_code_neutral_score"]
                - b["event_code_neutral_score"],
            }
        )
    cells = list(itertools.product((0, 1), repeat=3))
    hadamard = np.array(
        [
            [
                (-1) ** sum(m * b for m, b in zip(mask, cell, strict=True))
                for cell in cells
            ]
            for mask in cells
        ],
        float,
    )
    for c, f, p in itertools.product(m["targets"], (0, 1), (0, 1)):
        values = np.array(
            [
                lookup[(c, f, p, o, s, "A" if scope == 0 else "B")]["score"]
                for scope, s, o in cells
            ]
        )
        coefficients = hadamard @ values / 8
        if np.max(np.abs(hadamard.T @ coefficients - values)) > tol:
            raise ValueError("Complete character reconstruction failed")
        for mask, value in zip(cells, coefficients, strict=True):
            chars.append(
                {
                    "content": c,
                    "family": f,
                    "polarity": p,
                    "character_scope_syntax_order": "".join(map(str, mask)),
                    "coefficient": float(value),
                }
            )
    groups = []
    for name, data in (("scheme", paired), ("actor_interaction", interactions)):
        groups.append({"kind": name, "group": "all", **stats(data)})
        for key in ("content", "syntax", "order"):
            for value in sorted({r[key] for r in data}):
                groups.append(
                    {
                        "kind": name,
                        "group": key + "=" + str(value),
                        **stats([r for r in data if r[key] == value]),
                    }
                )
        if name == "scheme":
            for actor in ("A", "B"):
                part = [r for r in data if r["declaration_actor"] == actor]
                for content in ("all", *sorted({r["content"] for r in data})):
                    group = (
                        part
                        if content == "all"
                        else [r for r in part if r["content"] == content]
                    )
                    groups.append(
                        {
                            "kind": name,
                            "group": f"actor={actor}/content={content}",
                            **stats(group),
                        }
                    )
    tables = {
        "native": (native, ("unit",)),
        "probabilities": (laws, ("unit", "candidate")),
        "reference_laws": (refs, ("unit", "basis", "candidate")),
        "event_weights": (weight_rows, ("unit", "position")),
        "paired_changes": (paired, ("pair",)),
        "actor_interactions": (interactions, ("interaction",)),
        "character_components": (
            chars,
            ("content", "family", "polarity", "character_scope_syntax_order"),
        ),
        "summary_by_group": (groups, ("kind", "group")),
    }
    counts = {
        name: V.table_check(out, name, expected, keys, tol)
        for name, (expected, keys) in tables.items()
    }
    summary = V.read(out / "03/summary.json")
    if (
        summary["rows"] != counts
        or summary["original_scope_complete"]
        or summary["actual_calls"] != 384
    ):
        raise ValueError("Summary scope/count mismatch")
    for actual_group, expected_group in zip(summary["summaries"], groups, strict=True):
        for field, value in expected_group.items():
            V.compare(actual_group[field], value, tol)
    report = {
        "status": "VERIFIED_FINITE_HM59_NOT_GLOBAL",
        "raw_readouts": 384,
        "token_proofs": 192,
        "table_rows": counts,
        "actual_new_calls": 384,
        "maximum_alternate_error": V.MAX_ERROR,
        "minimum_candidate_mass": min(r["candidate_mass"] for r in native),
        "native_forward_seconds": forward_seconds,
        "model_seconds": state["model_seconds"],
        "fit_patch_generation_API_private_training": [0] * 6,
        "same_agent_alternate_formulas_not_third_party_replication": True,
        "original_scope_complete": False,
    }
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


def main():
    """Replay a specific frozen finite packet, without acquisition or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    verify(parser.parse_args().out)


if __name__ == "__main__":
    main()
