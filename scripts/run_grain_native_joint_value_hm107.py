#!/usr/bin/env python3
"""HM107: current-token joint value intervention and actual native decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_reading as J
from grain_research.construction import math_native_joint_value as M
from grain_research.construction.math_construct_routing import query
from grain_research.vector_mechanisms.navigation import MODEL, write_json
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_joint_reading_hm106 as P
from scripts import run_grain_text_response_hm104 as T
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_native_joint_value_hm107.json"
CODE = (
    "configs/grain_native_joint_value_hm107.json",
    "docs/GRAIN_NATIVE_JOINT_VALUE_PLAN_HM107.md",
    "grain_research/construction/math_native_joint_value.py",
    "scripts/run_grain_native_joint_value_hm107.py",
    "tests/test_grain_math_native_joint_value.py",
    "grain_research/construction/math_joint_reading.py",
    "grain_research/construction/math_construct_routing.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/prefix.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
)
UNITS = ("score", "log_odds", "mass", "entropy")


def check(out):
    """Authenticate all code, model, source and precommitted experiment inputs."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    P.check(ROOT / m["spec"]["account"])
    return m


def prepare(out):
    """Seal all 5208 actual full prompts and edits before native logits exist."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Never overwrite prior or partial research")
    spec = H.load(SPEC)
    account, source = ROOT / spec["account"], ROOT / spec["source"]
    P.check(account)
    if (
        H.load(account / "verification.json")["status"]
        != "VERIFIED_FINITE_HM106_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM106 required")
    rows = H.load(source / "00/rows.json")
    lookup = {
        tuple(r[k] for k in ("bank", "bridge", "carrier", "axis", "level", "owner")): r
        for r in rows
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    system = H.load(ROOT / "configs/grain_math_passive_event_composition_v1.json")[
        "system"
    ]
    marker = "__GRAIN_NATIVE_BODY__"
    template = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix, closing = template.split(marker)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    alphabet = {
        "labels": [str(i) for i in range(10)],
        "values": (np.arange(10) / 9).tolist(),
    }
    ids = [tokenizer.encode(x, add_special_tokens=False) for x in alphabet["labels"]]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != 10:
        raise ValueError("Single-token digit alphabet required")
    alphabet["token_ids"] = [x[0] for x in ids]
    suffixes = [
        tokenizer.encode(
            "\n\nEvidence ends here.\n\nQuestion: "
            + query(t).replace("the writer's", "Person A's")
            + "\nAnswer:"
            + closing,
            add_special_tokens=False,
        )
        for t in spec["targets"]
    ]
    groups, design, edits = [], [], {}
    for number, (key, keys, values, q, d) in enumerate(P.cases(source, spec)):
        picked = [
            lookup[*key, level, owner]
            for level, owner in ((2, 2), (0, 2), (2, 0), (0, 0))
        ]
        if any(r["input_ids"][: len(prefix_ids)] != prefix_ids for r in picked):
            raise ValueError("Native template changes the inherited system prefix")
        joint, random = M.joint_edits(values, key)
        groups.append({"group": number, "coordinates": list(key), "rows": picked})
        edits[f"g{number}__joint"] = joint
        edits[f"g{number}__random"] = random
        edits[f"g{number}__old_k"] = keys
        edits[f"g{number}__old_v"] = values
        edits[f"g{number}__q"] = q
        edits[f"g{number}__d"] = d
        arms = spec["arms"] + (["noop"] if list(key) == spec["noop_group"] else [])
        for corner, target, arm in itertools.product(range(4), range(6), arms):
            design.append(
                {
                    "unit": len(design),
                    "group": number,
                    "corner": corner,
                    "target": target,
                    "arm": arm,
                }
            )
    if len(groups) != 72 or len(design) != spec["max_forwards"]:
        raise ValueError("Full finite native battery required")
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    H.write(out / "00/groups.json", groups)
    H.write(out / "00/design.json", design)
    H.write(out / "00/queries.json", {"suffixes": suffixes, "alphabet": alphabet})
    np.savez_compressed(out / "00/edits.npz", **edits)
    sources = [
        source / "00/rows.json",
        source / "00/reader.json",
        source / "01/raw_seal.json",
        account / "summary.json",
        account / "verification.json",
        account / "output_seal.json",
        MODEL / "model.safetensors",
        MODEL / "config.json",
        MODEL / "tokenizer.json",
    ]
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {n: H.sha(ROOT / n) for n in CODE},
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in sources},
            "inputs": H.seals(out, out / "00"),
            "old_capture_count": len(rows),
            "new_forwards": len(design),
            "new_fit": False,
        },
    )
    write_json(
        out / "status.json",
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "model_seconds": 0.0,
            "pending": None,
        },
    )
    print(
        "HM107 PREPARED",
        len(groups),
        "groups",
        len(design),
        "native forwards",
        flush=True,
    )


def collect(out):
    """Run a finite resumable collector; terminal failures never auto-restart."""
    m = check(out)
    spec = m["spec"]
    status = H.load(out / "status.json")
    if status["state"] not in ("PREPARED", "RUNNING") or status["pending"] is not None:
        raise RuntimeError("Terminal or uncertain packet cannot restart automatically")
    design = H.load(out / "00/design.json")
    groups = H.load(out / "00/groups.json")
    queries = H.load(out / "00/queries.json")
    with np.load(out / "00/edits.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    clock = time.monotonic()
    observer = load_observer()
    status["model_seconds"] += time.monotonic() - clock
    write_json(out / "model_manifest.json", observer.manifest())
    with collector_guard(out):
        for item in design[status["completed"] :]:
            if status["model_seconds"] >= spec["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                write_json(out / "status.json", status)
                raise TimeoutError("Finite cumulative model budget exhausted")
            row = groups[item["group"]]["rows"][item["corner"]]
            stem = out / f"01/units/unit_{item['unit']:05d}"
            if stem.with_suffix(".json").exists() or stem.with_suffix(".npz").exists():
                raise RuntimeError(
                    "Existing uncertain unit requires review, not replay"
                )
            tag = f"g{item['group']}__"
            delta = (
                arrays[tag + ("joint" if item["arm"] == "joint_removed" else "random")][
                    item["corner"]
                ]
                if item["arm"] in ("joint_removed", "random")
                else np.zeros_like(arrays[tag + "old_v"][item["corner"]])
            )
            status.update(
                state="RUNNING", pending=item, attempts=status["attempts"] + 1
            )
            write_json(out / "status.json", status)
            start = time.monotonic()
            try:
                result, raw = M.native_forward(
                    observer,
                    row,
                    queries["suffixes"][item["target"]],
                    queries["alphabet"],
                    delta,
                    spec["layer"],
                )
                prefix_error = max(
                    float(np.max(abs(raw[name] - arrays[tag + old][item["corner"]])))
                    for name, old in (("k", "old_k"), ("before", "old_v"))
                )
                edit_error = float(np.max(abs(raw["after"] - raw["before"] - delta)))
                np.savez_compressed(stem.with_suffix(".npz"), **raw)
                H.write(
                    stem.with_suffix(".json"),
                    {
                        **item,
                        **result,
                        "sha256": H.sha(stem.with_suffix(".npz")),
                        "prefix_error": prefix_error,
                        "edit_error": edit_error,
                        "seconds": time.monotonic() - start,
                    },
                )
                if (
                    prefix_error > spec["prefix_tolerance"]
                    or edit_error > spec["edit_tolerance"]
                ):
                    raise ValueError(
                        "Actual prefix or value edit failed frozen numerical gate"
                    )
            except Exception as exc:
                status.update(state="FAILED", error=f"{type(exc).__name__}: {exc}")
                status["model_seconds"] += time.monotonic() - start
                write_json(out / "status.json", status)
                raise
            status["model_seconds"] += time.monotonic() - start
            status.update(completed=item["unit"] + 1, pending=None)
            write_json(out / "status.json", status)
            if status["completed"] % 144 == 0:
                print(
                    "HM107",
                    status["completed"],
                    "/",
                    len(design),
                    "seconds",
                    round(status["model_seconds"], 2),
                    flush=True,
                )
    status["state"] = "CAPTURED"
    write_json(out / "status.json", status)
    H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    print("HM107 CAPTURED", status, flush=True)


def outputs(out, independent=False):
    """Read each actual arm and recompute all observables with a second formula."""
    m = check(out)
    if H.load(out / "status.json")["state"] != "CAPTURED":
        raise ValueError("Complete actual native collection required")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    design = H.load(out / "00/design.json")
    groups = H.load(out / "00/groups.json")
    fields = np.empty((72, 3, 4, 6, 4))
    maximum = {
        "prefix": 0.0,
        "edit": 0.0,
        "noop": 0.0,
        "units": 0.0,
        "native_identity": 0.0,
        "local_identity": 0.0,
        "author": 0.0,
        "stored_units": 0.0,
    }
    arm_order = m["spec"]["arms"]
    noops, baseline = {}, {}
    with np.load(out / "00/edits.npz", allow_pickle=False) as a:
        for item in design:
            meta = H.load(out / f"01/units/unit_{item['unit']:05d}.json")
            z = np.asarray(meta["candidate_logits"])
            units = M.native_units(z, meta["full_vocab_log_normalizer"])
            stored = np.array(
                [
                    meta["conditional_score"],
                    units[1],
                    meta["candidate_mass"],
                    meta["conditional_entropy"],
                ]
            )
            maximum["stored_units"] = max(
                maximum["stored_units"], float(np.max(abs(units - stored)))
            )
            if independent:
                weights = np.exp(z - z.max())
                weights /= weights.sum()
                second = np.array(
                    [
                        sum(weights[i] * i / 9 for i in range(10)),
                        np.log(
                            np.exp(z[5:] - z.max()).sum()
                            / np.exp(z[:5] - z.max()).sum()
                        ),
                        sum(np.exp(z - meta["full_vocab_log_normalizer"])),
                        -sum(weights[i] * np.log(weights[i]) for i in range(10)),
                    ]
                )
                maximum["units"] = max(
                    maximum["units"], float(np.max(abs(units - second)))
                )
            maximum["prefix"] = max(maximum["prefix"], meta["prefix_error"])
            maximum["edit"] = max(maximum["edit"], meta["edit_error"])
            address = item["group"], item["corner"], item["target"]
            if item["arm"] == "noop":
                noops[address] = units
            else:
                fields[
                    item["group"],
                    arm_order.index(item["arm"]),
                    item["corner"],
                    item["target"],
                ] = units
                if item["arm"] == "baseline":
                    baseline[address] = units
        for key, value in noops.items():
            maximum["noop"] = max(
                maximum["noop"], float(np.max(abs(value - baseline[key])))
            )
        chars = np.einsum("jc,gacqu->gajqu", J.BASIS / 4, fields)
        local, local_after, local_random = [], [], []
        for group in groups:
            tag = f"g{group['group']}__"
            k, v, q, d = [a[tag + name] for name in ("old_k", "old_v", "q", "d")]
            p, c = J.primitives(k, v, q, d)
            old_scores = (p * c).sum(axis=(1, 3))
            scores = []
            for name in ("joint", "random"):
                pp, cc = J.primitives(k, v + a[tag + name], q, d)
                scores.append((pp * cc).sum(axis=(1, 3)))
            old_char = J.coefficients(old_scores)
            after, rand = [J.coefficients(s) for s in scores]
            value_term = J.product_law(p, c)[3, 0]
            maximum["local_identity"] = max(
                maximum["local_identity"],
                float(np.max(abs(old_char[3] - after[3] - value_term))),
            )
            local.append(old_char)
            local_after.append(after)
            local_random.append(rand)
        if independent:
            for g in range(72):
                for arm in range(3):
                    reconstructed = np.stack(
                        [
                            sum(J.BASIS[j, c] * fields[g, arm, c] for c in range(4)) / 4
                            for j in range(4)
                        ]
                    )
                    maximum["native_identity"] = max(
                        maximum["native_identity"],
                        float(np.max(abs(reconstructed - chars[g, arm]))),
                    )
                    plus, minus = (1 + 0.5 * J.BASIS[3]) / 4, (1 - 0.5 * J.BASIS[3]) / 4
                    contrast = sum(
                        (plus[c] - minus[c]) * fields[g, arm, c] for c in range(4)
                    )
                    maximum["author"] = max(
                        maximum["author"],
                        float(np.max(abs(contrast - chars[g, arm, 3]))),
                    )
    return (
        m,
        groups,
        fields,
        chars,
        np.stack([local, local_after, local_random], 1),
        maximum,
    )


def analyze(out):
    """Report complete native/local fields, not a winner or an accuracy leaderboard."""
    m, groups, fields, chars, local, maximum = outputs(out)
    if (out / "summary.json").exists():
        raise FileExistsError("Preserve existing analyses")
    rows, effects, authors = [], [], []
    for group in groups:
        g = group["group"]
        base = dict(
            zip(
                ("bank", "bridge", "carrier", "action"),
                group["coordinates"],
                strict=True,
            )
        )
        for arm, character, target in itertools.product(range(3), range(4), range(6)):
            rows.append(
                {
                    **base,
                    "arm": m["spec"]["arms"][arm],
                    "character": J.CHARACTERS[character],
                    "target": m["spec"]["targets"][target],
                    "local": local[g, arm, character, target],
                    **dict(zip(UNITS, chars[g, arm, character, target], strict=True)),
                }
            )
        for character, target in itertools.product(range(4), range(6)):
            effects.append(
                {
                    **base,
                    "character": J.CHARACTERS[character],
                    "target": m["spec"]["targets"][target],
                    "local_removed": local[g, 0, character, target]
                    - local[g, 1, character, target],
                    "local_random": local[g, 0, character, target]
                    - local[g, 2, character, target],
                    **{
                        f"{unit}_{arm}": chars[g, 0, character, target, u]
                        - chars[g, a, character, target, u]
                        for a, arm in ((1, "removed"), (2, "random"))
                        for u, unit in enumerate(UNITS)
                    },
                }
            )
        for arm, target in itertools.product(range(3), range(6)):
            authors.append(
                {
                    **base,
                    "arm": m["spec"]["arms"][arm],
                    "target": m["spec"]["targets"][target],
                    "local_law_contrast": local[g, arm, 3, target],
                    **{
                        unit + "_law_contrast": chars[g, arm, 3, target, u]
                        for u, unit in enumerate(UNITS)
                    },
                }
            )
    (out / "02").mkdir()
    H.table(out / "02/characters.csv", rows)
    H.table(out / "02/effects.csv", effects)
    H.table(out / "02/author_pairs.csv", authors)
    np.savez_compressed(out / "02/laws.npz", fields=fields, chars=chars, local=local)
    banks = {}
    for bank in range(3):
        ix = [g["group"] for g in groups if g["coordinates"][0] == bank]
        summary = {}
        for u, unit in enumerate(UNITS[:2]):
            full = chars[ix, 0, 3, :, u]
            removed = full - chars[ix, 1, 3, :, u]
            random = full - chars[ix, 2, 3, :, u]
            norm = np.vdot(full, full)
            summary[unit] = {
                "native_joint": T.stat(full),
                "removed_effect": T.stat(removed),
                "random_effect": T.stat(random),
                "removed_alignment": float(np.vdot(removed, full) / norm)
                if norm
                else None,
                "random_alignment": float(np.vdot(random, full) / norm)
                if norm
                else None,
                "native_local_cosine": P.cosine(full, local[ix, 0, 3]),
                "effect_local_cosine": P.cosine(
                    removed, local[ix, 0, 3] - local[ix, 1, 3]
                ),
                "per_target": {
                    target: {
                        "native": T.stat(full[:, q]),
                        "removed": T.stat(removed[:, q]),
                        "random": T.stat(random[:, q]),
                    }
                    for q, target in enumerate(m["spec"]["targets"])
                },
            }
        banks[str(bank)] = summary
    H.write(
        out / "summary.json",
        {
            "status": "ANALYZED_FINITE_HM107_NOT_GLOBAL",
            "banks": banks,
            "checks": maximum,
            "actual_status": H.load(out / "status.json"),
            "groups": 72,
            "classification": m["spec"]["classification"],
            "native_not_human_traits": True,
            "original_scope_complete": False,
        },
    )
    H.write(
        out / "output_seal.json",
        H.seals(out, out / "02") | {"summary.json": H.sha(out / "summary.json")},
    )
    print(
        "HM107 ANALYZED",
        {
            b: {
                u: {
                    k: v[k]
                    for k in (
                        "native_joint",
                        "removed_effect",
                        "random_effect",
                        "removed_alignment",
                        "native_local_cosine",
                    )
                }
                for u, v in s.items()
            }
            for b, s in banks.items()
        },
        flush=True,
    )


def verify(out):
    """Independently recompute digit units/characters/author laws and actual edits."""
    m, _, fields, chars, local, maximum = outputs(out, independent=True)
    H.verify_hashes(out, H.load(out / "output_seal.json"))
    spec = m["spec"]
    with np.load(out / "02/laws.npz", allow_pickle=False) as saved:
        if any(
            not np.array_equal(saved[k], v)
            for k, v in (("fields", fields), ("chars", chars), ("local", local))
        ):
            raise ValueError("Saved complete native/local law changed")
    groups = H.load(out / "00/groups.json")
    query_inputs = H.load(out / "00/queries.json")
    lookup = {tuple(g["coordinates"]): g["group"] for g in groups}
    maximum["tables"] = 0.0
    for name, count in (
        ("characters", 5184),
        ("effects", 1728),
        ("author_pairs", 1296),
    ):
        seen = set()
        with (out / f"02/{name}.csv").open() as stream:
            for row in csv.DictReader(stream):
                g = lookup[
                    tuple(int(row[k]) for k in ("bank", "bridge", "carrier", "action"))
                ]
                q = spec["targets"].index(row["target"])
                j = (
                    J.CHARACTERS.index(row["character"])
                    if name != "author_pairs"
                    else 3
                )
                arm = spec["arms"].index(row["arm"]) if name != "effects" else -1
                key = g, arm, j, q
                if key in seen:
                    raise ValueError("Duplicate scientific table address")
                seen.add(key)
                if name == "characters":
                    expected = {
                        "local": local[g, arm, j, q],
                        **dict(zip(UNITS, chars[g, arm, j, q], strict=True)),
                    }
                elif name == "effects":
                    expected = {
                        "local_removed": local[g, 0, j, q] - local[g, 1, j, q],
                        "local_random": local[g, 0, j, q] - local[g, 2, j, q],
                        **{
                            f"{unit}_{a}": sum(
                                J.BASIS[j, c]
                                * (fields[g, 0, c, q, u] - fields[g, i, c, q, u])
                                for c in range(4)
                            )
                            / 4
                            for i, a in ((1, "removed"), (2, "random"))
                            for u, unit in enumerate(UNITS)
                        },
                    }
                else:
                    expected = {
                        "local_law_contrast": local[g, arm, 3, q],
                        **{
                            unit + "_law_contrast": sum(
                                0.25 * J.BASIS[3, c] * fields[g, arm, c, q, u]
                                for c in range(4)
                            )
                            for u, unit in enumerate(UNITS)
                        },
                    }
                maximum["tables"] = max(
                    maximum["tables"],
                    max(abs(float(row[k]) - v) for k, v in expected.items()),
                )
        if len(seen) != count:
            raise ValueError("Missing scientific table entries")
    with np.load(out / "00/edits.npz", allow_pickle=False) as edits:
        realized = 0.0
        for item in H.load(out / "00/design.json"):
            path = out / f"01/units/unit_{item['unit']:05d}.npz"
            with np.load(path, allow_pickle=False) as raw:
                tag = f"g{item['group']}__"
                delta = (
                    edits[
                        tag + ("joint" if item["arm"] == "joint_removed" else "random")
                    ][item["corner"]]
                    if item["arm"] in ("joint_removed", "random")
                    else np.zeros_like(raw["before"])
                )
                realized = max(
                    realized, float(np.max(abs(raw["after"] - raw["before"] - delta)))
                )
                source_row = groups[item["group"]]["rows"][item["corner"]]
                n = len(source_row["input_ids"]) + len(
                    query_inputs["suffixes"][item["target"]]
                )
                support = np.tri(n, dtype=bool)
                support[:, source_row["padding"]] = False
                expected_hash = np.frombuffer(
                    hashlib.sha256(support.tobytes()).digest(), dtype=np.uint8
                )
                if not np.array_equal(raw["support_hash"], expected_hash):
                    raise ValueError(
                        "Actual native mask support differs from declared support"
                    )
                maximum["prefix"] = max(
                    maximum["prefix"],
                    float(
                        np.max(
                            abs(raw["before"] - edits[tag + "old_v"][item["corner"]])
                        )
                    ),
                    float(np.max(abs(raw["k"] - edits[tag + "old_k"][item["corner"]]))),
                )
                if not np.array_equal(raw["masked_blocks"], np.arange(28)) or len(
                    raw["before"]
                ) != len(groups[item["group"]]["rows"][item["corner"]]["current"]):
                    raise ValueError("Incomplete actual hook/support trace")
        maximum["realized_edit"] = realized
    if (
        maximum["prefix"] > spec["prefix_tolerance"]
        or max(maximum["edit"], maximum["realized_edit"]) > spec["edit_tolerance"]
        or maximum["noop"] > spec["noop_tolerance"]
        or max(
            maximum[k]
            for k in (
                "units",
                "native_identity",
                "local_identity",
                "author",
                "stored_units",
                "tables",
            )
        )
        > spec["arithmetic_tolerance"]
    ):
        raise ValueError("Finite frozen gate failed: " + str(maximum))
    H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM107_NOT_GLOBAL",
            "alternate_checks": maximum,
            "all_groups": 72,
            "same_agent_not_independent_reviewer": True,
            "actual_forwards": H.load(out / "status.json")["attempts"],
        },
    )
    print("HM107 VERIFIED", maximum, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / H.load(SPEC)["output"])
    args = parser.parse_args()
    globals()[args.phase](args.output)
