#!/usr/bin/env python3
"""One source-qualified direction/assessability observation,no answer fitting."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_person_bound_response as O
from grain_research.construction import math_person_bound_tree as M
from grain_research.construction import math_source_anchor_exchange as E
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_observer_operation_hm280 as N
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts.score_grain_author_belief_cold import load_bundle

CONFIG = "configs/grain_person_bound_observation_tree_hm288.json"
SPEC = R.read(ROOT / CONFIG)
PARENT, DEFAULT = ROOT / SPEC["parent"], ROOT / SPEC["output"]
TARGETS = ("E", "N", "A", "C", "O")
LAWS = ("observed_pair_F", "author_product_J", "F_minus_J")
COMPONENTS = ("total", "composition", "within_configuration")
SCIENCE = (
    CONFIG,
    "docs/GRAIN_PERSON_BOUND_OBSERVATION_TREE_PLAN_HM288.md",
    "grain_research/construction/math_person_bound_tree.py",
    "scripts/run_grain_person_bound_tree_hm288.py",
    "scripts/score_grain_person_bound_tree_cold.py",
    "tests/test_grain_person_bound_tree.py",
    "tests/test_grain_person_bound_response.py",
    "grain_research/construction/math_person_bound_response.py",
    "grain_research/construction/math_behavior_predictive.py",
    "grain_research/construction/math_item_bridge.py",
    "grain_research/construction/math_criterion_components.py",
    "grain_research/construction/math_source_anchor_exchange.py",
    "scripts/run_grain_observer_operation_hm280.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "grain_research/vector_mechanisms/trace.py",
)


def prepare(out):
    """Replace only the observer question;keep source law and every comment token."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No overwrite/retry of scientific outputs")
    parent = R.check(
        PARENT, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    tok = AutoTokenizer.from_pretrained(str(J.MODEL), local_files_only=True)

    def encode(text):
        """Tokenize new question markers only,never original comment bodies."""
        return tok.encode(text, add_special_tokens=False)

    tail = encode("<|im_end|>\n<|im_start|>assistant\n")
    bank = R.read(PARENT / "bank.json")
    alphabet = [encode(str(i)) for i in range(3)]
    if (
        any(len(a) != 1 for a in alphabet)
        or len({a[0] for a in alphabet}) != 3
        or [a[0] for a in alphabet[:2]] != parent["alphabet"]
    ):
        raise ValueError("Original binary plus new third token alphabet required")
    rows, lengths = [], []
    for row in R.read(PARENT / "units.json"):
        sequences = []
        for item, old in zip(bank, row["input_ids"], strict=True):
            boundary = "\nEvidence ends here.\n\nQuestion: "
            old_suffix = (
                encode(boundary + O.question(item["public_text"], row["queried"]))
                + tail
            )
            new_suffix = (
                encode(boundary + M.question(item["public_text"], row["queried"]))
                + tail
            )
            if old[-len(old_suffix) :] != old_suffix:
                raise ValueError("Original question boundary/tail changed")
            prefix = old[: -len(old_suffix)]
            sequences.append(prefix + new_suffix)
            if sequences[-1][: -len(new_suffix)] != prefix:
                raise ValueError("Original complete evidence prefix changed")
        lengths.extend(map(len, sequences))
        rows.append(
            {k: v for k, v in row.items() if k != "input_ids"}
            | {"input_ids": sequences}
        )
    for unit in range(48):
        for t in (-1, 1):
            selected = [
                r["input_ids"][0]
                for r in rows
                if r["unit"] == unit and r["queried"] == t
            ]
            if len(selected) != 4:
                raise ValueError("Full original ownership/display cube required")
            O.assert_same_bag(selected)
    if (
        len(rows) != SPEC["families"]
        or len(rows) * 50 != SPEC["inquiries"]
        or max(lengths) > SPEC["max_input_tokens"]
    ):
        raise ValueError("Full support exceeds cap;no clipping/subsetting")
    if R.digest(J.MODEL / "model.safetensors") != parent["model_weights_sha256"]:
        raise ValueError("Pinned local observer changed")
    paths = [
        PARENT / p
        for p in (
            "manifest.json",
            "units.json",
            "bank.json",
            "cases.json",
            "source.npz",
            "readings.npz",
            "prepared_seal.json",
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        )
    ]
    paths.extend(
        J.MODEL / p
        for p in (
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    )
    (out / "batches").mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "config": SPEC,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in paths},
            "alphabet": [a[0] for a in alphabet],
            "model_weights_sha256": parent["model_weights_sha256"],
            "new_observation_law_not_old_binary_refinement": True,
            "original_scope_complete": False,
        },
    )
    R.write(out / "units.json", rows)
    R.write(out / "bank.json", bank)
    R.write(out / "cases.json", R.read(PARENT / "cases.json"))
    R.archive(out / "source.npz", **load_bundle(PARENT / "source.npz"))
    R.write(
        out / "input_quality.json",
        {
            "families": len(rows),
            "inquiries": len(rows) * 50,
            "authors": 24,
            "comments": 96,
            "input_token_range": [min(lengths), max(lengths)],
            "all_evidence_prefixes_and_bags_preserved": True,
            "new_API_private_training_labels_fit": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("units.json", "bank.json", "cases.json", "source.npz", "input_quality.json"),
    )
    H.status(out, {"state": "PREPARED", "families": 0, "model_seconds": 0})
    print(json.dumps(R.read(out / "input_quality.json")), flush=True)


def collect(out):
    """One bounded cached capture;the sealed binary companion is not modified."""
    N.I.native_items = M.native_items
    N.collect(out)


def observed(out):
    """Load actual role-addressed endpoints and full-alphabet diagnostics."""
    z, error = np.empty((48, 2, 2, 2, 50, 3)), 0.0
    seen, diagnostics = set(), []
    head = load_bundle(out / "head.npz")
    alphabet = R.read(out / "manifest.json")["alphabet"]
    for i, row in enumerate(R.read(out / "units.json")):
        key = (
            row["unit"],
            (row["ownership"] + 1) // 2,
            (row["queried"] + 1) // 2,
            (row["display"] + 1) // 2,
        )
        if key in seen:
            raise ValueError("Duplicate role-addressed observation")
        seen.add(key)
        packet = load_bundle(out / "batches" / f"{i:03d}.npz")
        z[key] = packet["logits"]
        error = max(
            error,
            float(
                abs(
                    packet["states"] @ head["weights"].astype(float).T
                    + head["bias"]
                    - packet["logits"]
                ).max()
            ),
        )
        mass = logsumexp(packet["logits"], -1) - packet["normalizer"]
        if not np.isfinite(mass).all() or max(mass) > 1e-6:
            raise ArithmeticError("Three-category mass not a full-vocabulary subset")
        diagnostics.extend(
            {
                "family": i,
                "item": j,
                "log_category_mass": float(mass[j]),
                "greedy_id": int(packet["top"][j]),
                "greedy_category": alphabet.index(int(packet["top"][j]))
                if int(packet["top"][j]) in alphabet
                else -1,
            }
            for j in range(50)
        )
    if seen != set(np.ndindex(48, 2, 2, 2)) or error > SPEC["native_tolerance"]:
        raise ArithmeticError("Full observations or old adequate tolerance fail")
    return z, error, diagnostics


def analyze(out):
    """Separate tree fields and contract change;fit no score/probe/label."""
    R.check(out, ("collected_seal.json",))
    z, head_error, diagnostics = observed(out)
    source, bank = load_bundle(out / "source.npz"), R.read(out / "bank.json")
    fields, tree = M.readings(z, bank)
    old_logits = load_bundle(PARENT / "readings.npz")["logits"]
    old = O.readings(old_logits, bank)["logodds"]
    keys = ("items", "balanced_key", "common_support", "global_common")
    delta = {key: fields["direction"][key] - old[key] for key in keys}
    arrays, rows = {}, []
    for name, values in dict(fields, contract_direction_change=delta).items():
        arrays[name + "/items"] = values["items"]
        for key in keys[1:]:
            contact = np.stack(
                [
                    np.stack([M.contacts(s, values[key]) for s in space])
                    for space in source["components"]
                ]
            )
            arrays[name + "/" + key] = contact
            if key == "global_common":
                continue
            for sp, space in enumerate(("E5", "GRAIN15")):
                for ref, reference in enumerate(("bank0", "bank1", "pooled")):
                    for law, lawname in enumerate(LAWS):
                        for component, compname in enumerate(COMPONENTS):
                            for mask in range(8):
                                for q, target in enumerate(TARGETS):
                                    for r, query in enumerate(TARGETS):
                                        rows.append(
                                            {
                                                "space": space,
                                                "reference": reference,
                                                "law": lawname,
                                                "component": compname,
                                                "channel": name,
                                                "character": mask,
                                                "source_target": target,
                                                "query": query,
                                                "readout": key,
                                                "contact": float(
                                                    contact[
                                                        sp,
                                                        ref,
                                                        law,
                                                        component,
                                                        mask,
                                                        q,
                                                        r,
                                                    ]
                                                ),
                                            }
                                        )
    primary, alphas, nulls = [], [], []
    e = source["labels"] - source["means"]
    for channel in M.CHANNELS:
        gamma = fields[channel]["global_common"][:, 3]
        alpha = source["relative_weights"][1, 2, 0].T @ gamma / 48
        alphas.append(alpha)
        channel_nulls = []
        for q, target in enumerate(TARGETS):
            value = float(alpha @ e[:, q])
            null = e[source["assignments"][q], q] @ alpha
            variance = float(
                alpha
                @ C.orbit_covariance(source["labels"][:, q], source["groups"][:, q])
                @ alpha
            )
            primary.append(
                {
                    "channel": channel,
                    "target": target,
                    "source_person_bound_common": value,
                    "exact_conditional_mean": 0.0,
                    "exact_conditional_variance": variance,
                    "two_sided_conditional_rank": float(
                        (1 + np.sum(abs(null) >= abs(value))) / 200
                    ),
                }
            )
            channel_nulls.append(null)
        nulls.append(channel_nulls)
    for row, adjusted in zip(
        primary, E.holm([p["two_sided_conditional_rank"] for p in primary]), strict=True
    ):
        row["Holm10_rank"] = float(adjusted)
    reconstruction = M.tree_log_probabilities(tree["direction"], tree["assessability"])
    record = {
        "version": "HM288-v1",
        "cases": R.read(out / "cases.json"),
        "source_components": source["components"][1, 2].tolist(),
        "item_logits": z.tolist(),
        "item_bank": [{k: b[k] for k in ("item", "target", "pole")} for b in bank],
    }
    R.archive(
        out / "readings.npz",
        **arrays,
        alphas=np.array(alphas),
        nulls=np.array(nulls),
        logits=z,
        log_probabilities=tree["log_probabilities"],
    )
    R.table(out / "all_contacts.csv", rows)
    R.table(out / "primary.csv", primary)
    R.table(out / "reader_diagnostics.csv", diagnostics)
    R.write(out / "cold_record.json", record)
    R.write(
        out / "summary.json",
        {
            "primary": primary,
            "tree_reconstruction_error": float(
                abs(reconstruction - tree["log_probabilities"]).max()
            ),
            "head_error": head_error,
            "category_probability_means": np.exp(tree["log_probabilities"])
            .mean(axis=(0, 1, 2, 3, 4))
            .tolist(),
            "greedy_counts_0_1_2_outside": [
                sum(d["greedy_category"] == j for d in diagnostics)
                for j in (0, 1, 2, -1)
            ],
            "all8_global_common_RMS": {
                key: np.sqrt(np.mean(values["global_common"] ** 2, axis=0)).tolist()
                for key, values in dict(fields, contract_direction_change=delta).items()
            },
            "old_binary_common_source_ENACO": R.read(PARENT / "summary.json")[
                "primary"
            ],
            "actual_contract_change_not_only_uncertainty": True,
            "native_runtime": R.read(out / "collected_runtime.json"),
            "not_human_truth_or_accuracy_gain": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "readings.npz",
            "all_contacts.csv",
            "primary.csv",
            "reader_diagnostics.csv",
            "cold_record.json",
            "summary.json",
        ),
    )
    print(json.dumps(R.read(out / "summary.json")))


def verify(out):
    """Literal tree/source sums,categorical closure and cold replay,no new model."""
    R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    source, data = load_bundle(out / "source.npz"), load_bundle(out / "readings.npz")
    summary = R.read(out / "summary.json")
    z, bank = data["logits"], R.read(out / "bank.json")
    d = z[..., 1] - z[..., 0]
    a = np.logaddexp(z[..., 0], z[..., 1]) - z[..., 2]
    errors = {
        "tree_reconstruction": summary["tree_reconstruction_error"],
        "literal_references": 0.0,
        "variance": 0.0,
    }
    for channel, raw in zip(M.CHANNELS, (d, a), strict=True):
        literal = np.zeros((48, 8, 50))
        for mask in range(8):
            for corner in np.ndindex(2, 2, 2):
                signs = [2 * j - 1 for j in corner]
                weight = np.prod([signs[k] for k in range(3) if mask & (1 << k)])
                literal[:, mask] += weight * raw[(slice(None), *corner)] / 8
        errors[channel + "/characters"] = float(
            abs(literal - data[channel + "/items"]).max()
        )
        common = []
        for q in range(5):
            hi = [b["item"] for b in bank if b["target"] == q and b["pole"] == 1]
            lo = [b["item"] for b in bank if b["target"] == q and b["pole"] == -1]
            common.append(
                (
                    sum(literal[:, :, j] for j in hi) / len(hi)
                    + sum(literal[:, :, j] for j in lo) / len(lo)
                )
                / 2
            )
        gamma = np.stack(common, -1).mean(-1)[:, 3]
        cindex = M.CHANNELS.index(channel)
        with threadpool_limits(limits=1):
            for q in range(5):
                e = source["labels"][:, q] - source["means"][:, q]
                saved = np.r_[
                    summary["primary"][cindex * 5 + q]["source_person_bound_common"],
                    data["nulls"][cindex, q],
                ]
                for assignment, value in zip(
                    np.vstack((np.arange(386), source["assignments"][q])),
                    saved,
                    strict=True,
                ):
                    score = np.array(
                        [
                            sum(w * e[assignment])
                            for w in source["relative_weights"][1, 2, 0]
                        ]
                    )
                    direct = sum(score * gamma) / 48
                    errors["literal_references"] = max(
                        errors["literal_references"], abs(direct - value)
                    )
                    if not np.array_equal(
                        source["groups"][assignment, q], source["groups"][:, q]
                    ):
                        raise ArithmeticError(
                            "Original SOURCE assignment cells changed"
                        )
                variance = 0.0
                for group in np.unique(source["groups"][:, q]):
                    ix = source["groups"][:, q] == group
                    y, ell = source["labels"][ix, q], data["alphas"][cindex, ix]
                    m, p = len(y), y.mean()
                    if m > 1:
                        variance += (
                            p
                            * (1 - p)
                            * m
                            / (m - 1)
                            * (sum(ell**2) - sum(ell) ** 2 / m)
                        )
                errors["variance"] = max(
                    errors["variance"],
                    abs(
                        variance
                        - summary["primary"][cindex * 5 + q][
                            "exact_conditional_variance"
                        ]
                    ),
                )
        for key in ("balanced_key", "common_support", "global_common"):
            x = data[channel + "/" + key]
            errors[channel + "/" + key + "/levels"] = float(
                abs(x[:, :, 0] - x[:, :, 1] - x[:, :, 2]).max()
            )
            errors[channel + "/" + key + "/components"] = float(
                abs(x[:, :, :, 0] - x[:, :, :, 1] - x[:, :, :, 2]).max()
            )
    old = load_bundle(PARENT / "readings.npz")
    for key in ("balanced_key", "common_support", "global_common"):
        expected = data["direction/" + key] - old["logodds/" + key]
        errors["contract/" + key] = float(
            abs(expected - data["contract_direction_change/" + key]).max()
        )
    reply = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_person_bound_tree_cold.py"),
            "--input",
            str(out / "cold_record.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cold = json.loads(reply.stdout)
    for channel in M.CHANNELS:
        for key in ("balanced_key", "common_support", "global_common"):
            errors[channel + "/cold/" + key] = float(
                abs(
                    np.asarray(cold["channels"][channel][key])
                    - data[channel + "/" + key][1, 2]
                ).max()
            )
    if max(errors.values()) > 1e-10:
        raise ArithmeticError("Registered tree/source account verification failed")
    R.write(out / "cold_readback.json", cold)
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "errors": errors,
            "literal_primary_and_references": 2000,
            "same_agent_not_blind": True,
            "new_scientific_collection": False,
        },
    )
    R.seal(out, "verification_seal.json", ("cold_readback.json", "verification.json"))
    print(json.dumps(R.read(out / "verification.json")))


def main():
    """Run immutable finite stages;never renew failed captures or old queues."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(
            args.out,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM288 method/input/native/tree/verification seals PASS")
    else:
        {"prepare": prepare, "collect": collect, "analyze": analyze, "verify": verify}[
            args.stage
        ](args.out)


if __name__ == "__main__":
    main()
