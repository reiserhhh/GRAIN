#!/usr/bin/env python3
"""SOURCE-qualified ownership/shared response on actual public authors,one finite run."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_pair_criterion as A
from grain_research.construction import math_behavior_predictive as B
from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_item_bridge_mps as CPU
from grain_research.construction import math_person_bound_response as M
from grain_research.construction import math_source_anchor_exchange as E
from grain_research.construction import math_source_event_coupling as P
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_observer_operation_hm280 as N
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts.score_grain_author_belief_cold import load_bundle

CONFIG = "configs/grain_person_bound_shared_response_hm287.json"
SPEC = R.read(ROOT / CONFIG)
DEFAULT = ROOT / SPEC["output"]
TARGETS = ("E", "N", "A", "C", "O")
LAWS = ("observed_pair_F", "author_product_J", "F_minus_J")
COMPONENTS = ("total", "composition", "within_configuration")
SCIENCE = (
    CONFIG,
    "docs/GRAIN_PERSON_BOUND_SHARED_RESPONSE_PLAN_HM287.md",
    "grain_research/construction/math_person_bound_response.py",
    "scripts/run_grain_person_bound_response_hm287.py",
    "scripts/score_grain_person_bound_response_cold.py",
    "tests/test_grain_person_bound_response.py",
    "grain_research/construction/math_item_bridge_mps.py",
    "grain_research/construction/math_author_pair_criterion.py",
    "grain_research/construction/math_source_event_coupling.py",
    "grain_research/construction/math_criterion_components.py",
    "grain_research/construction/math_behavior_predictive.py",
    "grain_research/construction/math_source_anchor_exchange.py",
    "scripts/run_grain_observer_operation_hm280.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "grain_research/vector_mechanisms/trace.py",
)


def prepare(out):
    """Preserve old tokens and compile source-relative LEVELS before new outputs."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No overwrite or scientific retry")
    material, observer, source, criterion = (
        ROOT / SPEC[k + "_parent"]
        for k in ("material", "observer", "source", "criterion")
    )
    R.check(material, ("collected_seal.json", "analyzed_seal.json"))
    om = R.check(
        observer,
        ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
    )
    R.check(source, ("computed_seal.json", "verification_seal.json"))
    R.check(
        criterion,
        ("computed_seal.json", "analyzed_seal.json", "verification_seal.json"),
    )
    tok = AutoTokenizer.from_pretrained(str(J.MODEL), local_files_only=True)

    def encode(text):
        """Encode new design markers locally;original comment tokens are never reencoded."""
        return tok.encode(text, add_special_tokens=False)

    cases, bank, bindings = (
        R.read(material / f) for f in ("cases.json", "bank.json", "event_bindings.json")
    )
    old_head = B.observation_parts(tok, [0])[0][:-1]
    groups = []
    for u, case in enumerate(cases):
        if case["unit_id"] != u or bindings[u]["unit_id"] != u:
            raise ValueError("Old actual-case address mismatch")
        singleton = case["prefixes"][2:4]
        if any(p[: len(old_head)] != old_head for p in singleton):
            raise ValueError("Original public singleton token header changed")
        parts = [p[len(old_head) :] for p in singleton]
        if [len(p) for p in parts] != bindings[u]["token_lengths"]:
            raise ValueError("Original complete comment blocks changed")
        groups.append(parts)
    if len(groups) != 48 or len(bank) != 50:
        raise ValueError("All48 pairs/all50 complete items required")
    partner = np.array([2 * ((u + 1) % 24) + w for u in range(24) for w in (0, 1)])
    ids = tok.apply_chat_template(
        [
            {"role": "system", "content": M.SYSTEM},
            {"role": "user", "content": "Writing evidence:\n"},
        ],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    end = tok.convert_tokens_to_ids("<|im_end|>")
    pos = max(i for i, value in enumerate(ids) if value == end)
    start, tail = ids[:pos], ids[pos:] + encode("<|im_start|>assistant\n")
    alphabet = [encode(str(i)) for i in (0, 1)]
    if any(len(a) != 1 for a in alphabet) or alphabet[0] == alphabet[1]:
        raise ValueError("Original supported binary answer alphabet required")
    rows, ledger, lengths = [], [], []
    for unit, a in enumerate(groups):
        b = groups[partner[unit]]
        if bindings[unit]["author"] == bindings[partner[unit]]["author"]:
            raise ValueError("Self dyad not allowed")
        ledger.append(
            {
                "case_id": f"HM287-{unit}",
                "writer_id": f"panel-{unit // 2}",
                "partner_id": f"panel-{partner[unit] // 2}",
                "window": unit % 2,
                "writer_comments": [str(x) for x in bindings[unit]["comment_ids"]],
                "partner_comments": [
                    str(x) for x in bindings[partner[unit]]["comment_ids"]
                ],
            }
        )
        for queried in (-1, 1):
            bags = []
            for ownership in (-1, 1):
                for display in (-1, 1):
                    body, spans = M.compose_tokens((a, b), ownership, display, encode)
                    for span in spans:
                        if (
                            body[span["start"] : span["end"]]
                            != (a, b)[span["actor"]][span["event"]]
                        ):
                            raise ValueError("Unchanged comment failed token readback")
                    sequences = [
                        start
                        + body
                        + encode(
                            "\nEvidence ends here.\n\nQuestion: "
                            + M.question(item["public_text"], queried)
                        )
                        + tail
                        for item in bank
                    ]
                    bags.append(sequences[0])
                    lengths.extend(map(len, sequences))
                    rows.append(
                        {
                            "unit": unit,
                            "ownership": ownership,
                            "queried": queried,
                            "display": display,
                            "input_ids": sequences,
                        }
                    )
            M.assert_same_bag(bags)
    if (
        len(rows) != SPEC["families"]
        or len(rows) * 50 != SPEC["inquiries"]
        or max(lengths) > SPEC["max_input_tokens"]
    ):
        raise ValueError("Full fixed support exceeds cap;no clipping/subsetting")
    if R.digest(J.MODEL / "model.safetensors") != om["model_weights_sha256"]:
        raise ValueError("Pinned local model changed")
    sb = load_bundle(source / "source_pair_bank.npz")
    points = load_bundle(material / "numeric_pairs.npz")["points"]
    old = load_bundle(criterion / "source.npz")
    if not np.array_equal(P.BITS[sb["codes"]], old["labels"]) or not np.array_equal(
        sb["folds"], old["banks"]
    ):
        raise ValueError("SOURCE labels/banks are not identical")
    x = np.stack((points, points[:, ::-1]), 1).reshape(96, 2, 1024)
    weights = np.zeros((2, 3, 2, 48, 386))
    with threadpool_limits(limits=1):
        for sp in range(2):
            xx, zz = (
                (x, sb["prototypes"])
                if sp == 0
                else (x @ sb["projection"], sb["prototypes"] @ sb["projection"])
            )
            logs = P.event_logs(xx, zz, float(sb["variance"]))
            for ref, mask in enumerate(
                (sb["folds"] == 0, sb["folds"] == 1, np.ones(386, bool))
            ):
                w = A.source_author_weights(logs[:, mask], sb["relink"][mask])[:, :2]
                weights[sp, ref][:, :, mask] = (
                    w.reshape(48, 2, 2, -1).mean(1).transpose(1, 0, 2)
                )
    delta_error = float(
        abs(weights[:, :, 0] - weights[:, :, 1] - old["weights"].mean(3)).max()
    )
    if delta_error > 1e-12:
        raise ArithmeticError("Existing F-J source field changed")
    relative = M.actor_contrasts(weights, partner)
    relative = np.concatenate(
        (relative, (relative[:, :, 0] - relative[:, :, 1])[:, :, None]), 2
    )
    components = np.einsum(
        "sblui,ciq->sblcuq",
        relative,
        np.stack((old["labels"], old["means"], old["labels"] - old["means"])),
    )
    paths = [p / "manifest.json" for p in (material, observer, source, criterion)]
    for root, names in (
        (
            material,
            (
                "cases.json",
                "bank.json",
                "event_bindings.json",
                "numeric_pairs.npz",
                "collected_seal.json",
                "analyzed_seal.json",
            ),
        ),
        (
            source,
            ("source_pair_bank.npz", "computed_seal.json", "verification_seal.json"),
        ),
        (
            criterion,
            (
                "source.npz",
                "computed_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
            ),
        ),
        (observer, ("collected_seal.json", "verification_seal.json")),
        (
            J.MODEL,
            (
                "model.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
        ),
    ):
        paths.extend(root / p for p in names)
    (out / "batches").mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "config": SPEC,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in paths},
            "alphabet": [a[0] for a in alphabet],
            "model_weights_sha256": om["model_weights_sha256"],
            "new_contract_not_equivalent_old_anonymous_reader": True,
            "original_scope_complete": False,
        },
    )
    R.write(out / "units.json", rows)
    R.write(out / "bank.json", bank)
    R.write(out / "cases.json", ledger)
    R.archive(
        out / "source.npz",
        weights=weights,
        relative_weights=relative,
        components=components,
        labels=old["labels"],
        means=old["means"],
        groups=old["groups"],
        assignments=old["assignments"],
        partner=partner,
    )
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "related_dyads": 48,
            "unique_comments": 96,
            "families": len(rows),
            "inquiries": len(rows) * 50,
            "input_token_range": [min(lengths), max(lengths)],
            "old_F_J_error": delta_error,
            "source_cycle_balance_error": float(abs(relative.sum(-2)).max()),
            "application_labels_API_private_training": 0,
            "whole_bag_verified": True,
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
    """Reuse the native collector and CPU64 export with this frozen new design."""
    N.I.native_items = CPU.native_items
    N.collect(out)


def observed(out):
    """Bind eight actual role/order observations;missing endpoints cannot be fabricated."""
    from scipy.special import logsumexp

    values, head_error = np.empty((48, 2, 2, 2, 50, 2)), 0.0
    seen = set()
    head = load_bundle(out / "head.npz")
    alphabet = R.read(out / "manifest.json")["alphabet"]
    diagnostics = []
    for i, row in enumerate(R.read(out / "units.json")):
        key = (
            row["unit"],
            (row["ownership"] + 1) // 2,
            (row["queried"] + 1) // 2,
            (row["display"] + 1) // 2,
        )
        if key in seen:
            raise ValueError("Duplicate observed role corner")
        seen.add(key)
        z = load_bundle(out / "batches" / f"{i:03d}.npz")
        values[key] = z["logits"]
        mass = logsumexp(z["logits"], -1) - z["normalizer"]
        if not np.isfinite(mass).all() or np.max(mass) > 1e-6:
            raise ArithmeticError(
                "Observed binary mass is not a full-vocabulary subset"
            )
        diagnostics.extend(
            {
                "family": i,
                "item": j,
                "unit": row["unit"],
                "ownership": row["ownership"],
                "queried": row["queried"],
                "display": row["display"],
                "binary_log_mass": float(mass[j]),
                "greedy_id": int(z["top"][j]),
                "greedy_is_zero": bool(z["top"][j] == alphabet[0]),
                "greedy_is_one": bool(z["top"][j] == alphabet[1]),
            }
            for j in range(50)
        )
        head_error = max(
            head_error,
            float(
                abs(
                    z["states"] @ head["weights"].astype(float).T
                    + head["bias"]
                    - z["logits"]
                ).max()
            ),
        )
    if seen != set(np.ndindex(48, 2, 2, 2)):
        raise ValueError("Incomplete new response cube")
    return values, head_error, diagnostics


def analyze(out):
    """Qualify source LEVEL/owner-bound shared response;fit no model or target score."""
    R.check(out, ("collected_seal.json",))
    z, head_error, diagnostics = observed(out)
    if head_error > SPEC["native_tolerance"]:
        raise ArithmeticError("Old adequate head tolerance fails,no precision rescue")
    source, bank = load_bundle(out / "source.npz"), R.read(out / "bank.json")
    reader, arrays, rows = M.readings(z, bank), {}, []
    fields = source["components"]
    for scale, response in reader.items():
        arrays[scale + "/items"] = response["items"]
        for key in ("balanced_key", "common_support"):
            contact = np.einsum("sblcuq,umr->sblcmqr", fields, response[key]) / 48
            arrays[scale + "/" + key] = contact
            for sp, space in enumerate(("E5", "GRAIN15")):
                for ref, reference in enumerate(("bank0", "bank1", "pooled")):
                    for law, lawname in enumerate(LAWS):
                        for component, componentname in enumerate(COMPONENTS):
                            for mask in range(8):
                                for q, target in enumerate(TARGETS):
                                    for r, query in enumerate(TARGETS):
                                        rows.append(
                                            {
                                                "space": space,
                                                "reference": reference,
                                                "law": lawname,
                                                "component": componentname,
                                                "scale": scale,
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
        arrays[scale + "/global_common"] = arrays[scale + "/common_support"].mean(-1)
    gamma = reader["logodds"]["global_common"][:, 3]
    alpha = source["relative_weights"][1, 2, 0].T @ gamma / 48
    e = source["labels"] - source["means"]
    primary, nulls, replay = [], [], 0.0
    for q, target in enumerate(TARGETS):
        value = float(alpha @ e[:, q])
        expected = arrays["logodds/global_common"][1, 2, 0, 2, 3, q]
        replay = max(replay, abs(value - expected))
        null = e[source["assignments"][q], q] @ alpha
        variance = float(
            alpha
            @ C.orbit_covariance(source["labels"][:, q], source["groups"][:, q])
            @ alpha
        )
        primary.append(
            {
                "target": target,
                "source_F_person_bound_common": value,
                "exact_conditional_mean": 0.0,
                "exact_conditional_variance": variance,
                "two_sided_conditional_rank": float(
                    (1 + np.sum(abs(null) >= abs(value))) / 200
                ),
            }
        )
        nulls.append(null)
    for row, adjusted in zip(
        primary, E.holm([r["two_sided_conditional_rank"] for r in primary]), strict=True
    ):
        row["Holm5_rank"] = float(adjusted)
    record = {
        "version": "HM287-v1",
        "cases": R.read(out / "cases.json"),
        "source_components": fields[1, 2].tolist(),
        "item_logits": z.tolist(),
        "item_bank": [{k: b[k] for k in ("item", "target", "pole")} for b in bank],
    }
    R.archive(
        out / "readings.npz", **arrays, alpha=alpha, nulls=np.array(nulls), logits=z
    )
    R.table(out / "all_contacts.csv", rows)
    R.table(out / "reader_diagnostics.csv", diagnostics)
    R.table(out / "primary.csv", primary)
    R.write(out / "cold_record.json", record)
    R.write(
        out / "summary.json",
        {
            "primary": primary,
            "source_pullback_error": replay,
            "head_error": head_error,
            "global_common_owner_RMS": float(np.sqrt(np.mean(gamma**2))),
            "global_common_position_owner_RMS": float(
                np.sqrt(np.mean(reader["logodds"]["global_common"][:, 7] ** 2))
            ),
            "global_common_unqualified_RMS": float(
                np.sqrt(np.mean(reader["logodds"]["global_common"][:, 0] ** 2))
            ),
            "global_common_all8_character_RMS": np.sqrt(
                np.mean(reader["logodds"]["global_common"] ** 2, axis=0)
            ).tolist(),
            "greedy_zero_count": sum(d["greedy_is_zero"] for d in diagnostics),
            "greedy_one_count": sum(d["greedy_is_one"] for d in diagnostics),
            "mean_conditional_yes_probability": float(
                np.mean(1 / (1 + np.exp(z[..., 0] - z[..., 1])))
            ),
            "native_runtime": R.read(out / "collected_runtime.json"),
            "actor_bound_response_not_human_behavior": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "readings.npz",
            "all_contacts.csv",
            "reader_diagnostics.csv",
            "primary.csv",
            "cold_record.json",
            "summary.json",
        ),
    )
    print(json.dumps(R.read(out / "summary.json")))


def verify(out):
    """Literal role/source/key sums and cold readback,not a precision campaign."""
    R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    source = load_bundle(out / "source.npz")
    data = load_bundle(out / "readings.npz")
    bank = R.read(out / "bank.json")
    summary = R.read(out / "summary.json")
    raw = data["logits"][..., 1] - data["logits"][..., 0]
    literal = np.zeros((48, 8, 50))
    for mask in range(8):
        for corner in np.ndindex(2, 2, 2):
            signs = [2 * j - 1 for j in corner]
            weight = np.prod([signs[k] for k in range(3) if mask & (1 << k)])
            literal[:, mask] += weight * raw[(slice(None), *corner)] / 8
    errors = {"literal_characters": float(abs(literal - data["logodds/items"]).max())}
    replay = np.einsum("umj,mc->ucj", literal, M.CHARACTERS)
    errors["all_corner_reconstruction"] = float(
        abs(replay - raw.reshape(48, 8, 50)).max()
    )
    common, keyed = np.empty((48, 8, 5)), np.empty((48, 8, 5))
    for q in range(5):
        hi = [b["item"] for b in bank if b["target"] == q and b["pole"] == 1]
        lo = [b["item"] for b in bank if b["target"] == q and b["pole"] == -1]
        high = sum(literal[:, :, j] for j in hi) / len(hi)
        low = sum(literal[:, :, j] for j in lo) / len(lo)
        common[:, :, q], keyed[:, :, q] = (high + low) / 2, (high - low) / 2
    contacts = {}
    for key, value in (("common_support", common), ("balanced_key", keyed)):
        direct = (
            sum(
                source["components"][..., u, :, None, None]
                * value[u][None, None, None, None, None, :, :]
                for u in range(48)
            )
            / 48
        )
        direct = direct.transpose(0, 1, 2, 3, 5, 4, 6)
        errors[key + "/literal_contacts"] = float(
            abs(direct - data["logodds/" + key]).max()
        )
        contacts[key] = direct
    gamma = common[:, 3].mean(-1)
    errors["literal_references"] = 0.0
    errors["exact_orbit_variance"] = 0.0
    with threadpool_limits(limits=1):
        for q in range(5):
            residual = source["labels"][:, q] - source["means"][:, q]
            all_assignments = np.vstack((np.arange(386), source["assignments"][q]))
            saved = np.r_[
                summary["primary"][q]["source_F_person_bound_common"], data["nulls"][q]
            ]
            for assignment, value in zip(all_assignments, saved, strict=True):
                weights = source["relative_weights"][1, 2, 0]
                readout = np.array([sum(w * residual[assignment]) for w in weights])
                result = sum(readout * gamma) / 48
                errors["literal_references"] = max(
                    errors["literal_references"], abs(result - value)
                )
                if not np.array_equal(
                    source["groups"][assignment, q], source["groups"][:, q]
                ):
                    raise ArithmeticError("Old SOURCE condition assignment changed")
            variance = 0.0
            for group in np.unique(source["groups"][:, q]):
                ix = source["groups"][:, q] == group
                y, a = source["labels"][ix, q], data["alpha"][ix]
                m, p = len(y), y.mean()
                if m > 1:
                    variance += (
                        p * (1 - p) * m / (m - 1) * (sum(a**2) - sum(a) ** 2 / m)
                    )
                if abs(residual[ix].mean()) > 1e-12:
                    raise ArithmeticError("SOURCE residual expectation is not zero")
            errors["exact_orbit_variance"] = max(
                errors["exact_orbit_variance"],
                abs(variance - summary["primary"][q]["exact_conditional_variance"]),
            )
    for scale in ("logodds", "probability"):
        for key in ("common_support", "balanced_key"):
            a = data[scale + "/" + key]
            errors[scale + "/" + key + "/level_action"] = float(
                abs(a[:, :, 0] - a[:, :, 1] - a[:, :, 2]).max()
            )
            errors[scale + "/" + key + "/components"] = float(
                abs(a[:, :, :, 0] - a[:, :, :, 1] - a[:, :, :, 2]).max()
            )
    errors["source_cycle_balance"] = float(
        abs(source["relative_weights"].sum(-2)).max()
    )
    reply = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_person_bound_response_cold.py"),
            "--input",
            str(out / "cold_record.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cold = json.loads(reply.stdout)
    for scale in ("logodds", "probability"):
        for name, key in (
            ("keyed_contacts", "balanced_key"),
            ("common_contacts", "common_support"),
            ("global_common_contacts", "global_common"),
        ):
            errors[scale + "/cold/" + name] = float(
                abs(np.asarray(cold[scale][name]) - data[scale + "/" + key][1, 2]).max()
            )
    if max(errors.values()) > 1e-10:
        raise ArithmeticError(
            "Registered account verification failed;preserve outcomes"
        )
    R.write(out / "cold_readback.json", cold)
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "errors": errors,
            "literal_primary_and_references": 1000,
            "same_agent_not_blind": True,
            "no_new_scientific_collection": True,
        },
    )
    R.seal(out, "verification_seal.json", ("cold_readback.json", "verification.json"))
    print(json.dumps(R.read(out / "verification.json")))


def main():
    """Run immutable stages;never renew a failed collection cap."""
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
        print("HM287 method/input/collection/analysis seals PASS")
    else:
        {"prepare": prepare, "collect": collect, "analyze": analyze, "verify": verify}[
            args.stage
        ](args.out)


if __name__ == "__main__":
    main()
