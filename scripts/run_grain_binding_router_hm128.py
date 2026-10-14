#!/usr/bin/env python3
"""Identify a neutral binding operation, then test frozen behavioural routing."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_binding_router as M
from grain_research.construction.math_construct_routing import query
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import (
    MODEL,
    digest,
    read_json,
    write_json,
)
from scripts.run_grain_author_evidence_merge_hm102 import (
    seals,
    table,
    verify_hashes,
    write,
)

SPEC = ROOT / "configs/grain_binding_router_hm128.json"
DEFAULT = ROOT / "results/grain_math_response_v1/binding_router_r1"
CODE = (
    "configs/grain_binding_router_hm128.json",
    "docs/GRAIN_BINDING_ROUTER_PLAN_HM128.md",
    "docs/GRAIN_HM128_TOKEN_PREFLIGHT.md",
    "grain_research/construction/math_binding_router.py",
    "scripts/run_grain_binding_router_hm128.py",
    "scripts/score_grain_binding_router_cold.py",
    "tests/test_grain_binding_router.py",
    "grain_research/vector_mechanisms/trace.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/navigation.py",
    "grain_research/construction/math_construct_routing.py",
    "scripts/run_grain_author_evidence_merge_hm102.py",
    "scripts/run_grain_upstream_routes.py",
)
SYSTEM = "Use only the supplied account as evidence about the named people. Answer the question without explanation."


def save_arrays(path, **arrays):
    """Commit a new numerical artifact, never replace a failed or existing one."""
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def address(tokenizer, body, question, spans, labels):
    """Locate controlled source spans in actual chat tokens, not future suffixes."""
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": body + "\n\nQuestion: " + question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if text.count(body) != 1 or len(encoded["input_ids"]) > 768:
        raise ValueError("Unambiguous full controlled body required")
    start, positions = text.index(body), []
    for lo, hi in spans:
        candidates = [
            i
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            if a < start + hi and b > start + lo
        ]
        if not candidates:
            raise ValueError("Controlled span has no token endpoint")
        positions.append(candidates[-1])
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(i) != 1 for i in ids):
        raise ValueError("Fixed candidate labels must be single tokens")
    return {
        "input_ids": encoded["input_ids"],
        "positions": positions,
        "alphabet": {
            "labels": labels,
            "token_ids": [i[0] for i in ids],
            "values": np.linspace(0, 1, len(ids)).tolist(),
        },
    }


def factual(names, countries, wording):
    """Render new world facts and record name/country endpoints separately."""
    pieces = [f"{n} lives in {c}." for n, c in zip(names, countries, strict=True)]
    body = (
        " ".join(pieces)
        if wording == 0
        else "Records of residence: " + " ".join(pieces)
    )
    spans = []
    for values in (names, countries):
        for value in values:
            lo = body.index(value)
            spans.append([lo, lo + len(value)])
    return body, spans


def prepare(out):
    """Seal the whole finite study before any source gradient or application score."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve previous, incomplete and failed packets")
    spec = read_json(SPEC)
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    source, applications = [], []
    for pair, names_reverse, countries_reverse in itertools.product(
        spec["source_pairs"], range(2), range(2)
    ):
        names, countries = (
            pair[:2][:: (-1 if names_reverse else 1)],
            pair[2:][:: (-1 if countries_reverse else 1)],
        )
        body, spans = factual(names, countries, 0)
        source.append(
            {
                "id": len(source),
                "body": body,
                "names": names,
                "countries": countries,
                "prompts": [
                    address(
                        tok,
                        body,
                        f"Which country does {name} live in? Return only the country name.",
                        spans,
                        countries,
                    )
                    for name in names
                ],
            }
        )
    names = spec["application_names"]
    for countries, wording in itertools.product(
        spec["application_countries"], range(2)
    ):
        body, spans = factual(names, countries, wording)
        applications.append(
            {
                "id": len(applications),
                "kind": "fact",
                "body": body,
                "names": names,
                "countries": countries,
                "wording": wording,
                "prompts": [
                    address(
                        tok,
                        body,
                        f"Which country does {name} live in? Return only the country name.",
                        spans,
                        countries,
                    )
                    for name in names
                ],
            }
        )
    for family, wording, assignment in itertools.product(range(6), range(2), range(2)):
        low, high = spec["behaviours"][family]
        clauses = [high, low] if assignment else [low, high]
        opening = (
            "During a voluntary workshop, each person could choose how to respond. "
            if wording == 0
            else "Notes from a voluntary meeting where everyone chose their own response: "
        )
        body = (
            opening
            + f"{names[0]} {clauses[0]}. {names[1]} {clauses[1]}. {names[2]} wrote down the time."
        )
        spans = [[body.index(name), body.index(name) + len(name)] for name in names]
        spans += [
            [body.index(clause), body.index(clause) + len(clause)] for clause in clauses
        ]
        prompts = [
            address(
                tok,
                body,
                query(target).replace("writer's", name + "'s"),
                spans,
                list(map(str, range(10))),
            )
            for name in names
            for target in spec["targets"]
        ]
        applications.append(
            {
                "id": len(applications),
                "kind": "behaviour",
                "body": body,
                "names": names,
                "family": family,
                "wording": wording,
                "assignment": assignment,
                "prompts": prompts,
            }
        )
    for row in applications:
        if row["kind"] == "behaviour":
            other = next(
                v
                for v in applications
                if v["kind"] == "behaviour"
                and v["family"] == row["family"]
                and v["wording"] == row["wording"]
                and v["assignment"] != row["assignment"]
            )
            if Counter(row["body"].split()) != Counter(other["body"].split()):
                raise ValueError(
                    "Natural assignment control must have the same word bag"
                )
    if len(source) != 8 or len(applications) != 28:
        raise ValueError("Whole source/application design required")
    (out / "00").mkdir(parents=True)
    (out / "01").mkdir()
    (out / "02").mkdir()
    write(out / "00/source.json", source)
    write(out / "00/application.json", applications)
    model_files = [
        p for p in MODEL.iterdir() if p.is_file() and not p.name.startswith("._")
    ]
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {name: digest(ROOT / name) for name in CODE},
            "model": {p.name: digest(p) for p in model_files},
            "inputs": seals(out, out / "00"),
            "source_questions": 16,
            "application_questions": 444,
            "application_endpoints_all_arms": 1776,
            "prior_turn": "NO_NEW_MECHANISM_EXECUTION; reassessment requires changed scientific question",
            "original_scope_complete": False,
        },
    )
    write_json(
        out / "status.json",
        {
            "state": "PREPARED",
            "seconds": 0.0,
            "forwards": 0,
            "pending": None,
            "completed_applications": 0,
        },
    )
    print(
        "PREPARED HM128:8 neutral source worlds,28 new application worlds,4 fixed arms",
        flush=True,
    )


def check(out):
    """Read authoritative bytes, not a previous completion headline."""
    m = read_json(out / "00/manifest.json")
    verify_hashes(ROOT, m["code"])
    verify_hashes(out, m["inputs"])
    return m


class Execution:
    """Charge actual model and derivative work to one nonrenewable budget."""

    def __init__(self, out, manifest):
        self.out, self.spec = out, manifest["spec"]
        self.status = read_json(out / "status.json")
        if self.status["state"] != "PREPARED" or self.status["pending"] is not None:
            raise RuntimeError("No terminal or uncertain-run restart")
        self.start = time.monotonic()
        self.status["state"] = "RUNNING"
        self.observer = None

    def checkpoint(self):
        """Save current cost and reject exhausted execution without starting a new call."""
        self.status["seconds"] = time.monotonic() - self.start
        write_json(self.out / "status.json", self.status)
        if self.status["seconds"] >= self.spec["max_seconds"]:
            raise TimeoutError("Original cumulative HM128 cap exhausted")

    def forward(self, prompts, modifier=None, capture=False):
        """Use actual padding/endpoints and a specified block-output intervention."""
        import torch

        self.checkpoint()
        if self.status["forwards"] >= self.spec["max_forwards"]:
            raise TimeoutError("Original forward cap exhausted")
        self.status["forwards"] += 1
        write_json(self.out / "status.json", self.status)
        observer, device = self.observer, self.spec["device"]
        lengths = [len(p["input_ids"]) for p in prompts]
        ids = torch.zeros((len(prompts), max(lengths)), dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for b, p in enumerate(prompts):
            ids[b, : lengths[b]] = torch.tensor(p["input_ids"], device=device)
            mask[b, p.get("left_padding", 0) : lengths[b]] = 1
        saved = []

        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if capture:
                saved.extend(
                    hidden[b, p["positions"]].detach().float().cpu().numpy().copy()
                    for b, p in enumerate(prompts)
                )
            if modifier is not None:
                hidden = modifier(hidden, prompts)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        handle = observer.model.model.layers[self.spec["layer"]].register_forward_hook(
            hook
        )
        try:
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            result = observer.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
            # Right padding requires each true endpoint, not the batch end.
            # logits_to_keep=1 cannot select earlier endpoints; lengths must match.
            if len(set(lengths)) != 1:
                raise ValueError("This execution requires a fixed common true endpoint")
            logits = result.logits[:, -1]
        finally:
            handle.remove()
        return logits, saved


def pad_prompts(prompts, pad_id):
    """Left pad only, retaining all real tokens and relocating source span addresses."""
    maximum = max(len(p["input_ids"]) for p in prompts)
    # Left padding is masked in forward; it must not become accessible evidence.
    return [
        {
            **p,
            "input_ids": [pad_id] * (maximum - len(p["input_ids"])) + p["input_ids"],
            "positions": [i + maximum - len(p["input_ids"]) for i in p["positions"]],
            "left_padding": maximum - len(p["input_ids"]),
        }
        for p in prompts
    ]


def source_function(execution, prompts, delta_e, delta_a):
    """Return a factual retrieval functional with source midpoint and paired injections."""
    import torch

    def function(x, y):
        def modify(hidden, rows):
            addition = torch.zeros_like(hidden)
            for b, p in enumerate(rows):
                e0, e1, a0, a1 = p["positions"]
                addition[b, e0] = 0.5 * delta_e + x
                addition[b, e1] = -0.5 * delta_e - x
                addition[b, a0] = 0.5 * delta_a + y
                addition[b, a1] = -0.5 * delta_a - y
            return hidden + addition

        logits, _ = execution.forward(prompts, modifier=modify)
        probabilities = [
            torch.softmax(logits[b, p["alphabet"]["token_ids"]], dim=-1)[p["correct"]]
            for b, p in enumerate(prompts)
        ]
        return torch.stack(probabilities).mean()

    return function


def collect_source(execution, source):
    """Identify an operator from neutral facts alone, before behavioural readout."""
    import torch

    spec, out = execution.spec, execution.out
    batches, states, original = [], [], []
    count = spec["source_batch_worlds"]
    for start in range(0, len(source), count):
        prompts = [
            {**p, "correct": i}
            for row in source[start : start + count]
            for i, p in enumerate(row["prompts"])
        ]
        prompts = pad_prompts(prompts, execution.observer.tokenizer.pad_token_id)
        execution.status["pending"] = f"source capture {start}"
        with torch.no_grad():
            logits, captured = execution.forward(prompts, capture=True)
        states.extend(captured)
        original.extend(
            float(
                torch.softmax(logits[b, p["alphabet"]["token_ids"]], -1)[
                    p["correct"]
                ].cpu()
            )
            for b, p in enumerate(prompts)
        )
        batches.append(prompts)
        save_arrays(
            out / f"01/source_capture_{start}.npz",
            states=np.stack(captured),
            candidate_logits=np.stack(
                [
                    logits[b, p["alphabet"]["token_ids"]].cpu().numpy()
                    for b, p in enumerate(prompts)
                ]
            ),
        )
    states = np.stack(states)
    delta_e, delta_a = (
        (states[:, 1] - states[:, 0]).mean(0),
        (states[:, 3] - states[:, 2]).mean(0),
    )
    dimension = len(delta_e)
    omega = M.gaussian_basis(dimension, spec["sketch_width"], spec["seed"])
    de, da = (
        torch.tensor(v, device=spec["device"], dtype=torch.float32)
        for v in (delta_e, delta_a)
    )
    products, midpoints = [], []
    for index, prompts in enumerate(batches):
        execution.status["pending"] = f"source H Omega {index}"
        value, product = M.mixed_product(
            source_function(execution, prompts, de, da),
            dimension,
            omega,
            device=spec["device"],
            checkpoint=execution.checkpoint,
        )
        products.append(product)
        midpoints.append(value)
        save_arrays(
            out / f"01/h_omega_{index}.npz", product=product, midpoint=np.array(value)
        )
        print("SOURCE H Omega", index, value, flush=True)
    y = np.mean(products, axis=0)
    q = np.linalg.qr(y)[0]
    transposes = []
    for index, prompts in enumerate(batches):
        execution.status["pending"] = f"source H transpose Q {index}"
        _, product = M.mixed_product(
            source_function(execution, prompts, de, da),
            dimension,
            q,
            transpose=True,
            device=spec["device"],
            checkpoint=execution.checkpoint,
        )
        transposes.append(product)
        save_arrays(out / f"01/ht_q_{index}.npz", product=product)
    t = np.mean(transposes, axis=0)
    u, singular, v = M.compressed_svd(y, t, spec["rank"])
    random = M.gaussian_basis(dimension, spec["rank"], spec["seed"] + 1)
    transpose_error = float(np.max(abs(q.T @ y - t.T @ omega)))
    save_arrays(
        out / "01/operator.npz",
        entity_basis=u,
        attribute_basis=v,
        singular_values=singular,
        random_basis=random,
        delta_entity=delta_e,
        delta_attribute=delta_a,
        h_omega=y,
        ht_q=t,
        omega=omega,
        q=q,
    )
    write(
        out / "01/source_summary.json",
        {
            "original_correct_probability_mean": float(np.mean(original)),
            "original_correct_probabilities": original,
            "midpoint_correct_probability_mean": float(np.mean(midpoints)),
            "midpoint_not_assumed_chance": midpoints,
            "transpose_product_identity_max_error": transpose_error,
            "singular_values_compressed": singular.tolist(),
            "dimension": dimension,
            "sketch_width": spec["sketch_width"],
            "rank": spec["rank"],
            "entity_orthogonality_error": float(
                np.max(abs(u.T @ u - np.eye(spec["rank"])))
            ),
            "source_kind": "neutral_residence_only",
            "application_answers_used": False,
        },
    )
    write(out / "01/source_seal.json", seals(out, out / "01"))
    execution.status["pending"] = None
    execution.checkpoint()
    return u, random


def apply(execution, applications, basis, random):
    """Read all held-out persons/targets under fixed source-only operations."""
    import torch

    out = execution.out
    for row in applications:
        prompts = pad_prompts(row["prompts"], execution.observer.tokenizer.pad_token_id)
        for arm in execution.spec["arms"]:
            execution.status["pending"] = f"application {row['id']} {arm}"
            sizes = []

            def modify(hidden, addressed, arm=arm, sizes=sizes):
                additions = torch.zeros_like(hidden)
                for b, p in enumerate(addressed):
                    p0, p1 = p["positions"][:2]
                    left, right = (
                        hidden[b, p0].detach().cpu().numpy(),
                        hidden[b, p1].detach().cpu().numpy(),
                    )
                    delta = M.swap_delta(left, right, basis)
                    if arm == "random_matched":
                        delta = M.swap_delta(
                            left, right, random, match_norm=np.linalg.norm(delta)
                        )
                    elif arm == "full_swap":
                        delta = right - left
                    sizes.append(float(np.linalg.norm(delta)))
                    additions[b, p0] = torch.tensor(
                        delta, dtype=hidden.dtype, device=hidden.device
                    )
                    additions[b, p1] = -additions[b, p0]
                return hidden + additions

            with torch.no_grad():
                logits, slots = execution.forward(
                    prompts,
                    modifier=None if arm == "baseline" else modify,
                    capture=arm == "baseline",
                )
            readings = [
                native_readout(logits[b].float().cpu().numpy(), p["alphabet"])
                for b, p in enumerate(prompts)
            ]
            path = out / f"02/a{row['id']:02d}_{arm}.json"
            write(
                path,
                {
                    "row": row["id"],
                    "kind": row["kind"],
                    "arm": arm,
                    "readings": readings,
                    "edit_norms": sizes,
                    "family_not_used_by_intervention": True,
                },
            )
            if slots:
                save_arrays(
                    out / f"02/a{row['id']:02d}_slots.npz", slots=np.stack(slots)
                )
        execution.status["completed_applications"] += 1
        execution.status["pending"] = None
        execution.checkpoint()
        print("APPLICATION", row["id"], row["kind"], flush=True)
    write(out / "02/output_seal.json", seals(out, out / "02"))


def run(out):
    """Execute once; all failures and exhausted budgets remain terminal evidence."""
    from grain_research.vector_mechanisms.trace import DecoderObserver
    from scripts.run_grain_upstream_routes import collector_guard

    m = check(out)
    verify_hashes(MODEL, m["model"])
    execution = Execution(out, m)
    try:
        with collector_guard(out):
            execution.status["pending"] = "model load"
            execution.checkpoint()
            execution.observer = DecoderObserver.load(
                str(MODEL),
                device=m["spec"]["device"],
                dtype=m["spec"]["dtype"],
                max_tokens=768,
            )
            execution.observer.model.requires_grad_(False)
            write(out / "observer.json", execution.observer.manifest())
            execution.checkpoint()
            basis, random = collect_source(execution, read_json(out / "00/source.json"))
            apply(execution, read_json(out / "00/application.json"), basis, random)
            execution.status["state"] = "COLLECTED"
    except BaseException as error:
        execution.status["state"] = (
            "BUDGET_STOP" if isinstance(error, TimeoutError) else "FAILED"
        )
        execution.status["error"] = repr(error)
        raise
    finally:
        execution.status["seconds"] = time.monotonic() - execution.start
        write_json(out / "status.json", execution.status)


def analyze(out):
    """Read every signed causal effect; do not rescue a failed source operation."""
    m = check(out)
    status = read_json(out / "status.json")
    if status["state"] != "COLLECTED":
        raise RuntimeError("Only complete real collection can be analyzed")
    verify_hashes(out, read_json(out / "01/source_seal.json"))
    verify_hashes(out, read_json(out / "02/output_seal.json"))
    applications = read_json(out / "00/application.json")
    records = {
        (r["id"], arm): read_json(out / f"02/a{r['id']:02d}_{arm}.json")["readings"]
        for r in applications
        for arm in m["spec"]["arms"]
    }
    fact_rows, route_rows = [], []
    for row in applications:
        if row["kind"] == "fact":
            base = np.array(
                [r["conditional_probabilities"] for r in records[row["id"], "baseline"]]
            )
            for arm in m["spec"]["arms"]:
                changed = np.array(
                    [r["conditional_probabilities"] for r in records[row["id"], arm]]
                )
                fact_rows.append(
                    {
                        "row": row["id"],
                        "wording": row["wording"],
                        "arm": arm,
                        "original_correct": float(np.diag(changed).mean()),
                        "swapped_pair_probability": float(
                            (changed[0, 1] + changed[1, 0]) / 2
                        ),
                        "swapped_pair_gain": float(
                            (changed[0, 1] - base[0, 1] + changed[1, 0] - base[1, 0])
                            / 2
                        ),
                        "third_probability_l1": float(abs(changed[2] - base[2]).sum()),
                    }
                )
            continue
        other = next(
            r
            for r in applications
            if r["kind"] == "behaviour"
            and r["family"] == row["family"]
            and r["wording"] == row["wording"]
            and r["assignment"] != row["assignment"]
        )
        base = np.array(
            [r["conditional_score"] for r in records[row["id"], "baseline"]]
        ).reshape(3, 6)
        swapped = np.array(
            [r["conditional_score"] for r in records[other["id"], "baseline"]]
        ).reshape(3, 6)
        for arm in m["spec"]["arms"]:
            changed = np.array(
                [r["conditional_score"] for r in records[row["id"], arm]]
            ).reshape(3, 6)
            account = M.routing_accounts(base, changed, swapped)
            route_rows.append(
                {
                    "row": row["id"],
                    "family": row["family"],
                    "wording": row["wording"],
                    "assignment": row["assignment"],
                    "arm": arm,
                    **{k: v for k, v in account.items() if not isinstance(v, list)},
                    "matched_target_effect": account["person_difference_effect"][
                        row["family"]
                    ],
                    "matched_natural_effect": account["natural_assignment_effect"][
                        row["family"]
                    ],
                }
            )
    (out / "03").mkdir()
    table(out / "03/factual_interchange.csv", fact_rows)
    table(out / "03/behavioural_routing.csv", route_rows)
    summary = {"source": read_json(out / "01/source_summary.json"), "arms": {}}
    for arm in m["spec"]["arms"]:
        facts = [r for r in fact_rows if r["arm"] == arm]
        routes = [r for r in route_rows if r["arm"] == arm]
        summary["arms"][arm] = {
            key: float(np.mean([r[key] for r in rows]))
            for rows, keys in (
                (facts, ("swapped_pair_gain", "third_probability_l1")),
                (
                    routes,
                    (
                        "towards_natural_fraction",
                        "differential_norm",
                        "common_norm",
                        "untouched_norm",
                    ),
                ),
            )
            for key in keys
        }
        summary["arms"][arm]["matched_target_towards_count"] = sum(
            r["matched_target_effect"] * r["matched_natural_effect"] > 0 for r in routes
        )
        summary["arms"][arm]["behavioural_worlds"] = len(routes)
    summary.update(
        {
            "full_scope_complete": False,
            "psychological_truth": False,
            "source_answer_dependence": "neutral factual source calibration",
            "application_trait_answers_used_for_basis": False,
            "execution": status,
        }
    )
    write(out / "03/summary.json", summary)
    write(out / "03/seal.json", seals(out, out / "03"))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    """Run a declared phase without starting old queues or new agents."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.phase](args.output)


if __name__ == "__main__":
    main()
