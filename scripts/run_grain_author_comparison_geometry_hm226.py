#!/usr/bin/env python3
"""Observe one fixed inter-author graph, not a ranking/accuracy search."""

import argparse
import hashlib
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_comparison_geometry as M
from grain_research.construction import math_author_position_response as P
from grain_research.vector_mechanisms.direct_traits import BIG5
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts.run_grain_vector_mechanism_external_features import bundle_arms

CONFIG = "configs/grain_author_comparison_geometry_hm226.json"
DEFAULT = ROOT / "results/grain_math_response_v1/author_comparison_geometry_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_AUTHOR_COMPARISON_GEOMETRY_PLAN_HM226.md",
    "grain_research/construction/math_author_comparison_geometry.py",
    "scripts/run_grain_author_comparison_geometry_hm226.py",
    "tests/test_grain_author_comparison_geometry.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_vector_mechanism_external_features.py",
)
STAGES = ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
NAMES = (
    "comparison_position",
    "standalone_level",
    "source_full_position",
    "source_departure_position",
)


def finite(value):
    """Undefined contact remains missing, never a fabricated zero."""
    return float(value) if np.isfinite(value) else None


def prepare(out):
    """Freeze full histories and graph before new relation outputs or labels."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Existing or uncertain science cannot be replaced")
    cfg = R.read(ROOT / CONFIG)
    parent, positions = ROOT / cfg["parent"], ROOT / cfg["positions"]
    pm = R.check(parent, STAGES[:1] + ("analyzed_seal.json", "verification_seal.json"))
    R.check(
        positions,
        ("computed_seal.json", "analyzed_seal.json", "verification_seal.json"),
    )
    indices = H.selected(parent / "observations.npz", ("panel_indices",))[
        "panel_indices"
    ]
    records = H.selected(positions / "score_records.npz", ("objects",))["objects"]
    bundles = R.read(ROOT / cfg["material"])["bundles"]
    selections = R.read(parent / "selections.json")
    texts, binding = [], []
    for u, original in enumerate(indices):
        bundle = bundles[int(original)]
        if bundle["source_group"] != "big5_only":
            raise ValueError("Only previously authorized PUBLIC Big5-only material")
        ids = [c["comment_id"] for c in bundle["views"][2]]
        if ids != selections[u]["event_ids"] or len(set(ids)) != 12:
            raise ValueError("Changed SAME12-event author address")
        body = bundle_arms(bundle)["cue_mask"][24:36]
        texts.append(body)
        binding.append(
            {
                "author": u,
                "original_author_index": int(original),
                "event_ids": ids,
                "body_hashes": [hashlib.sha256(t.encode()).hexdigest() for t in body],
            }
        )
    if len(texts) != 24 or records.shape != (24, 10, 5):
        raise ValueError("Full24-author fixed record required")
    e = M.graph()
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    alphabet = [tok.encode(s, add_special_tokens=False) for s in ("A", "B")]
    if any(len(k) != 1 for k in alphabet) or alphabet[0] == alphabet[1]:
        raise ValueError("Distinct native single-token comparison alphabet required")
    units = []
    for edge, pair in enumerate(e.tolist() + [[u, u] for u in cfg["self_authors"]]):
        for order in range(2 if edge < len(e) else 1):
            left, right = pair if order == 0 else pair[::-1]
            for q, task in enumerate(BIG5):
                user = M.render(texts[left], texts[right], task)
                ids = tok.apply_chat_template(
                    [
                        {"role": "system", "content": M.SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if isinstance(ids, Mapping):
                    ids = ids["input_ids"]
                if (
                    not ids
                    or len(ids) + 16 > cfg["max_tokens"]
                    or any(not isinstance(i, int) for i in ids)
                ):
                    raise ValueError(
                        "Unsupported complete history; no truncation or cohort shrink"
                    )
                units.append(
                    {
                        "edge": edge,
                        "order": order,
                        "target": q,
                        "left": left,
                        "right": right,
                        "input_ids": ids,
                    }
                )
    if len(units) != cfg["enquiries"] or e.shape != (cfg["edges"], 2):
        raise ValueError("Changed registered comparison scope")
    model_files = {
        p.name: R.digest(p)
        for p in MODEL.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    if model_files != pm["model_files"]:
        raise ValueError("Changed fixed observer")
    inputs = [
        parent / p
        for p in (
            "manifest.json",
            "observations.npz",
            "selections.json",
            "verification_seal.json",
        )
    ]
    inputs += [
        positions / p
        for p in ("manifest.json", "score_records.npz", "verification_seal.json")
    ]
    inputs += [ROOT / cfg[k] for k in ("material", "labels")]
    (out / "batches").mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "model_files": model_files,
            "alphabet": [k[0] for k in alphabet],
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "outcomes_used_for_graph_or_prompt": False,
            "original_scope_complete": False,
        },
    )
    R.write(out / "units.json", units)
    R.write(out / "bindings.json", binding)
    R.archive(
        out / "observations.npz",
        edges=e,
        panel_indices=indices,
        standalone=records[:, 6],
        source_full=records[:, 0],
        source_departure=records[:, 2],
    )
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "events": 288,
            "edges": len(e),
            "cycle_dimension": len(M.cycles(e, 24)),
            "prompt_tokens_range": [
                min(len(u["input_ids"]) for u in units),
                max(len(u["input_ids"]) for u in units),
            ],
            "full_histories_no_truncation": True,
            "new_model_endpoints": len(units),
            "public_material_only": True,
            "application_y_values_read": False,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("units.json", "bindings.json", "observations.npz", "input_quality.json"),
    )
    H.status(out, {"status": "PREPARED", "completed_batches": 0})
    print("PREPARED HM226:32edges/9cycles/335endpoints;no labels used", flush=True)


def collect(out):
    """Observe each registered comparison once; no parameter fitting or retries."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    cfg = manifest["config"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No uncertain/terminal automatic retry")
    for name, digest in manifest["model_files"].items():
        if R.digest(MODEL / name) != digest:
            raise ValueError("Changed model")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    elapsed, completed = 0.0, 0
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    try:
        torch.set_num_threads(2)
        stamp = time.monotonic()
        observer = DecoderObserver.load(
            str(MODEL),
            device=cfg["device"],
            dtype=cfg["dtype"],
            max_tokens=cfg["max_tokens"],
        )
        load_seconds = time.monotonic() - stamp
        alpha = manifest["alphabet"]
        R.archive(
            out / "head.npz",
            weights=observer.model.lm_head.weight.detach()[alpha].float().cpu().numpy(),
        )
        units = R.read(out / "units.json")
        size = cfg["batch_size"]
        for start in range(0, len(units), size):
            batch, bi = units[start : start + size], start // size
            if (
                len(batch) != size
                or [b["target"] for b in batch] != list(range(5))
                or len({(b["edge"], b["order"]) for b in batch}) != 1
            ):
                raise ValueError("Changed endpoint-to-edge binding")
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError("Finite model cap reached;preserve, no extension")
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "completed_batches": completed,
                    "pending_batch": bi,
                    "model_seconds": elapsed,
                },
            )
            width = max(len(b["input_ids"]) for b in batch)
            pad = observer.tokenizer.pad_token_id
            if pad is None:
                pad = observer.tokenizer.eos_token_id
            ids = torch.full((size, width), pad, dtype=torch.long, device=cfg["device"])
            mask = torch.zeros_like(ids)
            last = []
            for j, b in enumerate(batch):
                n = len(b["input_ids"])
                ids[j, :n] = torch.tensor(b["input_ids"], device=cfg["device"])
                mask[j, :n] = 1
                last.append(n - 1)
            stamp = time.monotonic()
            try:
                with torch.inference_mode():
                    body = observer.model.model(
                        input_ids=ids,
                        attention_mask=mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    state = body.last_hidden_state[
                        torch.arange(size, device=cfg["device"]),
                        torch.tensor(last, device=cfg["device"]),
                    ]
                    logits = (
                        observer.model.lm_head(state)
                        .float()
                        .cpu()
                        .numpy()
                        .astype(float)
                    )
                    states = state.float().cpu().numpy()
                if not np.isfinite(logits).all() or not np.isfinite(states).all():
                    raise ValueError("Nonfinite native reading")
                R.archive(
                    out / "batches" / f"{bi:03d}.npz",
                    logits=logits[:, alpha],
                    normalizer=logsumexp(logits, -1),
                    states=states,
                    greedy=logits.argmax(-1),
                )
            finally:
                elapsed += time.monotonic() - stamp
            completed += 1
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "completed_batches": completed,
                    "pending_batch": None,
                    "model_seconds": elapsed,
                },
            )
            if completed % 8 == 0 or completed == cfg["batches"]:
                print(
                    f"HM226 {completed}/{cfg['batches']};model_seconds={elapsed:.3f}",
                    flush=True,
                )
            del body, state, logits, states, ids, mask
        H.status(
            out,
            {
                "status": "COLLECTED",
                "pid": None,
                "completed_batches": completed,
                "model_seconds": elapsed,
                "load_seconds": load_seconds,
                "new_API_train_private_calls": 0,
            },
        )
        R.seal(
            out,
            "collected_seal.json",
            ("head.npz", "runtime.json")
            + tuple(f"batches/{i:03d}.npz" for i in range(cfg["batches"])),
        )
    except BaseException as exc:
        H.status(
            out,
            {
                "status": "FAILED",
                "pid": None,
                "completed_batches": completed,
                "model_seconds": elapsed,
                "error": repr(exc),
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def read_native(out, cfg):
    """Bind every native endpoint and full-vocabulary mass, including self controls."""
    units = R.read(out / "units.json")
    batches = [
        H.selected(out / "batches" / f"{b:03d}.npz", ("logits", "normalizer"))
        for b in range(cfg["batches"])
    ]
    logits = np.concatenate([b["logits"] for b in batches])
    normalizer = np.concatenate([b["normalizer"] for b in batches])
    z = logits[:, 0] - logits[:, 1]
    mass = np.exp(logsumexp(logits, -1) - normalizer)
    values, self_values = np.empty((32, 2, 5)), np.empty((3, 5))
    rows = []
    for k, u in enumerate(units):
        if u["edge"] < 32:
            values[u["edge"], u["order"], u["target"]] = z[k]
        else:
            self_values[u["edge"] - 32, u["target"]] = z[k]
        rows.append(
            {
                "endpoint": k,
                "edge": u["edge"],
                "order": u["order"],
                "target": cfg["targets"][u["target"]],
                "left": u["left"],
                "right": u["right"],
                "log_odds_A_B": float(z[k]),
                "candidate_mass": float(mass[k]),
            }
        )
    return values, self_values, rows


def analyze(out):
    """Preserve all geometry and anchor contacts without selecting a better score."""
    cfg = R.check(out, STAGES[:1])["config"]
    obs = R.prepared(out)
    z, selfs, endpoints = read_native(out, cfg)
    d, even = M.ordered_accounts(z)
    a = M.decompose(obs["edges"], d)
    b, cb = M.incidence(obs["edges"], 24), M.cycles(obs["edges"], 24)
    objects = np.stack(
        (
            a["positions"],
            obs["standalone"],
            obs["source_full"],
            obs["source_departure"],
        ),
        1,
    )
    labels = H.selected(ROOT / cfg["labels"], ("y",))["y"][obs["panel_indices"]]
    if labels.shape != (24, 5) or not np.isfinite(labels).all():
        raise ValueError("Existing public anchor shape invalid")
    flat = objects.reshape(24, 20)
    contact = P.correlations(flat, labels)
    rng = np.random.default_rng(cfg["seed"])
    permutations = np.stack([rng.permutation(24) for _ in range(cfg["permutations"])])
    null = np.stack([P.correlations(flat, labels[ix]) for ix in permutations])
    contacts = []
    for j, name in enumerate(NAMES):
        for q, target in enumerate(cfg["targets"]):
            for h, anchor in enumerate(cfg["targets"]):
                index = j * 5 + q
                r = contact[index, h]
                contacts.append(
                    {
                        "object": name,
                        "target": target,
                        "human_anchor": anchor,
                        "pearson_r": finite(r),
                        "spearman_rho": finite(
                            spearmanr(flat[:, index], labels[:, h]).statistic
                        ),
                        "authors": 24,
                        "point_tail": (
                            int(np.count_nonzero(abs(null[:, index, h]) >= abs(r))) + 1
                        )
                        / (cfg["permutations"] + 1),
                        "family100_tail": (
                            int(
                                np.count_nonzero(
                                    np.nanmax(abs(null), axis=(1, 2)) >= abs(r)
                                )
                            )
                            + 1
                        )
                        / (cfg["permutations"] + 1),
                    }
                )
    edge_rows, cycle_rows, metrics = [], [], []
    edge_lookup = {tuple(edge): k for k, edge in enumerate(obs["edges"].tolist())}
    for q, target in enumerate(cfg["targets"]):
        for k, (left, right) in enumerate(obs["edges"]):
            edge_rows.append(
                {
                    "edge": k,
                    "left": int(left),
                    "right": int(right),
                    "target": target,
                    "odd": float(d[k, q]),
                    "even": float(even[k, q]),
                    "gradient": float(a["gradient"][k, q]),
                    "cycle_remainder": float(a["remainder"][k, q]),
                }
            )
        hard, triangle = 0, []
        for u in range(0, 24, 3):
            path = np.asarray(
                [
                    d[edge_lookup[(u, u + 1)], q],
                    d[edge_lookup[(u + 1, u + 2)], q],
                    -d[edge_lookup[(u, u + 2)], q],
                ]
            )
            triangle.append(float(path.sum()))
            hard += int(np.all(path > 0) or np.all(path < 0))
        for k, row in enumerate(a["cycles"][:, q]):
            cycle_rows.append(
                {
                    "target": target,
                    "fundamental_cycle": k,
                    "log_odds_circulation": float(row),
                    "edges": int(np.count_nonzero(cb[k])),
                }
            )
        total = float(np.sum(d[:, q] ** 2))
        metrics.append(
            {
                "target": target,
                "odd_rms": float(np.sqrt(np.mean(d[:, q] ** 2))),
                "even_rms": float(np.sqrt(np.mean(even[:, q] ** 2))),
                "gradient_rms": float(np.sqrt(np.mean(a["gradient"][:, q] ** 2))),
                "cycle_remainder_rms": float(
                    np.sqrt(np.mean(a["remainder"][:, q] ** 2))
                ),
                "cycle_squared_ratio_not_psych_fraction": float(
                    np.sum(a["remainder"][:, q] ** 2) / total
                )
                if total
                else None,
                "max_fundamental_cycle": float(np.max(abs(a["cycles"][:, q]))),
                "triangle_hard_cycles": hard,
                "triangles": 8,
                "triangle_rms": float(np.sqrt(np.mean(np.asarray(triangle) ** 2))),
                "standalone_edge_contact": finite(
                    P.correlations(d[:, [q]], (b @ obs["standalone"])[:, [q]])[0, 0]
                ),
                "source_edge_contact": finite(
                    P.correlations(d[:, [q]], (b @ obs["source_full"])[:, [q]])[0, 0]
                ),
            }
        )
    R.table(out / "endpoints.csv", endpoints)
    R.table(out / "edges.csv", edge_rows)
    R.table(out / "cycles.csv", cycle_rows)
    R.table(out / "geometry.csv", metrics)
    R.table(out / "anchor_contacts.csv", contacts)
    R.archive(
        out / "analysis.npz",
        ordered=z,
        odd=d,
        even=even,
        self_readings=selfs,
        incidence=b,
        cycle_basis=cb,
        **a,
        objects=objects,
        labels=labels,
        contact=contact,
        permutations=permutations,
    )
    R.write(
        out / "summary.json",
        {
            "packet": "HM226",
            "metrics": metrics,
            "self_log_odds_by_author_target": selfs.tolist(),
            "candidate_mass_range": [
                min(r["candidate_mass"] for r in endpoints),
                max(r["candidate_mass"] for r in endpoints),
            ],
            "all_comparison_contacts": [
                r for r in contacts if r["object"] == "comparison_position"
            ],
            "no_projection_used_as_prediction_promotion": True,
            "graph_shared_not_independent_edges": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "endpoints.csv",
            "edges.csv",
            "cycles.csv",
            "geometry.csv",
            "anchor_contacts.csv",
            "analysis.npz",
            "summary.json",
        ),
    )
    print("ANALYZED HM226;all positions/relations/orders retained", flush=True)


def verify(out):
    """Check an independent graph-Laplacian solution and literal native readouts."""
    cfg = R.check(out, STAGES[:2])["config"]
    obs = R.prepared(out)
    z = H.selected(
        out / "analysis.npz",
        (
            "odd",
            "positions",
            "gradient",
            "remainder",
            "cycles",
            "cycle_basis",
            "objects",
            "labels",
            "contact",
        ),
    )
    b = np.zeros((32, 24))
    for k, (i, j) in enumerate(obs["edges"]):
        b[k, i], b[k, j] = 1, -1
    theta = np.linalg.solve(b.T @ b + np.ones((24, 24)) / 24, b.T @ z["odd"])
    head = H.selected(out / "head.npz", ("weights",))["weights"].astype(float)
    maximum = 0.0
    for k in range(cfg["batches"]):
        v = H.selected(out / "batches" / f"{k:03d}.npz", ("logits", "states"))
        maximum = max(
            maximum,
            float(np.max(abs(v["states"].astype(float) @ head.T - v["logits"]))),
        )
    flat, y = z["objects"].reshape(24, 20), z["labels"]
    independent = np.empty((20, 5))
    for i in range(20):
        for j in range(5):
            x0, y0 = flat[:, i] - flat[:, i].mean(), y[:, j] - y[:, j].mean()
            independent[i, j] = (x0 @ y0) / np.sqrt((x0 @ x0) * (y0 @ y0))
    checks = {
        "laplacian_position_max": float(np.max(abs(theta - z["positions"]))),
        "divergence_max": float(np.max(abs(b.T @ z["remainder"]))),
        "signed_reconstruction_max": float(
            np.max(abs(z["gradient"] + z["remainder"] - z["odd"]))
        ),
        "cycle_closure_max": float(np.max(abs(z["cycle_basis"] @ b))),
        "energy_account_max": float(
            np.max(
                abs(
                    np.sum(z["odd"] ** 2, 0)
                    - np.sum(z["gradient"] ** 2, 0)
                    - np.sum(z["remainder"] ** 2, 0)
                )
            )
        ),
        "native_head_max": maximum,
        "all100_scalar_contact_max": float(np.nanmax(abs(independent - z["contact"]))),
        "same_agent_not_blind": True,
    }
    passed = all(
        checks[k] < (1e-4 if k == "native_head_max" else 1e-10)
        for k in checks
        if k != "same_agent_not_blind"
    )
    checks["PASS"] = passed
    R.write(out / "verification.json", checks)
    R.seal(out, "verification_seal.json", ("verification.json",))
    if not passed:
        raise ValueError("Independent verification failed; preserve results")
    print("VERIFIED HM226", checks, flush=True)


def main():
    """Run one requested stage; completed science is never overwritten."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, STAGES)
        print("HM226 ALL SEALS PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
