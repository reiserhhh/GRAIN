#!/usr/bin/env python3
"""Finite prospective SOURCE evidence selection and actual new native reading."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_prospective_evidence as M
from grain_research.vector_mechanisms import direct_traits as D
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_vector_mechanism_external_features import bundle_arms

CONFIG = "configs/grain_prospective_evidence_hm218.json"
DEFAULT = ROOT / "results/grain_math_response_v1/prospective_evidence_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_PROSPECTIVE_EVIDENCE_PLAN_HM218.md",
    "grain_research/construction/math_prospective_evidence.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "tests/test_grain_prospective_evidence.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_vector_mechanism_external_features.py",
    "grain_research/vector_mechanisms/external_material.py",
)
TASKS = tuple(D.BIG5)


def selected(path, keys):
    """Read an explicit key allowlist, never application labels or pickle."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def status(out, value):
    """Update only the recoverable runtime receipt, not scientific artifacts."""
    p = out / "runtime.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def prepare(out):
    """Select and freeze all new evidence inputs before native outputs exist."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No replacement or automatic restart")
    cfg = R.read(ROOT / CONFIG)
    source, panel, old, material = (
        ROOT / cfg[k]
        for k in ("source", "panel_parent", "historical_native", "material")
    )
    R.check(source, ("computed_seal.json", "verification_seal.json"))
    R.check(panel, ("computed_seal.json", "verification_seal.json"))
    bank = selected(
        source / "observations.npz",
        ("source_means", "codes", "variance", "source_indices"),
    )
    px = selected(panel / "observations.npz", ("panel_indices",))
    idx = px["panel_indices"]
    if len(idx) != 24 or set(idx) & set(bank["source_indices"]):
        raise ValueError("Changed panel or source overlap")
    nm, mm = R.read(old / "manifest.json"), R.read(material / "material_manifest.json")
    inputs = [
        source / "observations.npz",
        source / "verification_seal.json",
        panel / "observations.npz",
        panel / "verification_seal.json",
        old / "manifest.json",
        old / "private_material.json",
        material / "material_manifest.json",
        material / "private_material.json",
        material / "private_arrays.npz",
        ROOT / cfg["context_bundle"],
    ]
    if R.digest(old / "private_material.json") != nm["material_sha256"] or any(
        R.digest(material / p) != mm["files"][p]
        for p in ("private_material.json", "private_arrays.npz")
    ):
        raise ValueError("Changed public material/vector provenance")
    bundles = R.read(material / "private_material.json")["bundles"]
    ids = selected(material / "private_arrays.npz", ("author_ids",))["author_ids"]
    if not np.array_equal(ids, [b["author_id"] for b in bundles]):
        raise ValueError("Vector/author address mismatch")
    previous = {
        (r["author_id"], r["view"]): r for r in R.read(old / "private_material.json")
    }
    cache = {Path(p).stem: p for p in nm["encoder_cache_sha256"]}
    top = set(R.read(ROOT / cfg["context_bundle"])["context_rule"]["top"])
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    alphabet = D.candidates("agreeableness", tok)
    units, selections, all_vectors, all_points, all_scores, full_scores = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    all_conditions, all_lengths, all_subsets = [], [], []
    prompt_lengths, mask_changes = [], 0
    for u, original in enumerate(idx):
        bundle, uid = bundles[int(original)], str(ids[original])
        if bundle["source_group"] != "big5_only":
            raise ValueError("Public Big5-only panel required")
        arms = bundle_arms(bundle)
        if any(
            arms["cue_mask"][v * 12 : (v + 1) * 12] != previous[(uid, v)]["texts"]
            for v in (0, 1)
        ):
            raise ValueError("Changed old material-to-author binding")
        events = bundle["views"][2]
        event_ids = [c["comment_id"] for c in events]
        old_ids = {c["comment_id"] for v in bundle["views"][:2] for c in v}
        if len(events) != 12 or len(set(event_ids)) != 12 or old_ids & set(event_ids):
            raise ValueError("New window does not contain12 disjoint addressed events")
        path = ROOT / cache[uid]
        if R.digest(path) != nm["encoder_cache_sha256"][cache[uid]]:
            raise ValueError("Changed cached E5 comment vectors")
        meta = R.read(path.with_suffix(".json"))
        key = meta["feature_aliases"].get(
            "cue_mask_comment_vectors", "cue_mask_comment_vectors"
        )
        vectors = selected(path, (key,))[key][2].astype(float)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        inputs += [path, path.with_suffix(".json")]
        texts = arms["cue_mask"][24:36]
        mask_changes += sum(
            a != b for a, b in zip(texts, arms["raw"][24:36], strict=True)
        )
        conditions = np.array([e["subreddit"] in top for e in events], int)
        lengths = np.array(
            [len(tok.encode(t, add_special_tokens=False)) for t in texts], int
        )
        choice = M.select(
            bank,
            vectors,
            conditions,
            lengths,
            absolute=cfg["length_absolute_tolerance"],
            relative=cfg["length_relative_tolerance"],
        )
        si, ci = choice["support"], choice["counter"]
        sets = [list(range(12))] + [
            [i for i in range(12) if i not in choice["subsets"][j]] for j in (si, ci)
        ]
        all_vectors.append(vectors)
        all_points.append(choice["points"])
        all_scores.append(choice["accounts"])
        all_subsets.append(choice["subsets"])
        full_scores.append(M.accounts(bank, vectors.mean(0, keepdims=True))[0])
        all_conditions.append(conditions)
        all_lengths.append(lengths)
        selections.append(
            {
                "author": u,
                "original_author_index": int(original),
                "event_ids": event_ids,
                "support": choice["subsets"][si].tolist(),
                "counter": choice["subsets"][ci].tolist(),
                "support_index": si,
                "counter_index": ci,
                "feasible_pairs": choice["feasible_pairs"],
                "removed_condition_counts": [
                    int(choice["condition_counts"][j]) for j in (si, ci)
                ],
                "removed_token_lengths": [int(choice["lengths"][j]) for j in (si, ci)],
            }
        )
        for arm, slots in enumerate(sets):
            for target, task in enumerate(TASKS):
                tokens, _, _ = D.tokenize_prompt([texts[i] for i in slots], task, tok)
                if len(tokens) + 16 > cfg["max_tokens"]:
                    raise ValueError("Unsupported full scope; no truncation")
                prompt_lengths.append(len(tokens))
                units.append(
                    {"author": u, "arm": arm, "target": target, "input_ids": tokens}
                )
    if len(units) != cfg["enquiries"]:
        raise ValueError("Declared observations incomplete")
    model_files = {
        p.name: R.digest(p)
        for p in MODEL.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    if model_files["model.safetensors"] != nm["model_sha256"]:
        raise ValueError("Changed native observer")
    (out / "batches").mkdir(parents=True, exist_ok=False)
    R.archive(
        out / "observations.npz",
        **bank,
        panel_indices=idx,
        vectors=np.asarray(all_vectors),
        points=np.asarray(all_points),
        accounts=np.asarray(all_scores),
        subsets=np.asarray(all_subsets),
        full_accounts=np.asarray(full_scores),
        conditions=np.asarray(all_conditions),
        lengths=np.asarray(all_lengths),
    )
    R.write(out / "selections.json", selections)
    R.write(out / "units.json", units)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "alphabet": alphabet,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "model_files": model_files,
            "source_only_selection": True,
            "original_scope_complete": False,
        },
    )
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "events": 288,
            "source_authors": 386,
            "source_panel_overlap": 0,
            "old_event_overlap": 0,
            "prompt_tokens_range": [min(prompt_lengths), max(prompt_lengths)],
            "cue_mask_changed_comments": mask_changes,
            "application_labels_read": False,
            "new_native_material_not_new_authors": True,
            "all_selection_and_predictions_before_native": True,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "selections.json", "units.json", "input_quality.json"),
    )
    status(out, {"status": "PREPARED", "completed_batches": 0})
    print(
        "PREPARED HM218:24authors/288new-window events/72batches/360questions",
        flush=True,
    )


def collect(out):
    """Observe the frozen72 finite batches once, preserving any failed forward."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    cfg = manifest["config"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No uncertain/terminal automatic retry")
    for name, h in manifest["model_files"].items():
        if R.digest(MODEL / name) != h:
            raise ValueError("Changed native model")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    elapsed, completed = 0.0, 0
    try:
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(MODEL),
            device=cfg["device"],
            dtype=cfg["dtype"],
            max_tokens=cfg["max_tokens"],
        )
        units = R.read(out / "units.json")
        alpha = manifest["alphabet"]["token_ids"]
        head = observer.model.lm_head.weight.detach()[alpha].float().cpu().numpy()
        R.archive(out / "head.npz", weights=head)
        for start in range(0, len(units), 5):
            batch = units[start : start + 5]
            if [b["target"] for b in batch] != list(range(5)) or len(
                {(b["author"], b["arm"]) for b in batch}
            ) != 1:
                raise ValueError("Batch endpoint addressing mismatch")
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError("Finite cap reached; no extension")
            bi = start // 5
            runtime = {
                "status": "RUNNING",
                "pid": os.getpid(),
                "pending_batch": bi,
                "completed_batches": completed,
                "model_seconds": elapsed,
            }
            status(out, runtime)
            pad = observer.tokenizer.pad_token_id
            if pad is None:
                pad = observer.tokenizer.eos_token_id
            width = max(len(b["input_ids"]) for b in batch)
            ids = torch.full((5, width), pad, dtype=torch.long, device=cfg["device"])
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
                    output = observer.model.model(
                        input_ids=ids,
                        attention_mask=mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    state = output.last_hidden_state[
                        torch.arange(5, device=cfg["device"]),
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
                    raise ValueError("Nonfinite native observation")
                R.archive(
                    out / "batches" / f"{bi:03d}.npz",
                    logits=logits[:, alpha],
                    normalizer=logsumexp(logits, -1),
                    states=states,
                )
            finally:
                elapsed += time.monotonic() - stamp
            completed += 1
            status(
                out,
                runtime
                | {
                    "pending_batch": None,
                    "completed_batches": completed,
                    "model_seconds": elapsed,
                },
            )
            if completed % 12 == 0:
                print(
                    f"HM218 observed {completed}/72 batches;model_seconds={elapsed:.3f}",
                    flush=True,
                )
            del output, logits, state, states
        status(
            out,
            {
                "status": "COLLECTED",
                "completed_batches": completed,
                "model_seconds": elapsed,
                "pid": None,
                "new_API_train_private_calls": 0,
            },
        )
        R.seal(
            out,
            "collected_seal.json",
            ("head.npz", "runtime.json")
            + tuple(f"batches/{i:03d}.npz" for i in range(72)),
        )
        print("COLLECTED HM218", elapsed, flush=True)
    except BaseException as exc:
        status(
            out,
            {
                "status": "FAILED",
                "completed_batches": completed,
                "model_seconds": elapsed,
                "error": repr(exc),
                "no_auto_retry": True,
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def native(out):
    """Load exactly the declared complete output family, not favorable cells."""
    rows = [
        selected(out / "batches" / f"{i:03d}.npz", ("logits", "normalizer", "states"))
        for i in range(72)
    ]
    return {
        k: np.stack([r[k] for r in rows]).reshape((24, 3) + rows[0][k].shape)
        for k in rows[0]
    }


def analyze(out):
    """Compare prospective signed actions without choosing a predictor or head."""
    cfg = R.check(out, ("collected_seal.json",))["config"]
    z, n = R.prepared(out), native(out)
    selections = R.read(out / "selections.json")
    y, raw, specific = M.native_accounts(n["logits"])
    source_actions = np.stack(
        [
            z["accounts"][u, r["counter_index"]] - z["accounts"][u, r["support_index"]]
            for u, r in enumerate(selections)
        ]
    )
    raw_ci = M.bootstrap(raw, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
    specific_ci = M.bootstrap(specific, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
    rows = []
    for q, name in enumerate(cfg["targets"]):
        rows.append(
            {
                "target": name,
                "raw_mean": float(raw[:, q].mean()),
                "raw_ci_low": float(raw_ci[q, 0]),
                "raw_ci_high": float(raw_ci[q, 1]),
                "raw_positive_authors": int(np.count_nonzero(raw[:, q] > 0)),
                "specific_mean": float(specific[:, q].mean()),
                "specific_ci_low": float(specific_ci[q, 0]),
                "specific_ci_high": float(specific_ci[q, 1]),
                "specific_positive_authors": int(np.count_nonzero(specific[:, q] > 0)),
                "source_departure_mean": float(source_actions[:, 2, q].mean()),
            }
        )
    compliance = np.exp(logsumexp(n["logits"], -1) - n["normalizer"])
    R.table(out / "actions_by_target.csv", rows)
    R.table(
        out / "actions_by_author.csv",
        [
            {
                "author": u,
                "target": cfg["targets"][q],
                "source_departure": float(source_actions[u, 2, q]),
                "native_raw": float(raw[u, q]),
                "native_specific": float(specific[u, q]),
                "remove_support_minus_full": float(y[u, 1, q] - y[u, 0, q]),
                "remove_counter_minus_full": float(y[u, 2, q] - y[u, 0, q]),
            }
            for u in range(24)
            for q in range(5)
        ],
    )
    R.archive(
        out / "analysis.npz",
        native_readings=y,
        raw=raw,
        specific=specific,
        source_actions=source_actions,
        compliance=compliance,
    )
    R.write(
        out / "summary.json",
        {
            "packet": "HM218",
            "primary": rows[2],
            "all_targets": rows,
            "source_positive_A_by_construction": True,
            "generic_mean": float(raw.mean()),
            "compliance_range": [float(compliance.min()), float(compliance.max())],
            "new_native_batches": 72,
            "new_native_enquiries": 360,
            "new_authors": 0,
            "new_window_events": 288,
            "new_model_fit_API_training": 0,
            "no_prediction_accuracy_claim": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "analysis.npz",
            "summary.json",
            "actions_by_target.csv",
            "actions_by_author.csv",
        ),
    )
    print(
        json.dumps({"primary": rows[2], "generic_mean": float(raw.mean())}), flush=True
    )


def verify(out):
    """Audit literal distance kernels, output head arithmetic and all addresses."""
    R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    z, n = R.prepared(out), native(out)
    selections = R.read(out / "selections.json")
    source = z["source_means"]
    codes = z["codes"]
    errors = []
    for u, r in enumerate(selections):
        for index in (r["support_index"], r["counter_index"]):
            point = z["points"][u, index]
            k = np.exp(
                -np.sum((source - point) ** 2, -1) / (2 * float(z["variance"]))
            ).mean(1)
            law = np.array([k[codes == c].sum() / len(k) for c in range(32)])
            for q in range(5):
                high = ((np.arange(32) >> q) & 1).astype(bool)
                c = codes & ~(1 << q)
                base = np.array(
                    [
                        k[c == (g & ~(1 << q))].mean()
                        * np.count_nonzero(codes == g)
                        / len(k)
                        for g in range(32)
                    ]
                )
                f = np.log(law[high].sum() / law[~high].sum())
                b = np.log(base[high].sum() / base[~high].sum())
                errors.extend(
                    abs(np.array([f, b, f - b]) - z["accounts"][u, index, :, q])
                )
        s, c = r["support"], r["counter"]
        if (
            set(s) & set(c)
            or z["conditions"][u, s].sum() != z["conditions"][u, c].sum()
        ):
            raise ValueError("Matched event constraint violated")
        for index, removal in ((r["support_index"], s), (r["counter_index"], c)):
            errors.append(
                float(
                    np.max(
                        abs(
                            np.delete(z["vectors"][u], removal, 0).mean(0)
                            - z["points"][u, index]
                        )
                    )
                )
            )
    weights = selected(out / "head.npz", ("weights",))["weights"].astype(float)
    head_error = float(np.max(abs(n["states"].astype(float) @ weights.T - n["logits"])))
    source_error = float(max(errors))
    if source_error > 1e-11 or head_error > 3e-4:
        raise ValueError("Registered arithmetic tolerance failed; no precision rescue")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "literal_accounts": 720,
            "source_max_error": source_error,
            "native_head_max_error": head_error,
            "native_endpoints": 360,
            "same_agent_alternate_arithmetic_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED HM218", source_error, head_error, flush=True)


def main():
    """Execute one explicit stage; sealed stages never automatically restart."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(
            args.output,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("ALL HM218 SEALS PASS", flush=True)
    else:
        globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
