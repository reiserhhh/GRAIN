#!/usr/bin/env python3
"""Run the bounded source-only local action-metric diagnostic.

This packet reuses the frozen conditional-operator material.  It computes a
reader Jacobian only on source low-text contexts, fits Euclidean and
Jacobian-weighted action operators, and tests both by native intervention on
the existing application pairs.  It does not generate text or fit on
application outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.action_metric_diagnostic import (
    fit_action_weighted,
    fit_euclidean_action,
    paired_metrics,
    predict_euclidean,
    predict_weighted,
)
from grain_research.vector_mechanisms.direct_traits import candidates
from scripts.run_grain_conditional_operator import QUERIES, inputs
from scripts.run_grain_conditional_operator_native import (
    cache_path,
    context,
    intervene,
)
from scripts.run_grain_persona_generation_trace import load_observer, sha, write

NATIVE = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2"
DEFAULT = ROOT / "results/grain_measurement_discovery_v1/pv01/r1"
SOURCE_PAIRS = 36
APPLICATION_PAIRS = 72
MAX_FORWARD = 450
MAX_BACKWARD = 72


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a deterministic CSV, including an empty table when necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value) -> None:
    """Write strict JSON without silently converting non-finite values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, value)


def build_manifest(out: Path, max_seconds: int, max_model_calls: int) -> dict:
    """Describe all frozen inputs and the bounded numerical protocol."""
    files = [
        NATIVE / name
        for name in ("action_arrays.npz", "native_effects.jsonl", "pair_order.json")
    ]
    files += [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/action_metric_diagnostic.py",
        ROOT / "grain_research/vector_mechanisms/conditional_operator.py",
        ROOT / "scripts/run_grain_conditional_operator_native.py",
    ]
    return {
        "task_id": "PV01",
        "version": "r1",
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in files},
        "source_pairs": SOURCE_PAIRS,
        "application_pairs": APPLICATION_PAIRS,
        "queries": list(QUERIES),
        "alpha": 1.0,
        "finite_difference_steps": [0.01, 0.005],
        "finite_difference_cases": 6,
        "jacobian_site": "L12.H0 o_proj input, low-text source context",
        "max_seconds": int(max_seconds),
        "max_forward_calls": int(max_model_calls),
        "max_backward_calls": MAX_BACKWARD,
        "no_application_fit": True,
    }


def load_frozen_data():
    """Load and validate the existing pair order and hidden-action shapes."""
    rows, pairs = inputs()
    with np.load(NATIVE / "action_arrays.npz") as arrays:
        dx = arrays["dx"].astype(float)
        dh = arrays["dh"].astype(float)
        ds = arrays["ds"].astype(float)
        source = arrays["source"].astype(bool)
    order = json.loads((NATIVE / "pair_order.json").read_text())
    if order != [pair["pair"] for pair in pairs]:
        raise ValueError("Frozen pair order differs from current input reconstruction")
    if (len(pairs), dx.shape, ds.shape, source.shape) != (
        108,
        (108, 4),
        (108, 2),
        (108,),
    ):
        raise ValueError("Unexpected frozen action-array shapes")
    if int(source.sum()) != SOURCE_PAIRS:
        raise ValueError("Unexpected source pair count")
    return rows, pairs, dx, dh, ds, source


def load_effects() -> dict[tuple[str, str, str], dict]:
    """Index cached native effects and reject duplicate application keys."""
    indexed = {}
    path = NATIVE / "native_effects.jsonl"
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["pair"], row["query"], row["kind"])
            if key in indexed:
                raise ValueError(f"Duplicate cached native effect {key}")
            indexed[key] = row
    return indexed


def source_shapes(pairs: list[dict]) -> dict[str, tuple[int, int]]:
    """Read the fixed suffix/head shapes from one source pair's frozen cache."""
    source = next(pair for pair in pairs if pair["split"] == "source")
    shapes = {}
    for query in QUERIES:
        with np.load(cache_path(NATIVE, source["high"]["text_id"], query)) as cached:
            shapes[query] = tuple(int(v) for v in cached["head"].shape)
    return shapes


def native_score_gradient(observer, ids, positions, alphabet, counters):
    """Return the native conditional-score gradient at the fixed head input."""
    import torch

    if counters["forward"] >= counters["max_forward"]:
        raise TimeoutError("forward budget exhausted")
    model = observer.model
    device = next(model.parameters()).device
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(tokens)
    dim = model.config.hidden_size // model.config.num_attention_heads
    box = {}
    module = model.model.layers[12].self_attn.o_proj

    def hook(_module, args):
        current = args[0].detach().clone().requires_grad_(True)
        box["input"] = current
        return (current,)

    handle = module.register_forward_pre_hook(hook)
    model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
            output = model(
                input_ids=tokens,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            )
            logits = output.logits[0, -1]
            token_ids = torch.tensor(
                alphabet["token_ids"], dtype=torch.long, device=device
            )
            values = torch.tensor(alphabet["values"], dtype=logits.dtype, device=device)
            local = logits[token_ids]
            probabilities = torch.softmax(local, dim=0)
            score = (probabilities * values).sum()
            score.backward()
        captured = box.get("input")
        if captured is None or captured.grad is None:
            raise RuntimeError("L12.H0 input did not retain a gradient")
        gradient = captured.grad[0, positions, :dim].detach().float().cpu().numpy()
        score_value = float(score.detach().cpu().item())
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    counters["forward"] += 1
    counters["backward"] += 1
    if gradient.ndim != 2 or not np.isfinite(gradient).all():
        raise RuntimeError("Non-finite or malformed native reader gradient")
    return gradient, score_value


def run_intervention(observer, ids, positions, alphabet, delta, counters):
    """Run one bounded native intervention and account for its forward."""
    if counters["forward"] >= counters["max_forward"]:
        raise TimeoutError("forward budget exhausted")
    result = intervene(observer, ids, positions, alphabet, delta)
    counters["forward"] += 1
    return float(result["score"]), float(result["mass"])


def finite_difference_checks(observer, pairs, gradients, shapes, counters):
    """Check six fixed source directions at two central-difference scales."""
    cases = []
    source_pairs = [pair for pair in pairs if pair["split"] == "source"]
    requested = [(i, "extraversion") for i in range(3)] + [
        (i, "neuroticism") for i in range(3)
    ]
    for index, query in requested:
        pair = source_pairs[index % len(source_pairs)]
        low = pair["low"]
        ids, positions = context(observer, low["text"], query)
        shape = shapes[query]
        if (
            len(positions),
            observer.model.config.hidden_size
            // observer.model.config.num_attention_heads,
        ) != shape:
            raise ValueError("Reader suffix shape differs from frozen hidden action")
        dim = int(np.prod(shape))
        rng = np.random.default_rng(20260915 + index)
        direction = rng.normal(size=dim)
        direction /= np.linalg.norm(direction)
        direction = direction.reshape(shape)
        analytic = float(
            np.dot(gradients[(pair["pair"], query)].ravel(), direction.ravel())
        )
        alphabet = candidates(query, observer.tokenizer)
        row = {
            "pair": pair["pair"],
            "query": query,
            "direction_seed": 20260915 + index,
            "dimension": dim,
            "analytic_directional_derivative": analytic,
        }
        for step in (0.01, 0.005):
            plus, _ = run_intervention(
                observer, ids, positions, alphabet, step * direction, counters
            )
            minus, _ = run_intervention(
                observer, ids, positions, alphabet, -step * direction, counters
            )
            estimate = (plus - minus) / (2 * step)
            key = f"h_{step:g}"
            error = abs(estimate - analytic)
            tolerance = 1e-5 + 1e-2 * max(abs(estimate), abs(analytic))
            row[f"fd_{key}"] = estimate
            row[f"abs_error_{key}"] = error
            row[f"tolerance_{key}"] = tolerance
            row[f"status_{key}"] = "PASS" if error <= tolerance else "PRECISION_LIMITED"
        cases.append(row)
    return cases


def application_metrics(
    observer, pairs, dx, dh, ds, source, gradients, shapes, effects, counters
):
    """Fit source operators and evaluate native application interventions."""
    source_index = np.flatnonzero(source)
    application_index = np.flatnonzero(~source)
    models = {}
    predictions = {}
    for qi, query in enumerate(QUERIES):
        shape = shapes[query]
        start = 0 if query == "extraversion" else int(np.prod(shapes["extraversion"]))
        stop = start + int(np.prod(shape))
        target = dh[:, start:stop]
        jacobian = np.asarray(
            [gradients[(pairs[i]["pair"], query)].ravel() for i in source_index]
        )
        euclidean = fit_euclidean_action(dx[source_index], target[source_index])
        weighted = fit_action_weighted(dx[source_index], target[source_index], jacobian)
        models[query] = {"euclidean": euclidean, "weighted": weighted}
        predictions[(query, "euclidean")] = predict_euclidean(euclidean, dx)
        predictions[(query, "weighted")] = predict_weighted(weighted, dx)

    observed = {
        key: []
        for key in [
            (q, method) for q in QUERIES for method in ("euclidean", "weighted")
        ]
    }
    donor = {q: [] for q in QUERIES}
    full = {q: [] for q in QUERIES}
    pair_rows = []
    for i in application_index:
        pair = pairs[i]
        for qi, query in enumerate(QUERIES):
            shape = shapes[query]
            start = (
                0 if query == "extraversion" else int(np.prod(shapes["extraversion"]))
            )
            stop = start + int(np.prod(shape))
            ids, positions = context(observer, pair["low"]["text"], query)
            alphabet = candidates(query, observer.tokenizer)
            base = float(pair["low"]["scores"][qi])
            contrast = float(ds[i, qi])
            cached_donor = effects[(pair["pair"], query, "donor")]
            donor_effect = float(cached_donor["effect"])
            donor[query].append(donor_effect)
            full[query].append(contrast)
            row = {
                "pair": pair["pair"],
                "query": query,
                "contrast": contrast,
                "donor_effect": donor_effect,
                "low_score": base,
            }
            for method in ("euclidean", "weighted"):
                action = predictions[(query, method)][i].reshape(shape)
                score, mass = run_intervention(
                    observer, ids, positions, alphabet, action, counters
                )
                effect = score - base
                observed[(query, method)].append(effect)
                row[f"{method}_effect"] = effect
                row[f"{method}_mass"] = mass
            pair_rows.append(row)

    metric_rows = []
    for query in QUERIES:
        for method in ("euclidean", "weighted"):
            for target_name, target in (
                ("full_text_delta", full[query]),
                ("native_donor", donor[query]),
            ):
                values = paired_metrics(
                    np.asarray(observed[(query, method)]), np.asarray(target)
                )
                metric_rows.append(
                    {"query": query, "method": method, "target": target_name, **values}
                )
        legacy = [
            float(effects[(pairs[i]["pair"], query, "compiled")]["effect"])
            for i in application_index
        ]
        for target_name, target in (
            ("full_text_delta", full[query]),
            ("native_donor", donor[query]),
        ):
            values = paired_metrics(np.asarray(legacy), np.asarray(target))
            metric_rows.append(
                {
                    "query": query,
                    "method": "legacy_rank2_compiled",
                    "target": target_name,
                    **values,
                }
            )
    return models, metric_rows, pair_rows


def decision_text(
    manifest, gradient_checks, metric_rows, counters, status="COMPLETE"
) -> str:
    """Create a concise decision record without promoting a reader metric."""
    passed = all(
        row.get("status_h_0.01") == "PASS" and row.get("status_h_0.005") == "PASS"
        for row in gradient_checks
    )
    lines = [
        "# PV01 decision",
        "",
        f"- implementation_status: `{status}`",
        "- scientific_scope: `READER` local action-metric diagnostic; not human personality validity.",
        f"- gradient_checks: `{'PASS' if passed else 'PRECISION_LIMITED'}` for {len(gradient_checks)} fixed source directions.",
        f"- forward_calls: `{counters['forward']}` / `{counters['max_forward']}`; backward_calls: `{counters['backward']}` / `{MAX_BACKWARD}`.",
        "- fitting_scope: source pairs only; application outcomes were not used for fitting or model selection.",
        "",
        "## Interpretation rule",
        "",
        "The action-weighted arm is informative only if its native application error improves against the ordinary Euclidean arm without being a numerical-gradient artifact. This packet does not convert a local Jacobian into a psychological or cross-model claim.",
    ]
    for row in metric_rows:
        if row["target"] == "full_text_delta":
            lines.append(
                f"- {row['query']} / {row['method']} vs full text: RMSE={row['rmse']:.6f}, r={row['r'] if row['r'] is not None else 'null'}."
            )
    lines += [
        "",
        "## Next trigger",
        "",
        "If the weighted arm has no finite gain over Euclidean, close this diagnostic and do not promote it. If it has a finite gain, propose one separately reviewed representation task; do not restart generation or tune on application labels.",
    ]
    return "\n".join(lines) + "\n"


def run(args) -> None:
    """Execute the complete bounded PV01 packet or save a partial packet."""
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(out, args.max_seconds, args.max_model_calls)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("PV01 run identity changed; choose a new output directory")
    save_json(manifest_path, manifest)
    counters = {
        "forward": 0,
        "backward": 0,
        "max_forward": min(args.max_model_calls, MAX_FORWARD),
    }
    started = time.monotonic()
    try:
        _, pairs, dx, dh, ds, source = load_frozen_data()
        if args.smoke:
            keep = np.r_[np.flatnonzero(source)[:1], np.flatnonzero(~source)[:2]]
            pairs = [pairs[int(i)] for i in keep]
            dx, dh, ds, source = dx[keep], dh[keep], ds[keep], source[keep]
        effects = load_effects()
        shapes = source_shapes(pairs)
        observer = load_observer(max_tokens=1024, layer_count=2)
        save_json(out / "model_manifest.json", observer.manifest())
        gradients = {}
        source_pairs = [pair for pair in pairs if pair["split"] == "source"]
        for pair in source_pairs:
            for query in QUERIES:
                ids, positions = context(observer, pair["low"]["text"], query)
                if (
                    len(positions),
                    observer.model.config.hidden_size
                    // observer.model.config.num_attention_heads,
                ) != shapes[query]:
                    raise ValueError(
                        "Source reader shape differs from frozen hidden action"
                    )
                alphabet = candidates(query, observer.tokenizer)
                gradient, score = native_score_gradient(
                    observer, ids, positions, alphabet, counters
                )
                with np.load(
                    cache_path(NATIVE, pair["low"]["text_id"], query)
                ) as cached:
                    if abs(score - float(cached["score"])) > 1e-5:
                        raise ValueError(
                            "Gradient reader score did not replay cached score"
                        )
                gradients[(pair["pair"], query)] = gradient
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("PV01 time budget exhausted during source Jacobians")
        gradient_checks = finite_difference_checks(
            observer, pairs, gradients, shapes, counters
        )
        models, metric_rows, pair_rows = application_metrics(
            observer, pairs, dx, dh, ds, source, gradients, shapes, effects, counters
        )
        save_json(out / "gradient_checks.json", gradient_checks)
        write_csv(out / "metrics.csv", metric_rows)
        write_csv(out / "application_effects.csv", pair_rows)
        save_json(
            out / "call_accounting.json",
            {
                **counters,
                "elapsed_seconds": time.monotonic() - started,
                "status": "COMPLETE",
                "new_generation_calls": 0,
                "new_backward_calls": counters["backward"],
            },
        )
        save_json(
            out / "model_fit_summary.json",
            {
                query: {
                    method: {
                        "scale": model["scale"].tolist(),
                        "weight_shape": list(model["weight"].shape),
                        "source_loss": model.get("source_loss"),
                        "source_residual_rmse": model.get("source_residual_rmse"),
                    }
                    for method, model in methods.items()
                }
                for query, methods in models.items()
            },
        )
        (out / "decision.md").write_text(
            decision_text(manifest, gradient_checks, metric_rows, counters)
        )
        save_json(
            out / "summary.json",
            {
                "task_id": "PV01",
                "status": "PASS",
                "smoke": bool(args.smoke),
                "metric_rows": len(metric_rows),
                "application_rows": len(pair_rows),
            },
        )
        print(
            json.dumps(
                {
                    "task_id": "PV01",
                    "status": "PASS",
                    **counters,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
        )
    except Exception as exc:
        status = "PARTIAL" if isinstance(exc, TimeoutError) else "FAIL"
        save_json(
            out / "call_accounting.json",
            {
                **counters,
                "elapsed_seconds": time.monotonic() - started,
                "status": status,
                "error": str(exc),
            },
        )
        (out / "decision.md").write_text(
            "# PV01 decision\n\n"
            f"- implementation_status: `{status}`\n"
            f"- error: `{exc!s}`\n"
            "- scientific_status: `INCONCLUSIVE`; no partial native metrics are promoted.\n"
            "- trigger: inspect the saved call accounting before any rerun; do not expand the budget.\n"
        )
        print(
            json.dumps(
                {"task_id": "PV01", "status": status, **counters, "error": str(exc)}
            )
        )
        if status == "FAIL":
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--max-model-calls", type=int, default=MAX_FORWARD)
    parser.add_argument(
        "--smoke", action="store_true", help="Use one source and two application pairs"
    )
    run(parser.parse_args())
