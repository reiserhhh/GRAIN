#!/usr/bin/env python3
"""One SOURCE-qualified decoder-visible response experiment; no transformer calls."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_observer_operation as O
from grain_research.construction import math_output_observability as M
from grain_research.construction import math_source_anchor_exchange as E
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_conditional_memory_hm282 import status
from scripts.run_grain_joint_persona_style_hm150 import MODEL
from scripts.score_grain_author_belief_cold import load_bundle

CONFIG = "configs/grain_output_observability_hm284.json"
DEFAULT = ROOT / "results/grain_math_response_v1/output_observability_r1"
SPACES, BANKS, TARGETS = (
    ("E5", "GRAIN15"),
    ("bank0", "bank1", "pooled"),
    ("E", "N", "A", "C", "O"),
)
SCIENCE = (
    CONFIG,
    "docs/GRAIN_OUTPUT_OBSERVABILITY_PLAN_HM284.md",
    "grain_research/construction/math_output_observability.py",
    "grain_research/construction/math_criterion_components.py",
    "grain_research/construction/math_source_anchor_exchange.py",
    "scripts/run_grain_output_observability_hm284.py",
    "scripts/score_grain_output_observability_cold.py",
    "scripts/run_grain_conditional_memory_hm282.py",
    "tests/test_grain_output_observability.py",
)


def prepare(out):
    """Bind exact old source/event/query identities and retain complete hidden fields."""
    if out.exists():
        raise FileExistsError("No overwrite or scientific retry")
    cfg = R.read(ROOT / CONFIG)
    reader, criterion = ROOT / cfg["reader_parent"], ROOT / cfg["criterion_parent"]
    native = R.check(
        reader, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    R.check(criterion, ("computed_seal.json", "verification_seal.json"))
    data, inf = (
        load_bundle(criterion / "readings.npz"),
        load_bundle(criterion / "inference.npz"),
    )
    rows, bank, bindings = (
        R.read(reader / "units.json"),
        R.read(reader / "bank.json"),
        R.read(reader / "event_bindings.json"),
    )
    if (
        data["weights"].shape != (2, 3, 48, 2, 386)
        or inf["assignments"].shape != (5, 199, 386)
        or len(bindings) != 48
        or len(rows) != 386
    ):
        raise ValueError("Changed complete source/observer scope")
    if (
        not R.read(MODEL / "config.json")["tie_word_embeddings"]
        or R.digest(MODEL / "model.safetensors") != native["model_weights_sha256"]
    ):
        raise ValueError("Frozen tied output rows changed")
    for u in range(24):
        a, b = bindings[2 * u : 2 * u + 2]
        if (
            a["author"] != b["author"]
            or (a["window"], b["window"]) != (0, 1)
            or set(a["comment_ids"]) & set(b["comment_ids"])
        ):
            raise ValueError("Not disjoint pairs of the SAME author")
    h = np.empty((48, 2, 4, 50, 1536))
    logits, empty, empty_logits = (
        np.empty((48, 2, 4, 50, 2)),
        np.empty((2, 50, 1536)),
        np.empty((2, 50, 2)),
    )
    seen = set()
    for i, row in enumerate(rows):
        observed = load_bundle(reader / "batches" / f"{i:03d}.npz")
        op = O.OPERATIONS.index(row["operation"])
        key = (row["unit"], op, row["context"])
        if key in seen or observed["states"].shape != (50, 1536):
            raise ValueError("Duplicated or partial neural endpoint")
        seen.add(key)
        if row["unit"] < 0:
            empty[op], empty_logits[op] = observed["states"], observed["logits"]
        else:
            h[row["unit"], op, row["context"]] = observed["states"]
            logits[row["unit"], op, row["context"]] = observed["logits"]
    fields = M.joint_fields(h, bank)
    head = load_bundle(reader / "head.npz")
    vector = head["weights"][1].astype(float) - head["weights"][0]
    keyed = O.operation_accounts(logits, bank)["log_odds"]["balanced_key"]
    expected = np.stack([keyed["parts"][op]["join"] for op in O.OPERATIONS], axis=1)
    guard = float(abs(fields @ vector - expected).max())
    if guard > native["config"]["native_tolerance"]:
        raise ArithmeticError("Old adequate head tolerance fails; no precision rescue")
    files = {
        str((reader / p).relative_to(ROOT)): R.digest(reader / p)
        for p in (
            "manifest.json",
            "units.json",
            "bank.json",
            "event_bindings.json",
            "head.npz",
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        )
        + tuple(f"batches/{i:03d}.npz" for i in range(386))
    }
    files.update(
        {
            str((criterion / p).relative_to(ROOT)): R.digest(criterion / p)
            for p in (
                "manifest.json",
                "readings.npz",
                "inference.npz",
                "computed_seal.json",
                "verification_seal.json",
            )
        }
    )
    files[str((MODEL / "model.safetensors").relative_to(ROOT))] = native[
        "model_weights_sha256"
    ]
    files[str((MODEL / "config.json").relative_to(ROOT))] = R.digest(
        MODEL / "config.json"
    )
    out.mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": files,
            "alphabet": native["alphabet"],
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "fields.npz",
        joint=fields,
        empty=empty.reshape(100, 1536),
        empty_logits=empty_logits.reshape(100, 2),
        head_vector=vector,
        head_rows=head["weights"],
        head_bias=head["bias"],
        keyed_joint=expected,
    )
    R.archive(out / "source.npz", **data, assignments=inf["assignments"])
    R.write(out / "event_bindings.json", bindings)
    R.write(
        out / "quality.json",
        {
            "source_authors": 386,
            "panel_authors": 24,
            "pairs": 48,
            "items": 50,
            "hidden_fields": 480,
            "observed_empty_states": 100,
            "same_author_disjoint_pair_views": True,
            "keyed_head_replay_error": guard,
            "old_tolerance": native["config"]["native_tolerance"],
            "new_transformer_API_private_training": 0,
            "application_labels_read": False,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("fields.npz", "source.npz", "event_bindings.json", "quality.json"),
    )
    status(out, {"state": "PREPARED", "pid": None})
    print(
        "HM284 prepared: SAME48 events/480 joint fields/100 empty controls; no new transformer"
    )


def expired(_signal, _frame):
    """Stop at the finite full-vocabulary arithmetic budget, without a retry."""
    raise TimeoutError("Original300-second output geometry cap")


def output_geometry(out, cfg, alphabet):
    """Stream exact frozen BF16 rows as floats; never instantiate a language model."""
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    fields = load_bundle(out / "fields.npz")
    x, empty = fields["joint"].reshape(480, 1536), fields["empty"]
    vocab, block = cfg["vocabulary"], cfg["vocabulary_block"]
    logits = np.empty((100, vocab))
    with safe_open(
        MODEL / "model.safetensors", framework="pt", device="cpu"
    ) as archive:
        weights = archive.get_slice("model.embed_tokens.weight")
        if weights.get_shape() != [vocab, 1536] or "lm_head.bias" in set(
            archive.keys()
        ):
            raise ValueError("Pinned tied bias-free head required")
        for start in range(0, vocab, block):
            stop = min(start + block, vocab)
            w = weights[start:stop].float().numpy().astype(float)
            logits[:, start:stop] = empty @ w.T
        replay = float(abs(logits[:, alphabet] - fields["empty_logits"]).max())
        if replay > R.read(out / "quality.json")["old_tolerance"]:
            raise ArithmeticError("Full empty head differs from old adequate tolerance")
        probabilities = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True)).mean(
            0
        )
        del logits
        mean, second = np.zeros(480), np.zeros((480, 480))
        binary = np.empty((480, 2))
        spot_ids = np.array([0, 1, 4, 5, 20, 21, 240, 245, 470, 475])
        spot_logits = np.empty((len(spot_ids), vocab))
        for start in range(0, vocab, block):
            stop = min(start + block, vocab)
            w = weights[start:stop].float().numpy().astype(float)
            z, p = x @ w.T, probabilities[start:stop]
            second += (z * p) @ z.T
            mean += z @ p
            spot_logits[:, start:stop] = z[spot_ids]
            for a, token in enumerate(alphabet):
                if start <= token < stop:
                    binary[:, a] = z[:, token - start]
            if start // block % 8 == 0:
                print(f"HM284 output rows {stop}/{vocab}", flush=True)
    parts = M.fisher_from_sums(second, mean, probabilities[alphabet], binary)
    spot = M.fisher_parts(spot_logits, probabilities, alphabet)
    check = float(abs(parts[:, spot_ids[:, None], spot_ids[None, :]] - spot).max())
    if check > 1e-10 or abs(probabilities.sum() - 1) > 1e-10:
        raise ArithmeticError("Streamed/dense geometry differs")
    return {
        "grams": parts,
        "reference_probability": probabilities,
        "binary_logit_fields": binary,
        "spot_ids": spot_ids,
        "spot_logits": spot_logits,
    }, {
        "empty_head_error": replay,
        "stream_dense_error": check,
        "binary_reference_mass": float(probabilities[alphabet].sum()),
        "full_vocabulary_rows": vocab,
        "output_reference_states": 100,
        "joint_differential_fields": 480,
    }


def compute(out):
    """One fixed observation geometry, without choosing a metric or target direction."""
    cfg = R.check(out, ("prepared_seal.json",))["config"]
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No scientific rerun or automatic recovery")
    started = time.monotonic()
    status(out, {"state": "RUNNING", "pid": os.getpid()})
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(cfg["compute_seconds_cap"])
    try:
        arrays, info = output_geometry(
            out, cfg, R.read(out / "manifest.json")["alphabet"]
        )
        R.archive(out / "geometry.npz", **arrays)
        R.write(out / "geometry_quality.json", info)
        runtime = {
            "state": "COMPUTED",
            "pid": None,
            "seconds": time.monotonic() - started,
            "new_transformer_forwards": 0,
            "new_API_private_training": 0,
            "output_head_arithmetic_only": True,
            "original_scope_complete": False,
        }
        status(out, runtime)
        R.seal(
            out,
            "computed_seal.json",
            ("geometry.npz", "geometry_quality.json", "runtime.json"),
        )
        print(json.dumps(runtime), flush=True)
    except BaseException as error:
        status(
            out,
            {
                "state": "FAILED",
                "pid": None,
                "error": repr(error),
                "seconds": time.monotonic() - started,
                "no_retry": True,
            },
        )
        raise
    finally:
        signal.alarm(0)


def windows(data):
    """Match disjoint comments of the SAME24 authors, not48 independent people."""
    left, right = (
        np.flatnonzero(data["windows"] == 0),
        np.flatnonzero(data["windows"] == 1),
    )
    if (
        len(left) != 24
        or not np.array_equal(left, np.arange(0, 48, 2))
        or not np.array_equal(right, left + 1)
    ):
        raise ValueError("Changed panel pair/window alignment")
    return left, right


def analyze(out):
    """Qualify source targets with old conditional assignments; retain every crossing."""
    R.check(out, ("computed_seal.json",))
    source, geom = load_bundle(out / "source.npz"), load_bundle(out / "geometry.npz")
    left, right = windows(source)
    residual = source["labels"] - source["means"]
    rows, primary, saved = [], [], {}
    primary_p = []
    for sp, space in enumerate(SPACES):
        for b, bank in enumerate(BANKS):
            d = source["weights"][sp, b].mean(1)
            means = np.einsum("uj,jq->uq", d, source["means"])
            actual = np.einsum("uj,jq->uq", d, residual)
            for q in range(5):
                e = residual[source["assignments"][q], q]
                a_null, b_null = (
                    M.centered_scores(e @ d[left].T),
                    M.centered_scores(e @ d[right].T),
                )
                a, c = (
                    M.centered_scores(actual[left, q]),
                    M.centered_scores(actual[right, q]),
                )
                am, cm = (
                    M.centered_scores(means[left, q]),
                    M.centered_scores(means[right, q]),
                )
                covariance = C.orbit_covariance(
                    source["labels"][:, q], source["groups"][:, q]
                )
                for op, operation in enumerate(O.OPERATIONS):
                    for r, query in enumerate(TARGETS):
                        ids0, ids1 = left * 10 + op * 5 + r, right * 10 + op * 5 + r
                        kernels = geom["grams"][:, ids0[:, None], ids1[None, :]]
                        obs, null, exact = [], [], []
                        for ch, channel in enumerate(M.CHANNELS):
                            k = kernels[ch]
                            account = M.cross_author_accounts(a, c, k)
                            refs = M.cross_author_accounts(a_null, b_null, k)[
                                "distinct_authors"
                            ]
                            expected = M.orbit_mean(d[left], d[right], covariance, k)
                            rank = float(
                                (1 + np.sum(refs >= account["distinct_authors"])) / 200
                            )
                            obs.append(float(account["distinct_authors"]))
                            null.append(refs)
                            exact.append(expected)
                            shared = M.cross_author_accounts(am, cm, k)[
                                "distinct_authors"
                            ]
                            total = M.cross_author_accounts(a + am, c + cm, k)[
                                "distinct_authors"
                            ]
                            mixed = (
                                M.cross_author_accounts(am, c, k)["distinct_authors"]
                                + M.cross_author_accounts(a, cm, k)["distinct_authors"]
                            )
                            if (
                                abs(
                                    total - shared - account["distinct_authors"] - mixed
                                )
                                > 1e-12
                            ):
                                raise ArithmeticError(
                                    "Signed source component closure failed"
                                )
                            rows.append(
                                {
                                    "space": space,
                                    "bank": bank,
                                    "operation": operation,
                                    "source_target": TARGETS[q],
                                    "reader_query": query,
                                    "output_channel": channel,
                                    "within_distinct_author": float(
                                        account["distinct_authors"]
                                    ),
                                    "within_all_pair_product": float(
                                        account["all_pairs"]
                                    ),
                                    "within_same_author_diagonal": float(
                                        account["same_author"]
                                    ),
                                    "shared_distinct_author": float(shared),
                                    "total_distinct_author": float(total),
                                    "signed_source_mixed": float(mixed),
                                    "exact_assignment_mean": expected,
                                    "old_reference_mean": float(refs.mean()),
                                    "upper_conditional_rank": rank,
                                }
                            )
                            if (sp, b, op, q == r, ch) == (1, 2, 0, True, 0):
                                primary_p.append(rank)
                                primary.append(rows[-1].copy())
                        stem = f"{space}/{bank}/{operation}/{TARGETS[q]}/{query}"
                        saved[f"{stem}/observed"] = np.array(obs)
                        saved[f"{stem}/null"] = np.array(null)
                        saved[f"{stem}/exact_mean"] = np.array(exact)
    for row, adjusted in zip(primary, E.holm(primary_p), strict=True):
        row["Holm5_rank"] = float(adjusted)
    R.table(out / "all_contacts.csv", rows)
    R.table(out / "primary.csv", primary)
    R.archive(out / "inference.npz", **saved)
    d = source["weights"][1, 2].mean(1)
    scores = d @ residual
    cross = np.array(
        [
            geom["grams"][:, (left * 10 + r)[:, None], (right * 10 + r)[None, :]]
            for r in range(5)
        ]
    ).transpose(1, 0, 2, 3)
    bindings = R.read(out / "event_bindings.json")
    R.write(
        out / "cold_record.json",
        {
            "version": "HM284-v1",
            "author_keys": [f"panel-author/{i}" for i in range(24)],
            "left_event_ids": [
                [str(v) for v in bindings[i]["comment_ids"]] for i in left
            ],
            "right_event_ids": [
                [str(v) for v in bindings[i]["comment_ids"]] for i in right
            ],
            "source_readings": np.stack([scores[left], scores[right]]).tolist(),
            "cross_output_kernels": cross.tolist(),
        },
    )
    R.write(
        out / "summary.json",
        {
            "primary": primary,
            "geometry": R.read(out / "geometry_quality.json"),
            "runtime": R.read(out / "runtime.json"),
            "all_contact_rows": len(rows),
            "old_SOURCE_assignments": 995,
            "no_metric_fit_or_reference_draw_extension": True,
            "local_output_geometry_not_human_information_fraction_or_causal_mediation": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "all_contacts.csv",
            "primary.csv",
            "inference.npz",
            "summary.json",
            "cold_record.json",
        ),
    )
    print(json.dumps(R.read(out / "summary.json")), flush=True)


def verify(out):
    """Different arithmetic for vocabulary/account/reference checks, no new forwards."""
    R.check(out, ("computed_seal.json", "analyzed_seal.json"))
    source, geom, inf = (
        load_bundle(out / "source.npz"),
        load_bundle(out / "geometry.npz"),
        load_bundle(out / "inference.npz"),
    )
    left, right = windows(source)
    errors = {
        "output_channel": float(abs(geom["grams"][0] - geom["grams"][1:].sum(0)).max())
    }
    z, p = geom["spot_logits"], geom["reference_probability"]
    centered = z - (z @ p)[:, None]
    literal = (centered * p) @ centered.T
    ids = geom["spot_ids"]
    errors["centered_token_sum"] = float(
        abs(literal - geom["grams"][0][ids[:, None], ids[None, :]]).max()
    )
    residual = source["labels"] - source["means"]
    worst, checks = 0.0, 0
    d = source["weights"][1, 2].mean(1)
    for q in range(5):
        ix0, ix1 = left * 10 + q, right * 10 + q
        kernels = geom["grams"][:, ix0[:, None], ix1[None, :]]
        stem = f"GRAIN15/pooled/usual_author/{TARGETS[q]}/{TARGETS[q]}"
        for ref in [-1, *range(199)]:
            e = (
                residual[:, q]
                if ref < 0
                else residual[source["assignments"][q, ref], q]
            )
            a, b = d[left] @ e, d[right] @ e
            a, b = a - a.sum() / 24, b - b.sum() / 24
            for ch, k in enumerate(kernels):
                direct = sum(
                    float(a[i] * b[j] * k[i, j])
                    for i in range(24)
                    for j in range(24)
                    if i != j
                ) / (24 * 23)
                expected = (
                    inf[f"{stem}/observed"][ch]
                    if ref < 0
                    else inf[f"{stem}/null"][ch, ref]
                )
                worst = max(worst, abs(direct - expected))
                checks += 1
        for q_indices in source["assignments"][q]:
            if not np.array_equal(
                source["groups"][q_indices, q], source["groups"][:, q]
            ):
                raise ArithmeticError("Old SOURCE conditional assignment changed")
    errors["literal_primary_and_old_references"] = worst
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_output_observability_cold.py"),
            "--input",
            str(out / "cold_record.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cold = json.loads(run.stdout)
    values = np.asarray(cold["distinct_author_source_query_channels"])
    cold_error = max(
        float(
            abs(
                values[q, r]
                - inf[f"GRAIN15/pooled/usual_author/{TARGETS[q]}/{TARGETS[r]}/observed"]
            ).max()
        )
        for q in range(5)
        for r in range(5)
    )
    errors["cold_readback"] = cold_error
    if max(errors.values()) > 1e-10:
        raise ArithmeticError(json.dumps(errors))
    report = {
        "PASS": True,
        "errors": errors,
        "literal_primary_reference_channels": checks,
        "same_agent_not_blind": True,
        "new_transformer_API_private_training": 0,
        "original_scope_complete": False,
    }
    R.write(out / "verification.json", report)
    R.write(out / "cold_readback.json", cold)
    R.seal(out, "verification_seal.json", ("verification.json", "cold_readback.json"))
    print(json.dumps(report))


def main():
    """Run one immutable observable experiment, never renew its scientific budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "compute", "analyze", "verify", "check")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(
            args.output,
            ("computed_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM284 all seals PASS")
    else:
        globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
