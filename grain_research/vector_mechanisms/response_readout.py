"""Exact softmax readout geometry; no learned personality directions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .response_context import paired_interval


def decompose_logits(actual, matched, targets):
    """Compute exact target, normalization, gradient and KL terms in float64."""
    a, m = np.asarray(actual, dtype=np.float64), np.asarray(matched, dtype=np.float64)
    y = np.asarray(targets)
    if (
        a.ndim != 2
        or a.shape != m.shape
        or not a.size
        or y.shape != (len(a),)
        or not np.issubdtype(y.dtype, np.integer)
        or (y < 0).any()
        or (y >= a.shape[1]).any()
        or not np.isfinite(a).all()
        or not np.isfinite(m).all()
    ):
        raise ValueError("Finite paired logits and valid integer targets required")
    aa, am = logsumexp(a, axis=1), logsumexp(m, axis=1)
    la, lm = a - aa[:, None], m - am[:, None]
    pa, pm = np.exp(la), np.exp(lm)
    dz = a - m
    ix = np.arange(len(a))
    target = dz[ix, y]
    em, ea = np.sum(pm * dz, axis=1), np.sum(pa * dz, axis=1)
    km = np.sum(pm * (lm - la), axis=1)
    ka = np.sum(pa * (la - lm), axis=1)
    curvature = np.sum(pm * (dz - em[:, None]) ** 2, axis=1)
    result = {
        "actual_logp": la[ix, y],
        "matched_logp": lm[ix, y],
        "delta_logp": la[ix, y] - lm[ix, y],
        "target_logit_shift": target,
        "log_partition_shift": aa - am,
        "matched_gradient_dot_delta": target - em,
        "actual_gradient_dot_delta": target - ea,
        "kl_matched_to_actual": km,
        "kl_actual_to_matched": ka,
        "matched_directional_curvature": curvature,
    }
    errors = [
        result["delta_logp"] - (target - aa + am),
        result["delta_logp"] - (target - em - km),
        result["delta_logp"] - (target - ea + ka),
        km - (aa - am - em),
        ka - (am - aa + ea),
    ]
    error = max(float(abs(e).max()) for e in errors)
    if (
        error > 1e-9
        or min(float(km.min()), float(ka.min()), float(curvature.min())) < -1e-9
    ):
        raise ValueError("Softmax/KL identity or positivity failed")
    return result, error


def state_geometry(actual, matched):
    """Separate positionwise distances from cosine of pooled predictor states."""
    a, m = np.asarray(actual, dtype=np.float64), np.asarray(matched, dtype=np.float64)
    if (
        a.ndim != 2
        or a.shape != m.shape
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(m).all()
    ):
        raise ValueError("Finite paired state matrices required")
    na, nm = np.linalg.norm(a, axis=1), np.linalg.norm(m, axis=1)
    ma, mm = a.mean(axis=0), m.mean(axis=0)
    denominator = np.linalg.norm(ma) * np.linalg.norm(mm)
    if (na <= 0).any() or (nm <= 0).any() or denominator <= 0:
        raise ValueError("Zero states have undefined angular geometry")
    return {
        "state_cosine": np.clip(np.sum(a * m, axis=1) / (na * nm), -1, 1),
        "state_displacement_norm": np.linalg.norm(a - m, axis=1),
    }, float(np.clip(np.dot(ma, mm) / denominator, -1, 1))


def pair_readout(model, actual, matched, body):
    """Replay the LM head in bounded batches and return only coded scalar rows."""
    import torch

    ha = actual["trace"]["final_norm"]["states"][:-1]
    hm = matched["trace"]["final_norm"]["states"][:-1]
    if len(ha) != len(body) - 1 or len(ha) < 1:
        raise ValueError("Off-by-one in tail predictor alignment")
    geometry, mean_cos = state_geometry(ha, hm)
    rows, algebra_error, replay_error, batches = [], 0.0, 0.0, 0
    device = next(model.parameters()).device
    for start in range(0, len(ha), 8):
        stop = min(start + 8, len(ha))
        with torch.inference_mode():
            za = (
                model.lm_head(torch.as_tensor(ha[start:stop], device=device))
                .float()
                .cpu()
                .numpy()
            )
            zm = (
                model.lm_head(torch.as_tensor(hm[start:stop], device=device))
                .float()
                .cpu()
                .numpy()
            )
        batches += 2
        values, error = decompose_logits(za, zm, body[start + 1 : stop + 1])
        algebra_error = max(algebra_error, error)
        for role, native in (("actual", actual), ("matched", matched)):
            replay_error = max(
                replay_error,
                float(
                    abs(
                        values[role + "_logp"]
                        - native["token_logp"][start + 1 : stop + 1]
                    ).max()
                ),
            )
        for j in range(stop - start):
            rows.append(
                {
                    "response_token_position": start + j + 2,
                    **{k: float(v[j]) for k, v in values.items()},
                    **{k: float(v[start + j]) for k, v in geometry.items()},
                }
            )
        del za, zm
    if replay_error > 1e-5:
        raise ValueError(
            f"Independent final-head token logp replay failed: {replay_error}"
        )
    return rows, {
        "head_batches": batches,
        "algebra_max_error": algebra_error,
        "head_replay_max_error": replay_error,
        "mean_predictor_state_cosine": mean_cos,
    }


def summarize(frame, checks):
    """Equal-person summaries, not token-as-person inference or a new score."""
    columns = [
        x
        for x in frame.columns
        if x not in ("participant_code", "response_token_position")
    ]
    persons = frame.groupby("participant_code", sort=True)[columns].mean()
    extra = pd.DataFrame(checks).set_index("participant_code")
    if extra.index.has_duplicates or set(extra.index) != set(persons.index):
        raise ValueError("Participant summary coverage mismatch")
    persons["mean_predictor_state_cosine"] = extra.mean_predictor_state_cosine
    return persons, {k: paired_interval(persons[k]) for k in persons}
