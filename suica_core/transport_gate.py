#!/usr/bin/env python3
"""G-6 (HM281): Condition-scoped Riccati stream, pre-arrival innovation, & TransportabilityGate.

Prevents historical negative transfer across incompatible regimes by scoping dynamics strictly
to (author, condition) keys, isolating pre-arrival forecast innovations, and enforcing a paired
author-clustered test against the fresh-start baseline before enabling history carry.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .riccati_mobius import ConstrainedRiccatiCell


class ConditionScopedRiccatiStream:
    """Strictly isolates Riccati-Mobius states per (author_id, condition_key)."""

    def __init__(
        self,
        cell: ConstrainedRiccatiCell,
        *,
        pulse_tau: float = 1.0,
        readout_matrix: np.ndarray | None = None,
    ) -> None:
        self.cell = cell
        self.pulse_tau = float(pulse_tau)
        if not np.isfinite(self.pulse_tau) or self.pulse_tau <= 0:
            raise ValueError("positive finite pulse required")
        if readout_matrix is None:
            # Deterministic orthonormal-scaled default map H -> U
            rng = np.random.default_rng(cell.seed + 17)
            self.readout_matrix = rng.normal(
                0.0,
                1.0 / np.sqrt(max(1, cell.hidden_dim)),
                (cell.hidden_dim, cell.input_dim),
            )
        else:
            W = np.asarray(readout_matrix, dtype=np.float64)
            if W.shape != (cell.hidden_dim, cell.input_dim) or not np.isfinite(W).all():
                raise ValueError("readout_matrix must match (hidden_dim, input_dim)")
            self.readout_matrix = W.copy()
        self._states: dict[tuple[str, str], tuple[float, np.ndarray]] = {}

    def step(
        self,
        author_id: Any,
        condition_key: Any,
        timestamp: float,
        u_vec: np.ndarray,
    ) -> dict[str, Any]:
        key = (str(author_id), str(condition_key))
        t_cur = float(timestamp)
        u_arr = np.asarray(u_vec, dtype=np.float64)
        if (
            not np.isfinite(t_cur)
            or u_arr.shape != (self.cell.input_dim,)
            or not np.isfinite(u_arr).all()
        ):
            raise ValueError("u_vec must match cell.input_dim")

        if key in self._states:
            t_prev, x_last = self._states[key]
            if t_cur < t_prev:
                raise ValueError(
                    "timestamps within (author, condition) stream must be non-decreasing"
                )
            dt_gap = t_cur - t_prev
            M_idle = self.cell.step_matrices(np.zeros(self.cell.input_dim), dt_gap)
            x_pre = self.cell.apply_mobius(M_idle, x_last)
            is_first = False
        else:
            dt_gap = 0.0
            x_last = np.zeros(self.cell.hidden_dim, dtype=np.float64)
            x_pre = x_last.copy()
            is_first = True

        u_hat = x_pre @ self.readout_matrix
        innovation = u_arr - u_hat
        M_step = self.cell.impulse_step_matrices(
            u_arr,
            dt_gap,
            pulse_tau=self.pulse_tau,
            subtract_pulse=False,
            is_first_step=is_first,
        )
        x_post = self.cell.apply_mobius(M_step, x_last)
        self._states[key] = (t_cur, x_post)
        return {
            "is_fresh_start": bool(is_first),
            "dt_gap": float(dt_gap),
            "pre_arrival_state": x_pre,
            "u_pred_pre": u_hat,
            "innovation": innovation,
            "post_arrival_state": x_post,
        }


def evaluate_transportability_gate(
    author_ids: Sequence[Any],
    loss_carry: Sequence[float] | np.ndarray,
    loss_fresh: Sequence[float] | np.ndarray,
    *,
    condition_ids: Sequence[Any] | None = None,
    alpha: float = 0.05,
    max_subcondition_harm: float = 0.01,
    n_boot: int = 1000,
    seed: int = 517,
) -> dict[str, Any]:
    """Author-clustered paired comparison vs Fresh-Start Null (HM281 gate)."""
    lc = np.asarray(loss_carry, dtype=np.float64)
    lf = np.asarray(loss_fresh, dtype=np.float64)
    if (
        lc.ndim != 1
        or not len(lc)
        or lf.shape != lc.shape
        or len(author_ids) != len(lc)
        or not 0 < alpha < 0.5
        or n_boot < 1
    ):
        raise ValueError(
            "author_ids, loss_carry, and loss_fresh must be aligned 1D arrays"
        )
    if not (np.isfinite(lc).all() and np.isfinite(lf).all()):
        raise ValueError("losses must be finite")

    gain = lf - lc  # positive => history carry beats fresh start
    obs_gain = float(np.mean(gain))

    # Group event sums/counts by author for fast exact author-clustered bootstrap
    au_order = sorted({str(a) for a in author_ids})
    au_to_idx = {a: i for i, a in enumerate(au_order)}
    n_au = len(au_order)
    au_sums = np.zeros(n_au, dtype=np.float64)
    au_cnts = np.zeros(n_au, dtype=np.float64)
    for au, g_val in zip(author_ids, gain, strict=True):
        idx = au_to_idx[str(au)]
        au_sums[idx] += float(g_val)
        au_cnts[idx] += 1.0

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_au, size=(n_boot, n_au))
    boot_gains = np.sum(au_sums[boot_idx], axis=1) / np.maximum(
        np.sum(au_cnts[boot_idx], axis=1), 1.0
    )
    ci_low = float(np.quantile(boot_gains, alpha))
    ci_high = float(np.quantile(boot_gains, 1.0 - alpha))

    worst_harm = 0.0
    sub_table: dict[str, float] = {}
    if condition_ids is not None:
        if len(condition_ids) != len(lc):
            raise ValueError("condition_ids must align with loss arrays")
        c_arr = np.asarray([str(c) for c in condition_ids])
        for co in sorted(set(c_arr.tolist())):
            mask = c_arr == co
            harm = float(np.mean(lc[mask] - lf[mask]))
            sub_table[co] = harm
            worst_harm = max(worst_harm, harm)
    else:
        worst_harm = float(-obs_gain)

    gate_open = bool(
        n_au >= 3 and ci_low > 0.0 and worst_harm <= float(max_subcondition_harm)
    )
    return {
        "gate_open": gate_open,
        "transport_weight": 1.0 if gate_open else 0.0,
        "mean_advantage_over_fresh": obs_gain,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "worst_subcondition_harm": float(worst_harm),
        "subcondition_harm": sub_table,
        "n_authors": n_au,
        "n_events": len(lc),
    }


__all__ = ["ConditionScopedRiccatiStream", "evaluate_transportability_gate"]
