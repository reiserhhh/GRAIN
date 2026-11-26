#!/usr/bin/env python3
"""G-3: Observed participation, activity elsewhere and bridge flux.

Implements Equation (1) of GRAIN Theory V1 strictly from anonymized event metadata:
(timestamp, author_id, condition_id, coord_vec). An absent event is scoped to the
observed feed; neither connectivity, intention nor missingness mechanism is observed.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


def compute_selective_silence_panel(
    timestamps: Sequence[float] | np.ndarray,
    author_ids: Sequence[Any],
    condition_ids: Sequence[Any],
    coords: np.ndarray,
    *,
    window_edges: Sequence[float] | np.ndarray,
    base_lookback_windows: int = 3,
    min_incumbent_events: int = 2,
    decay_half_life: float | None = None,
) -> list[dict[str, Any]]:
    """Describe the prior roster at each completed window using no later events."""
    ts = np.asarray(timestamps, dtype=np.float64)
    c_arr = np.asarray(coords, dtype=np.float64)
    edges = np.asarray(window_edges, dtype=np.float64)
    if ts.ndim != 1 or len(author_ids) != len(ts) or len(condition_ids) != len(ts):
        raise ValueError("timestamps, author_ids, and condition_ids must align")
    if c_arr.ndim != 2 or len(c_arr) != len(ts) or not np.isfinite(c_arr).all():
        raise ValueError("coords must be finite 2D array aligned with timestamps")
    if (
        not np.isfinite(ts).all()
        or not np.isfinite(edges).all()
        or edges.ndim != 1
        or len(edges) < 3
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(
            "window_edges must be strictly increasing with at least 2 windows"
        )
    if base_lookback_windows < 1 or min_incumbent_events < 1:
        raise ValueError("base_lookback_windows and min_incumbent_events must be >= 1")
    if decay_half_life is not None and (
        not np.isfinite(decay_half_life) or decay_half_life <= 0
    ):
        raise ValueError("decay_half_life must be finite and positive")

    n_windows = len(edges) - 1
    dim = c_arr.shape[1]
    known_conditions = set()
    win_idx = np.searchsorted(edges, ts, side="right") - 1

    # Bucket events by window -> author/condition
    win_author_total: list[dict[str, int]] = [
        defaultdict(int) for _ in range(n_windows)
    ]
    win_cond_total: list[dict[str, int]] = [defaultdict(int) for _ in range(n_windows)]
    win_cond_author_events: list[
        dict[str, dict[str, list[tuple[float, np.ndarray]]]]
    ] = [defaultdict(lambda: defaultdict(list)) for _ in range(n_windows)]
    for i in range(len(ts)):
        w = int(win_idx[i])
        if 0 <= w < n_windows:
            au = str(author_ids[i])
            co = str(condition_ids[i])
            win_author_total[w][au] += 1
            win_cond_total[w][co] += 1
            win_cond_author_events[w][co][au].append((float(ts[i]), c_arr[i]))

    rows: list[dict[str, Any]] = []
    for w in range(1, n_windows):
        current = win_cond_author_events[w]
        known_conditions.update(win_cond_author_events[w - 1])
        known_conditions.update(current)
        all_conditions = sorted(known_conditions)
        lb_start = max(0, w - base_lookback_windows)
        lb_wins = range(lb_start, w)
        n_lb = float(len(lb_wins))
        t_cut = float(edges[w])
        half_life = (
            float(decay_half_life)
            if decay_half_life is not None
            else max(1e-6, 0.5 * (t_cut - float(edges[lb_start])))
        )
        lam = math.log(2.0) / max(half_life, 1e-9)

        # Bridge authors active in >= 2 conditions during lookback
        lb_author_conds: dict[str, set[str]] = defaultdict(set)
        history: dict[str, dict[str, list[list[tuple[float, np.ndarray]]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for lw in lb_wins:
            for co, authors in win_cond_author_events[lw].items():
                for au, events in authors.items():
                    lb_author_conds[au].add(co)
                    history[co][au].append(events)
        bridge_authors = {au for au, cs in lb_author_conds.items() if len(cs) >= 2}
        # Count sparse observed pairs, not the cross product of authors and venues.
        prev_bridge_by_cond = {
            co: sum(
                len(evs)
                for au, bins in authors.items()
                if au in bridge_authors
                for evs in bins
            )
            for co, authors in history.items()
        }
        cur_bridge_by_cond = {
            co: sum(len(evs) for au, evs in authors.items() if au in bridge_authors)
            for co, authors in current.items()
        }
        prev_bridge_tot = sum(prev_bridge_by_cond.values())
        cur_bridge_tot = sum(cur_bridge_by_cond.values())

        w_rows: list[dict[str, Any]] = []
        raw_bridge_flux = []
        log_vols = []
        for co in all_conditions:
            cond_history = history.get(co, {})
            cond_current = current.get(co, {})
            p_weights: list[float] = []
            active_flags: list[float] = []
            online_silent_flags: list[float] = []
            offline_flags: list[float] = []
            x_pres: list[np.ndarray] = []

            for au in sorted(cond_history):
                bins = cond_history[au]
                ev_list = [event for evs in bins for event in evs]
                if len(ev_list) < min_incumbent_events:
                    continue
                p_uc = float(len(bins)) / n_lb
                dec_w = np.array(
                    [math.exp(-lam * max(0.0, t_cut - t_e)) for t_e, _ in ev_list],
                    dtype=float,
                )
                dec_w /= max(float(np.sum(dec_w)), 1e-12)
                x_pre = np.sum(
                    dec_w[:, None] * np.stack([vec for _, vec in ev_list]), axis=0
                )

                cur_in_c = len(cond_current.get(au, ()))
                cur_tot = win_author_total[w].get(au, 0)
                is_act = 1.0 if cur_in_c > 0 else 0.0
                is_on_sil = 1.0 if (cur_in_c == 0 and cur_tot > 0) else 0.0
                is_off = 1.0 if cur_tot == 0 else 0.0

                p_weights.append(p_uc)
                active_flags.append(is_act)
                online_silent_flags.append(is_on_sil)
                offline_flags.append(is_off)
                x_pres.append(x_pre)

            cur_vecs = [vec for evs in cond_current.values() for _, vec in evs]
            vis_mean = np.mean(cur_vecs, axis=0) if cur_vecs else np.full(dim, np.nan)
            n_inc = len(p_weights)
            if n_inc > 0:
                pw = np.asarray(p_weights, dtype=float)
                pw_sum = float(np.sum(pw))
                act_rate = float(np.sum(pw * np.asarray(active_flags)) / pw_sum)
                on_sil_rate = float(
                    np.sum(pw * np.asarray(online_silent_flags)) / pw_sum
                )
                off_rate = float(np.sum(pw * np.asarray(offline_flags)) / pw_sum)
                xp_mat = np.stack(x_pres)
                on_arr = np.asarray(online_silent_flags, dtype=float)
                act_arr = np.asarray(active_flags, dtype=float)
                asym_abs = np.sum((pw * on_arr)[:, None] * xp_mat, axis=0) / pw_sum
                asym_contrast = (
                    np.sum((pw * (on_arr - act_arr))[:, None] * xp_mat, axis=0) / pw_sum
                )
            else:
                act_rate = math.nan
                on_sil_rate = math.nan
                off_rate = math.nan
                asym_abs = np.full(dim, math.nan)
                asym_contrast = np.full(dim, math.nan)

            prev_sh = (
                prev_bridge_by_cond.get(co, 0) / prev_bridge_tot
                if prev_bridge_tot > 0
                else 0.0
            )
            cur_sh = (
                cur_bridge_by_cond.get(co, 0) / cur_bridge_tot
                if cur_bridge_tot > 0
                else 0.0
            )
            b_flux = cur_sh - prev_sh
            raw_bridge_flux.append(b_flux)
            log_vols.append(math.log1p(win_cond_total[w].get(co, 0)))

            w_rows.append(
                {
                    "window_index": w,
                    "window_start": float(edges[w]),
                    "window_end": float(edges[w + 1]),
                    "condition_id": co,
                    "n_events": int(win_cond_total[w].get(co, 0)),
                    "n_incumbents": int(n_inc),
                    "active_retention_rate": act_rate,
                    "online_avoid_rate": on_sil_rate,
                    "offline_absent_rate": off_rate,
                    "active_elsewhere_rate": on_sil_rate,
                    "not_observed_rate": off_rate,
                    "asym_signed_abstention": asym_abs,
                    "asym_abstention_contrast": asym_contrast,
                    "visible_mean": vis_mean,
                    "raw_bridge_share_shift": float(b_flux),
                    "log_volume": float(log_vols[-1]),
                }
            )

        # Orthogonalize bridge share shift against current log1p(volume) across venues in window w.
        y_bf = np.asarray(raw_bridge_flux, dtype=float)
        lv = np.asarray(log_vols, dtype=float)
        if not w_rows:
            continue
        if len(w_rows) >= 3 and float(np.std(lv)) > 1e-8:
            X = np.column_stack([np.ones(len(lv)), lv])
            beta, *_ = np.linalg.lstsq(X, y_bf, rcond=None)
            bf_orth = y_bf - X @ beta
        else:
            bf_orth = y_bf - float(np.mean(y_bf))
        for rec, val in zip(w_rows, bf_orth, strict=True):
            rec["volume_orthogonal_bridge_flux"] = float(val)
        rows.extend(w_rows)
    return rows


__all__ = ["compute_selective_silence_panel"]


def pool(events):
    """Rows are (actor, condition, signed scalar reading); one item is one arrival."""
    bank = defaultdict(list)
    for u, c, x in events:
        if not np.isfinite(x):
            raise ValueError("expected a finite observed reading")
        bank[u, c].append(float(x))
    return bank


def assess(history, current, target, *, decay=1.0, min_windows=1):
    """History is ordered equal-duration bins BEFORE current, including observed empty bins.

    One actor's target exposure weight is its past bin-level posting fraction.
    Caller must mark observation outages separately, never pass them as empty bins.
    """
    if not history or not 0 < decay <= 1 or min_windows < 1:
        raise ValueError("positive history and valid exposure rule required")
    weights = Counter()
    sums = Counter()
    support = Counter()
    shares = Counter()
    active_weight = Counter()
    denom = sum(decay**i for i in range(len(history)))
    elsewhere = Counter()
    for age, events in enumerate(reversed(history)):
        bank = pool(events)
        actors = Counter()
        for (u, c), vals in bank.items():
            actors[u] += len(vals)
            if c != target:
                elsewhere[u] += 1
        for u, total in actors.items():
            w = decay**age
            active_weight[u] += w
            vals = bank.get((u, target), [])
            shares[u] += w * len(vals) / total
            if vals:
                weights[u] += w
                sums[u] += w * np.mean(vals)
                support[u] += 1
    cur = pool(current)
    totals = Counter()
    for (u, c), vals in cur.items():
        totals[u] += len(vals)
    eligible = [u for u in weights if support[u] >= min_windows]
    W = sum(weights[u] / denom for u in eligible)
    channels = Counter()
    signed = 0.0
    bridge = []
    for u in eligible:
        wt = weights[u] / denom
        nt = len(cur.get((u, target), []))
        label = "present" if nt else "active_elsewhere" if totals[u] else "not_observed"
        channels[label] += wt
        if label == "active_elsewhere":
            signed += wt * sums[u] / weights[u]
        if totals[u] and elsewhere[u]:
            bridge.append(nt / totals[u] - shares[u] / active_weight[u])
    return {
        "eligible": len(eligible),
        "exposure_mass": W,
        **{
            k: channels[k] / W if W else None
            for k in ("present", "active_elsewhere", "not_observed")
        },
        "signed_elsewhere_flux": signed / W if W else None,
        "bridge_active": len(bridge),
        "bridge_share_shift": float(np.mean(bridge)) if bridge else None,
    }


def adjust_volume(source_shifts, source_volumes, shift, volume):
    """Freeze a source linear volume baseline; residuals needn't be orthogonal out of sample."""
    s, n = np.asarray(source_shifts, float), np.asarray(source_volumes, float)
    if (
        s.ndim != 1
        or s.shape != n.shape
        or len(s) < 3
        or np.any(n < 0)
        or volume < 0
        or not np.isfinite([s, n]).all()
        or not np.isfinite([volume, shift]).all()
        or np.ptp(n) == 0
    ):
        raise ValueError("distinct finite calibration volumes required")
    a, b = np.linalg.lstsq(
        np.column_stack([np.ones(len(n)), np.log1p(n)]), s, rcond=None
    )[0]
    return float(shift - a - b * np.log1p(volume))
