#!/usr/bin/env python3
"""G-4: Open-cohort Kitagawa-Oaxaca-Blinder turnover decomposition & strict causal PrefixHygiene.

Separates within-incumbent psychological/behavioral evolution from compositional cohort
turnover (entrant takeover + leaver selection) without leaking future corpus statistics.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from typing import Any

import numpy as np


class PrefixHygiene:
    """Strictly causal O(1) prefix hygiene filter (invariant to any future events t' > t)."""

    def __init__(
        self,
        *,
        burst_window: float = 60.0,
        max_burst: int = 5,
        max_dup_ratio: float = 0.5,
        dup_window_events: int = 10,
    ) -> None:
        if (
            burst_window <= 0
            or max_burst < 1
            or not (0.0 <= max_dup_ratio <= 1.0)
            or dup_window_events < 1
        ):
            raise ValueError("invalid PrefixHygiene thresholds")
        self.burst_window = float(burst_window)
        self.max_burst = int(max_burst)
        self.max_dup_ratio = float(max_dup_ratio)
        self.dup_window_events = int(dup_window_events)
        self._last_ts: float = -float("inf")
        self._author_counts: dict[str, int] = defaultdict(int)
        self._author_times: dict[str, deque[float]] = defaultdict(deque)
        self._author_hashes: dict[str, deque[str]] = defaultdict(deque)

    def observe(
        self, timestamp: float, author_id: Any, text_or_hash: str
    ) -> dict[str, Any]:
        ts = float(timestamp)
        if not np.isfinite(ts):
            raise ValueError("non-finite timestamp")
        if ts < self._last_ts:
            raise ValueError(
                "PrefixHygiene requires monotonically non-decreasing timestamps"
            )
        self._last_ts = ts
        au = str(author_id)
        h = hashlib.blake2b(text_or_hash.encode("utf-8"), digest_size=16).hexdigest()

        times_q = self._author_times[au]
        times_q.append(ts)
        while times_q and (ts - times_q[0]) > self.burst_window:
            times_q.popleft()
        burst_count = len(times_q)

        hash_q = self._author_hashes[au]
        hash_q.append(h)
        while len(hash_q) > self.dup_window_events:
            hash_q.popleft()
        n_win = len(hash_q)
        dup_ratio = (
            float(sum(1 for x in hash_q if x == h) - 1) / float(max(1, n_win - 1))
            if n_win > 1
            else 0.0
        )

        prior_count = self._author_counts[au]
        self._author_counts[au] = prior_count + 1
        is_clean = (burst_count <= self.max_burst) and (dup_ratio <= self.max_dup_ratio)
        return {
            "timestamp": ts,
            "prior_author_events": int(prior_count),
            "causal_burst_count": int(burst_count),
            "causal_dup_ratio": float(dup_ratio),
            "is_clean": bool(is_clean),
        }

    def filter_stream(
        self,
        timestamps: Sequence[float] | np.ndarray,
        author_ids: Sequence[Any],
        texts_or_hashes: Sequence[str],
    ) -> list[dict[str, Any]]:
        if not (len(timestamps) == len(author_ids) == len(texts_or_hashes)):
            raise ValueError("stream sequences must have equal length")
        return [
            self.observe(t, au, txt)
            for t, au, txt in zip(timestamps, author_ids, texts_or_hashes, strict=True)
        ]


def decompose_open_cohort_step(
    prev_author_means: dict[Any, np.ndarray],
    cur_author_means: dict[Any, np.ndarray],
    *,
    prev_weights: dict[Any, float] | None = None,
    cur_weights: dict[Any, float] | None = None,
) -> dict[str, Any]:
    """Exact Kitagawa-Oaxaca-Blinder turnover decomposition between two cohort snapshots."""
    if not prev_author_means or not cur_author_means:
        raise ValueError("both previous and current cohort snapshots must be non-empty")
    prev_ids = list(prev_author_means)
    cur_ids = list(cur_author_means)
    p_map = {k: np.asarray(v, dtype=np.float64) for k, v in prev_author_means.items()}
    c_map = {k: np.asarray(v, dtype=np.float64) for k, v in cur_author_means.items()}
    dim = next(iter(c_map.values())).shape[0]

    wp = {
        k: (1.0 if prev_weights is None else float(prev_weights[k])) for k in prev_ids
    }
    wc = {k: (1.0 if cur_weights is None else float(cur_weights[k])) for k in cur_ids}
    wp_tot = float(sum(wp.values()))
    wc_tot = float(sum(wc.values()))
    if any(not np.isfinite(v) or v <= 0 for v in (*wp.values(), *wc.values())) or any(
        z.shape != (dim,) or not np.isfinite(z).all()
        for z in (*p_map.values(), *c_map.values())
    ):
        raise ValueError("positive total weights required")

    stayers = [k for k in prev_ids if k in c_map]
    entrants = [k for k in cur_ids if k not in p_map]
    leavers = [k for k in prev_ids if k not in c_map]

    mu_prev = sum((wp[k] / wp_tot) * p_map[k] for k in prev_ids)
    mu_cur = sum((wc[k] / wc_tot) * c_map[k] for k in cur_ids)
    total_delta = mu_cur - mu_prev

    pi_E = sum(wc[k] for k in entrants) / wc_tot
    pi_L = sum(wp[k] for k in leavers) / wp_tot

    if stayers:
        ws_prev_tot = sum(wp[k] for k in stayers)
        ws_cur_tot = sum(wc[k] for k in stayers)
        ws_prev_n = {k: wp[k] / ws_prev_tot for k in stayers}
        ws_cur_n = {k: wc[k] / ws_cur_tot for k in stayers}
        mu_S_prev = sum(ws_prev_n[k] * p_map[k] for k in stayers)
        mu_S_cur = sum(ws_cur_n[k] * c_map[k] for k in stayers)

        within_incumbent_shift = sum(
            ws_prev_n[k] * (c_map[k] - p_map[k]) for k in stayers
        )
        stayer_weight_reallocation = sum(
            (ws_cur_n[k] - ws_prev_n[k]) * c_map[k] for k in stayers
        )
        stayer_evolution = mu_S_cur - mu_S_prev

        mu_E = (
            (sum(wc[k] * c_map[k] for k in entrants) / sum(wc[k] for k in entrants))
            if entrants
            else np.zeros(dim)
        )
        mu_L = (
            (sum(wp[k] * p_map[k] for k in leavers) / sum(wp[k] for k in leavers))
            if leavers
            else np.zeros(dim)
        )
        entrant_takeover = pi_E * (mu_E - mu_S_cur) if entrants else np.zeros(dim)
        leaver_selection = pi_L * (mu_S_prev - mu_L) if leavers else np.zeros(dim)
    else:
        within_incumbent_shift = np.zeros(dim)
        stayer_weight_reallocation = np.zeros(dim)
        stayer_evolution = np.zeros(dim)
        entrant_takeover = mu_cur
        leaver_selection = -mu_prev

    recon = (
        within_incumbent_shift
        + stayer_weight_reallocation
        + entrant_takeover
        + leaver_selection
    )
    identity_err = float(np.max(np.abs(total_delta - recon)))
    tot_norm = float(np.linalg.norm(total_delta))
    comp_norm = float(np.linalg.norm(entrant_takeover + leaver_selection))
    turnover_share = comp_norm / tot_norm if tot_norm > 1e-12 else 0.0

    return {
        "status": "decomposed" if stayers else "no_stayer_reference",
        "n_prev_authors": len(prev_ids),
        "n_cur_authors": len(cur_ids),
        "n_stayers": len(stayers),
        "n_entrants": len(entrants),
        "n_leavers": len(leavers),
        "entrant_mass": float(pi_E),
        "leaver_mass": float(pi_L),
        "total_delta": total_delta,
        "stayer_evolution": stayer_evolution,
        "within_incumbent_shift": within_incumbent_shift,
        "stayer_weight_reallocation": stayer_weight_reallocation,
        "entrant_takeover": entrant_takeover,
        "leaver_selection": leaver_selection,
        "turnover_to_total_norm_ratio": float(turnover_share),
        "identity_max_abs_err": identity_err,
    }


def decompose_open_cohort_panel(
    timestamps: Sequence[float] | np.ndarray,
    author_ids: Sequence[Any],
    condition_ids: Sequence[Any],
    coords: np.ndarray,
    *,
    window_edges: Sequence[float] | np.ndarray,
    weight_by_events: bool = True,
) -> list[dict[str, Any]]:
    """Apply exact Kitagawa-Oaxaca-Blinder step decomposition across sequential venue windows."""
    ts = np.asarray(timestamps, dtype=np.float64)
    c_arr = np.asarray(coords, dtype=np.float64)
    edges = np.asarray(window_edges, dtype=np.float64)
    win_idx = np.searchsorted(edges, ts, side="right") - 1
    n_windows = len(edges) - 1
    buckets: dict[tuple[int, str, str], list[np.ndarray]] = defaultdict(list)
    conds = sorted({str(c) for c in condition_ids})
    for i in range(len(ts)):
        w = int(win_idx[i])
        if 0 <= w < n_windows:
            buckets[(w, str(condition_ids[i]), str(author_ids[i]))].append(c_arr[i])

    out: list[dict[str, Any]] = []
    for co in conds:
        prev_means: dict[str, np.ndarray] | None = None
        prev_w: dict[str, float] | None = None
        for w in range(n_windows):
            cur_means = {}
            cur_w = {}
            for (wk, ck, au), vecs in buckets.items():
                if wk == w and ck == co:
                    cur_means[au] = np.mean(vecs, axis=0)
                    cur_w[au] = float(len(vecs)) if weight_by_events else 1.0
            if prev_means and cur_means:
                res = decompose_open_cohort_step(
                    prev_means, cur_means, prev_weights=prev_w, cur_weights=cur_w
                )
                out.append(
                    {
                        "window_index": w,
                        "window_start": float(edges[w]),
                        "window_end": float(edges[w + 1]),
                        "condition_id": co,
                        **res,
                    }
                )
            prev_means = cur_means if cur_means else None
            prev_w = cur_w if cur_means else None
    return out


__all__ = ["PrefixHygiene", "decompose_open_cohort_panel", "decompose_open_cohort_step"]


def decompose(before, after):
    """Mapping actor -> (positive exposure mass, observed mean) in each period."""

    def unpack(panel):
        if not panel or any(
            not np.isfinite([n, z]).all() or n <= 0 for n, z in panel.values()
        ):
            raise ValueError("positive observed masses and finite means required")
        N = sum(n for n, z in panel.values())
        return {u: (n / N, z) for u, (n, z) in panel.items()}

    a, b = unpack(before), unpack(after)
    common = a.keys() & b.keys()
    change = sum(w * z for w, z in b.values()) - sum(w * z for w, z in a.values())
    if not common:
        return {"status": "no_stayer_reference", "change": change}

    def stayer(p):
        s = sum(p[u][0] for u in common)
        w = {u: p[u][0] / s for u in common}
        return w, sum(w[u] * p[u][1] for u in common)

    wa, ma = stayer(a)
    wb, mb = stayer(b)
    within = sum((wa[u] + wb[u]) / 2 * (b[u][1] - a[u][1]) for u in common)
    reweight = sum((wb[u] - wa[u]) * (a[u][1] + b[u][1]) / 2 for u in common)
    entry = sum(b[u][0] * (b[u][1] - mb) for u in b.keys() - common)
    exit_ = sum(a[u][0] * (ma - a[u][1]) for u in a.keys() - common)
    return {
        "status": "decomposed",
        "change": change,
        "within": within,
        "stayer_reweight": reweight,
        "entry": entry,
        "exit": exit_,
        "closure": change - within - reweight - entry - exit_,
    }


class LifetimePrefixHygiene:
    """Before-arrival actor counts; ties follow declared ingestion order. All arrivals count."""

    def __init__(self, span=60.0, max_prior_repeats=2, max_prior_burst=10):
        if (
            not np.isfinite(span)
            or span <= 0
            or min(max_prior_repeats, max_prior_burst) < 1
        ):
            raise ValueError("invalid filter limits")
        self.span, self.repeats, self.burst = span, max_prior_repeats, max_prior_burst
        self.last = -np.inf
        self.by_actor = Counter()
        self.by_text = Counter()
        self.recent = defaultdict(deque)

    def observe(self, time, actor, text):
        if not np.isfinite(time) or time < self.last or not isinstance(text, str):
            raise ValueError("monotone times and text required")
        self.last = time
        key = actor, hashlib.blake2b(text.encode(), digest_size=16).digest()
        q = self.recent[actor]
        while q and q[0] <= time - self.span:
            q.popleft()
        n, k, b = self.by_actor[actor], self.by_text[key], len(q)
        self.by_actor[actor] += 1
        self.by_text[key] += 1
        q.append(time)
        return {
            "prior_count": n,
            "prior_identical": k,
            "prior_burst": b,
            "prior_share": k / n if n else None,
            "keep": k < self.repeats and b < self.burst,
        }
