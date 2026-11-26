#!/usr/bin/env python3
"""Join observed evidence, collection coverage, and logged opportunities without imputation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any

import numpy as np

from .open_cohort import decompose_open_cohort_step

VERSION = "grain.evidence-panel.v1"


def _key(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a non-empty string identifier")
    return value


def _flag(value: bool | None) -> bool | None:
    if value is not None and type(value) is not bool:
        raise ValueError("expected true, false, or null")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    id: str
    window: str
    actor: str
    context: str
    channel: str
    values: Sequence[float] | None = None
    observation_id: str | None = None


@dataclass(frozen=True, slots=True)
class Coverage:
    window: str
    complete: bool
    context: str | None = None  # None is the entire declared feed.


@dataclass(frozen=True, slots=True)
class Opportunity:
    id: str
    window: str
    actor: str
    context: str
    measure: str
    eligible: bool | None
    outcome: bool | None


Record = EvidenceEvent | Coverage | Opportunity


def _values(values, width):
    if values is None:
        return None
    if (
        width == 1
        and type(values) in (list, tuple)
        and len(values) == 1
        and type(values[0]) in (int, float)
    ):
        x = float(values[0])
        if math.isfinite(x):
            return x
    else:
        x = np.asarray(values, dtype=float)
        if x.shape == (width,) and np.isfinite(x).all():
            return float(x[0]) if width == 1 else x
    raise ValueError("values must be a finite vector matching the channel")


@dataclass(slots=True)
class _Cell:
    total: float | np.ndarray | None = None
    reports: int = 0
    scored: int = 0
    _mean: np.ndarray | None = None

    def add(self, x: float | np.ndarray | None) -> None:
        self.reports += 1
        if x is not None:
            self.scored += 1
            self._mean = None
            if self.total is None:
                self.total = x if isinstance(x, float) else x.copy()
            else:
                self.total += x
                finite = (
                    math.isfinite(self.total)
                    if isinstance(self.total, float)
                    else np.isfinite(self.total).all()
                )
                if not finite:
                    raise ValueError("non-finite evidence sum")

    def mean(self) -> np.ndarray:
        assert self.total is not None
        if self._mean is None:
            self._mean = np.atleast_1d(self.total / self.scored)
        return self._mean


class EvidencePanel:
    """A read-only index of named coordinate channels and independently logged actions.

    Window IDs need not be timestamps. The caller declares which windows are
    comparable, and supplies the precise prior history for participation queries.
    """

    def __init__(
        self,
        channels: Mapping[str, Sequence[str]],
        events: Iterable[EvidenceEvent],
        *,
        coverage: Iterable[Coverage] = (),
        opportunities: Iterable[Opportunity] = (),
    ) -> None:
        self._index(channels, chain(coverage, events, opportunities))

    @classmethod
    def from_records(
        cls, channels: Mapping[str, Sequence[str]], records: Iterable[Record]
    ) -> EvidencePanel:
        """Consume a mixed record stream once. Validate before exposing the index."""
        panel = cls.__new__(cls)
        panel._index(channels, records)
        return panel

    def _index(self, channels, records):
        self._channels: dict[str, tuple[str, ...]] = {}
        for channel, names in channels.items():
            if isinstance(names, str):
                raise TypeError("coordinate names must be an ordered sequence")
            names = tuple(map(_key, names))
            if not names or len(set(names)) != len(names):
                raise ValueError("coordinate names must be non-empty and unique")
            self._channels[_key(channel)] = names
        self._coverage: dict[tuple[str, str | None], bool] = {}
        self._activity: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self._cells: dict[tuple[str, str, str], dict[str, _Cell]] = defaultdict(dict)
        self._choices: dict[tuple[str, str, str], dict[str, Counter[str]]] = (
            defaultdict(dict)
        )
        self._choice_totals: dict[tuple[str, str, str], Counter[str]] = defaultdict(
            Counter
        )
        self._by_context: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._record_counts: Counter[str] = Counter()
        addresses: dict[str, tuple[str, str, str, str]] = {}
        other_channels: set[tuple[str, str]] = set()
        seen = set()
        seen_opportunities = set()
        for row in records:
            if isinstance(row, EvidenceEvent):
                if _key(row.id) in seen:
                    raise ValueError("duplicate evidence id")
                seen.add(row.id)
                names = self._channels.get(row.channel)
                if names is None:
                    raise ValueError("undeclared evidence channel")
                win, actor, ctx = map(_key, (row.window, row.actor, row.context))
                x = _values(row.values, len(names))
                obs = row.id if row.observation_id is None else _key(row.observation_id)
                prior = addresses.get(obs)
                if prior is None:
                    addresses[obs] = (win, actor, ctx, row.channel)
                    self._activity[win, actor][ctx] += 1
                    self._by_context[win, ctx].add(actor)
                else:
                    if prior[:3] != (win, actor, ctx):
                        raise ValueError(
                            "observation id reused at an inconsistent address"
                        )
                    address = (obs, row.channel)
                    if prior[3] == row.channel or address in other_channels:
                        raise ValueError("duplicate channel for one observation")
                    other_channels.add(address)
                people = self._cells[win, ctx, row.channel]
                cell = people.get(actor)
                if cell is None:
                    cell = people[actor] = _Cell()
                cell.add(x)
                self._record_counts["events"] += 1
            elif isinstance(row, Coverage):
                key = (
                    _key(row.window),
                    _key(row.context) if row.context is not None else None,
                )
                if type(row.complete) is not bool or key in self._coverage:
                    raise ValueError(
                        "coverage must be explicit and declared at most once per scope"
                    )
                self._coverage[key] = row.complete
                self._record_counts["coverage"] += 1
            elif isinstance(row, Opportunity):
                if _key(row.id) in seen_opportunities:
                    raise ValueError("duplicate opportunity id")
                seen_opportunities.add(row.id)
                win, actor, ctx, measure = map(
                    _key, (row.window, row.actor, row.context, row.measure)
                )
                eligible, outcome = _flag(row.eligible), _flag(row.outcome)
                if eligible is not True and outcome is not None:
                    raise ValueError("an outcome requires a known eligible opportunity")
                label = (
                    (
                        "yes"
                        if outcome is True
                        else "no"
                        if outcome is False
                        else "outcome_unknown"
                    )
                    if eligible is True
                    else "ineligible"
                    if eligible is False
                    else "eligibility_unknown"
                )
                people = self._choices[win, ctx, measure]
                cell = people.get(actor)
                if cell is None:
                    cell = people[actor] = Counter()
                cell[label] += 1
                self._choice_totals[win, ctx, measure][label] += 1
                self._record_counts["opportunities"] += 1
            else:
                raise TypeError("unrecognized evidence record type")
        if any(
            context is not None and not val and self._coverage.get((win, None)) is True
            for (win, context), val in self._coverage.items()
        ):
            raise ValueError("complete feed conflicts with incomplete context coverage")

    def diagnostics(self) -> dict[str, Any]:
        """Report how much data and collection coverage actually reached the index."""
        channels = {k: {"records": 0, "scored": 0} for k in self._channels}
        for (_, _, name), people in self._cells.items():
            for c in people.values():
                channels[name]["records"] += c.reports
                channels[name]["scored"] += c.scored

        return {
            "records": {
                k: self._record_counts[k]
                for k in ("events", "opportunities", "coverage")
            },
            "unique_observations": sum(
                sum(x.values()) for x in self._activity.values()
            ),
            "channels": channels,
            "complete_feed_windows": sum(
                c is None and v for (_, c), v in self._coverage.items()
            ),
            "complete_context_windows": sum(
                c is not None and v for (_, c), v in self._coverage.items()
            ),
            "observed_context_windows_without_complete_coverage": sum(
                self._coverage.get((w, c)) is not True
                and self._coverage.get((w, None)) is not True
                for w, c in self._by_context
            ),
        }

    def participation(
        self, window: str, actor: str, target: str, *, history: Sequence[str] = ()
    ) -> dict[str, Any]:
        """Presence is positive evidence; local absence requires target coverage."""
        _key(window)
        _key(actor)
        _key(target)
        if (
            isinstance(history, str)
            or window in history
            or len(set(history)) != len(history)
        ):
            raise ValueError("history must contain distinct prior window ids")
        feed = self._coverage.get((window, None))
        focal = True if feed is True else self._coverage.get((window, target))
        complete_feed, complete_target = feed is True, focal is True
        activity = self._activity.get((window, actor), {})
        prior = {
            ctx for win in history for ctx in self._activity.get((_key(win), actor), {})
        }
        elsewhere = [
            {"context": ctx, "records": n, "previously_observed": ctx in prior}
            for ctx, n in sorted(activity.items())
            if ctx != target
        ]
        if target in activity:
            state = "observed_at_target"
        elif not complete_target:
            state = "target_coverage_unknown"
        elif elsewhere:
            state = "observed_elsewhere_only"
        elif complete_feed:
            state = "not_observed_in_feed"
        else:
            state = "not_observed_at_target_elsewhere_unknown"
        return {
            "state": state,
            "target_records": activity.get(target, 0),
            "target_complete": complete_target,
            "feed_complete": complete_feed,
            "target_coverage": "unknown"
            if focal is None
            else "complete"
            if focal
            else "incomplete",
            "feed_coverage": "unknown"
            if feed is None
            else "complete"
            if feed
            else "incomplete",
            "elsewhere": elsewhere,
            "target_previously_observed": target in prior,
        }

    def compare(
        self,
        before: str,
        after: str,
        context: str,
        channel: str,
        *,
        weighting: str = "actor",
    ) -> dict[str, Any]:
        """Decompose the observed scored cohort, never the imputed population."""
        if before == after or weighting not in ("actor", "event"):
            raise ValueError("use distinct windows and actor or event weighting")
        if channel not in self._channels:
            raise ValueError("undeclared evidence channel")
        ctx = _key(context)
        sides = [self._cells.get((_key(w), ctx, channel), {}) for w in (before, after)]
        means = [{a: c.mean() for a, c in side.items() if c.scored} for side in sides]
        weights = [{a: c.scored for a, c in side.items() if c.scored} for side in sides]
        support = [
            {
                "window": w,
                "context_complete": (
                    self._coverage.get((w, None)) is True
                    or self._coverage.get((w, ctx)) is True
                ),
                "actors_with_evidence": len(side),
                "scored_actors": len(mean),
                "records": sum(c.reports for c in side.values()),
                "scored_records": sum(count.values()),
            }
            for w, side, mean, count in zip(
                (before, after), sides, means, weights, strict=True
            )
        ]
        out = {
            "context": ctx,
            "channel": channel,
            "coordinates": self._channels[channel],
            "weighting": weighting,
            "support": support,
        }
        if not all(means):
            return dict(out, status="insufficient_scored_evidence", decomposition=None)
        d = decompose_open_cohort_step(
            *means,
            prev_weights=weights[0] if weighting == "event" else None,
            cur_weights=weights[1] if weighting == "event" else None,
        )
        if not d["n_stayers"]:
            for term in (
                "within_incumbent_shift",
                "stayer_weight_reallocation",
                "stayer_evolution",
            ):
                d[term] = None
        return dict(
            out,
            status=d["status"],
            decomposition={
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()
            },
        )

    def participation_flow(
        self, before: str, after: str, target: str, *, history: Sequence[str] = ()
    ) -> dict[str, Any]:
        """Track the observed prior roster, tallying destination links once per actor."""
        if (
            before == after
            or isinstance(history, str)
            or before in history
            or after in history
            or len(set(history)) != len(history)
        ):
            raise ValueError(
                "use two distinct windows and additional distinct prior window ids"
            )
        before, after, target = map(_key, (before, after, target))
        windows = (before, *map(_key, history))
        a = self._by_context.get((before, target), set())
        b = self._by_context.get((after, target), set())
        states: Counter[str] = Counter()
        destinations: dict[str, Counter[str]] = defaultdict(Counter)
        for actor in a:
            p = self.participation(after, actor, target, history=windows)
            states[p["state"]] += 1
            if not p["state"].startswith("observed_at_target"):
                for node in p["elsewhere"]:
                    k = (
                        "previously_observed"
                        if node["previously_observed"]
                        else "not_in_observed_history"
                    )
                    destinations[node["context"]][k] += 1
                    destinations[node["context"]]["target_coverage_unconfirmed"] += int(
                        not p["target_complete"]
                    )
        return {
            "before": before,
            "after": after,
            "target": target,
            "prior_actors": len(a),
            "current_actors": len(b),
            "not_in_prior_observed_roster": len(b - a),
            "prior_target_complete": self._coverage.get((before, None)) is True
            or self._coverage.get((before, target)) is True,
            "current_target_complete": self._coverage.get((after, None)) is True
            or self._coverage.get((after, target)) is True,
            "prior_roster_current_states": dict(sorted(states.items())),
            "destinations_without_target_records": {
                k: dict(v) for k, v in sorted(destinations.items())
            },
        }

    def context_shift(
        self,
        before: str,
        after: str,
        actor: str,
        origin: str,
        destination: str,
        channel: str,
    ) -> dict[str, Any]:
        """Split a same-person cross-context contrast without imputing the third cell."""
        if before == after or origin == destination:
            raise ValueError("context shift needs two windows and two contexts")
        if channel not in self._channels:
            raise ValueError("undeclared evidence channel")
        _key(actor)
        address = {
            "origin_before": (before, origin),
            "destination_before": (before, destination),
            "destination_after": (after, destination),
        }
        support = {}
        points = {}
        for k, (win, ctx) in address.items():
            c = self._cells.get((_key(win), _key(ctx), channel), {}).get(actor)
            m = c.mean() if c is not None and c.scored else None
            points[k] = m
            support[k] = {
                "window": win,
                "context": ctx,
                "records": c.reports if c else 0,
                "scored_records": c.scored if c else 0,
                "mean": m.tolist() if m is not None else None,
            }

        def difference(high, low):
            x, y = points[high], points[low]
            return (x - y).tolist() if x is not None and y is not None else None

        return {
            "channel": channel,
            "coordinates": self._channels[channel],
            "actor": actor,
            "status": "decomposed"
            if all(v is not None for v in points.values())
            else "incomplete_support",
            "support": support,
            "total": difference("destination_after", "origin_before"),
            "context_at_baseline": difference("destination_before", "origin_before"),
            "within_destination": difference("destination_after", "destination_before"),
        }

    def opportunity_card(
        self, window: str, context: str, measure: str, *, actor: str | None = None
    ) -> dict[str, Any]:
        """Bounds concern the logged opportunity ledger, not unseen opportunities."""
        key = tuple(map(_key, (window, context, measure)))
        counts = (
            self._choice_totals.get(key, Counter())
            if actor is None
            else self._choices.get(key, {}).get(_key(actor), Counter())
        )
        return self._opportunity_summary(counts)

    @staticmethod
    def _opportunity_summary(counts):
        yes, no, oc, ec, excluded = (
            counts[k]
            for k in (
                "yes",
                "no",
                "outcome_unknown",
                "eligibility_unknown",
                "ineligible",
            )
        )
        n = yes + no + oc
        upper_n = n + ec
        return {
            "yes": yes,
            "no": no,
            "outcome_unknown": oc,
            "eligible": n,
            "eligibility_unknown": ec,
            "ineligible": excluded,
            "known_eligible_bounds": [yes / n, (yes + oc) / n] if n else None,
            "all_logged_bounds": [yes / upper_n, (yes + oc + ec) / upper_n]
            if upper_n
            else None,
        }

    def paired_opportunities(
        self,
        before: str,
        after: str,
        origin: str,
        destination: str,
        measure: str,
        *,
        channel: str | None = None,
    ) -> dict[str, Any]:
        """Compare two logged menus for the same people, retaining incomplete outcomes."""
        before, after, origin, destination, measure = map(
            _key, (before, after, origin, destination, measure)
        )
        if (before, origin) == (after, destination):
            raise ValueError("opportunity pairs require distinct cells")
        if channel is not None and channel not in self._channels:
            raise ValueError("undeclared evidence channel")
        a, b = (
            self._choices.get((w, c, measure), {})
            for w, c in ((before, origin), (after, destination))
        )
        matched = sorted(a.keys() & b.keys())
        pairs = []
        linked = []
        shifts = []
        for u in matched:
            x, y = self._opportunity_summary(a[u]), self._opportunity_summary(b[u])
            if x["eligible"] == 0 or y["eligible"] == 0:
                continue
            bounds = []
            for k in ("known_eligible_bounds", "all_logged_bounds"):
                lo_x, hi_x = x[k]
                lo_y, hi_y = y[k]
                bounds.append((lo_y - hi_x, hi_y - lo_x))
            pairs.append(bounds)
            if channel is not None:
                ca = self._cells.get((before, origin, channel), {}).get(u)
                cb = self._cells.get((after, destination, channel), {}).get(u)
                if ca is not None and cb is not None and ca.scored and cb.scored:
                    linked.append(bounds)
                    shifts.append(cb.mean() - ca.mean())

        def reduce(rows):
            mean = np.mean(rows, axis=0).tolist() if rows else [None, None]
            return {
                "actors": len(rows),
                "known_eligible_delta_bounds": mean[0],
                "all_logged_delta_bounds": mean[1],
            }

        out = {
            "measure": measure,
            "before": [before, origin],
            "after": [after, destination],
            "before_logged_actors": len(a),
            "after_logged_actors": len(b),
            "paired_logged_actors": len(matched),
            "paired_known_eligible": reduce(pairs),
        }
        if channel is not None:
            out["with_paired_coordinates"] = {
                "channel": channel,
                "coordinates": self._channels[channel],
                **reduce(linked),
                "mean_coordinate_shift": np.mean(shifts, axis=0).tolist()
                if shifts
                else None,
            }
        return out


def run_panel(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run ordered, ID-addressed queries in the versioned interchange form."""
    if payload.get("schema_version") != VERSION:
        raise ValueError(f"expected schema_version {VERSION}")
    if payload.keys() - {
        "schema_version",
        "channels",
        "events",
        "coverage",
        "opportunities",
        "queries",
    }:
        raise ValueError("unknown evidence panel field")
    panel = EvidencePanel(
        payload.get("channels", {}),
        (EvidenceEvent(**row) for row in payload.get("events", [])),
        coverage=(Coverage(**row) for row in payload.get("coverage", [])),
        opportunities=(Opportunity(**row) for row in payload.get("opportunities", [])),
    )
    return run_queries(panel, payload.get("queries", []))


def run_queries(
    panel: EvidencePanel, queries: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Reuse an existing index for multiple versioned query batches."""
    operations = {
        "participation": panel.participation,
        "compare": panel.compare,
        "participation_flow": panel.participation_flow,
        "context_shift": panel.context_shift,
        "opportunity_card": panel.opportunity_card,
        "paired_opportunities": panel.paired_opportunities,
    }
    results = {}
    for row in queries:
        args = dict(row)
        name, op = _key(args.pop("id")), args.pop("op")
        if name in results or op not in operations:
            raise ValueError("duplicate query id or unknown operation")
        results[name] = operations[op](**args)
    return {"schema_version": VERSION, "results": results}
