# Two optional readers: event evidence and two-view increments

**English** | [简体中文](GRAIN_EVENT_AND_VIEW_INCREMENT_ZH.md) | [日本語](GRAIN_EVENT_AND_VIEW_INCREMENT_JA.md)

Run from the repository root. Both commands write JSON to stdout and leave
the native GRAIN score and its reference unchanged.

Install the repository dependencies with `python -m pip install -r requirements-lock-main.txt`
(Python 3.12 is the release reference). These examples require no network after installation.
The tool is on the current main branch, not the older v1.1.1 tag.

## Did the focal speaker adopt a corrected premise?

`python -m grain_core.event_response examples/event_response.json`

Supply the correction turn, old and new premise, focal speaker, and **inclusive,
zero-based** cutoff. An explicit English pattern reader distinguishes a focal
first-person choice from a mention or a question. It only handles an immediate
reply; an intervening partner turn can change the instruction. Inconclusive
speech and no observed reply are separate. Speech is not independent action.

The [input fixture](../examples/event_response.json) yields `events[0].status = "support"`.
The observed condition has weight 0.75 and the unobserved condition 0.25, so
`cards[0].reference_rate_bounds = [0.75, 1.0]`. Verdicts are `support`,
`refute`, and `insufficient`; inspect `opportunity` and `reason` before counting.
The JSON includes quotes: keep output local for confidential dialogue.

For broader language, a versioned external coder may supply a `correction`
with `confirmed` and `prefix_id` (the returned `trigger_prefix_id`), then a
`semantic` vote with its `version`, `actor` and the returned reply `prefix_id`.
Generate each annotation using only its prefix; a digest checks binding, not
reader independence. By default a semantic suggestion is only a candidate;
use `kind: "coded_response"` only for adjudicated behavior. Conflicting adopted
votes go to review. A `later_action` has source,
id, actor, exact choice, dialogue and `after_turn`; it appears only in a
separate verification record. It never changes an earlier judgment.
Optional `--liquid` takes the output of `grain_core.liquid_dialogue` for
the same input document; it exposes aligned slice readouts, not behavioral votes.

For a person's observed reply opportunities, `weights` defines the target
condition mix. In each condition, n episodes with s supports and f refutations
give completion bounds [s/n, 1-f/n]; no observations gives [0,1].
Unconfirmed triggers and unobserved replies have separate counts, outside
that estimand. These are not population confidence intervals or persona scores.
Repeated copies of an action are rejected. To assess a conjunction on one
action use Python `joined_opportunity(events, feasible_votes)` before aggregation.
The feasible rows must come from an external joint constraint, not independent
coin flips; with no consistent row, the case goes to review.

## When does two-view decomposition earn its cost?

`python -m grain_core.dual_view_increment examples/dual_view_increment.json`

Inputs must be in the same coordinates, with matched opportunity steps and
the same as-of cutoff. Stop or split at missing steps; the optional equal
sorted identifiers `dates1`, `dates2` verify alignment, not calendar adjacency.
Use lexically sortable labels (e.g. ISO dates). The [input fixture](../examples/dual_view_increment.json)
specifies `rates` and `budgets` in common, first-view, second-view order.
The three innovation bounds are nonnegative; `half_widths` has one nonnegative
entry per step and `c_bound` is a positive outer bound. `threshold` defaults to zero.
Without calibration, `assess` returns both levels, their midpoint and signed
gap. Those last two numbers together preserve both original levels exactly.
For a unit-loading uncorrelated-components moment calibration, pass training
pairs as `previous_levels`: nonpositive implied variance gives
`unsupported_calibration`. Positive sample moments are necessary checks, not
validation of independence. Pools with overlapping messages may violate it.

In the calibrated model, y_i = c + p_i. Per step, common state and private
states have rates (r_c, r_1, r_2) and absolute innovation budgets (D, E1, E2).
Supply a source calibration ID; obtain rates, budgets, midpoint half-widths,
and the outer box independently of evaluation outcomes. All four comparisons
use the same cutoff and bounds:

1. `static`: midpoint band on the last date, clipped to the agreed outer box.
2. `common_only`: propagate that band and the past bands using only the common law.
3. `simple_lag`: use both histories. For each view impose
   |(y_i' - r_i y_i) - (r_c - r_i)c| <= D + E_i; propagate and intersect the
   individual one-step ranges. This cheap baseline discards some joint constraints.
4. `full`: solve the existing trajectory LP on the prefix. Both explanations must
   use **the same** prior and subsequent common state at every step.

Read `vs_simple_lag`, not just the width relative to `static`.
`simple_lag_sufficient`: use the simpler answer.
`narrows_only`: retain the tighter range if width matters; threshold action unchanged.
`changes_threshold_decision`: a previously straddled, predeclared threshold is now excluded.
An incompatible model has no range: do not use a fitted midpoint or zero.
Ranges are conditional compatibility sets, not predictive probabilities.
New constraints can distinguish explanations of the same data without adding
observations. To claim predictive utility, compare against a both-view lag
baseline on the same subsequent dates, decision cutoffs and coverage.

The worked JSON has both views moving 0 to 1. Independent interval
propagation gives [-0.4,2]; requiring one common history gives [0.8,1.2].
A different pair of histories can share an endpoint and still be distinguished
by the simple lag contrast alone. A changed decision about c does not by itself
establish a changed decision about the next price.
