# Conditional operator: finite construction test and remainder

2026-09-15. Codex. Finite research packet COMPLETE; compiled operator NOT promoted.
Protocol: docs/GRAIN_CONDITIONAL_OPERATOR_GOAL_20260915.md.
Result-triggered extension: docs/GRAIN_OPERATOR_REMAINDER_FOLLOWUP_20260915.md.
Outputs: results/grain_vector_mechanisms/conditional_operator_v1/r2/.

## Discovery and decision

The reader has a usable native expression-to-score action that the current four
GRAIN coordinates plus fixed low-rank estimator did not recover across conditions.
The uncompiled remainder retains native action; its interaction with the compiled
part is small in this finite intervention. This locates a constructive next
question: learn a representation/loss that preserves measurement action, instead
of assuming a population-derived four-coordinate geometry already preserves it.

This does NOT distinguish insufficient coordinates from a misspecified linear
map, rank-2 restriction or hidden-state Euclidean loss. It is not a rejection of
the complete GRAIN framework or of conditional measurement. In particular, the
estimated W is shared/fixed, while the native reader remains context-dependent;
we did not estimate a flexible W(c).

## Actual experiment

All 270 V1 expanded texts were joined to frozen GRAIN, local-reader and Flash
scores. There are 216 trait texts, 108 paired contrasts, 36 cells with 3 seed pairs
per cell, and 54 non-persona controls. All 12 refusal texts remain. Queries are E
and N; assigned generation axis and readout query are separately reported.

Source: named/described x diary/opinion/advice, both generation axes, 36 pairs.
Application: 72 pairs across new tasks, new wording, or both. These materials were
previously observed; this is an exploratory transfer design, not a new lockbox.

Standard deviations, Ridge alpha=1 estimator, and rank-2 joint two-query action
matrix are fit on source pairs only. No intercept ensures zero evidence change
implies zero action. Application head states are measured for intervention targets
and reference controls, never used in fitting. All original full articles remain.

The candidate action uses four text-computable coordinate differences. Applying
that action requires the frozen reader and its low-text context. It is not a fully
LLM-free scorer. Direct response Ridge uses exactly the same source features and
pairs. Donor-head intervention is a mechanistic reference requiring the high text,
not an independently deployable prediction competitor.

## Evidence movement is not target-specific movement

Mean source-standardized GRAIN displacement norms:

| Controlled change | Paired contrasts | Norm |
|---|---:|---:|
| Extraversion prompt | 54 | 1.74635 |
| Neuroticism prompt | 54 | 1.78556 |
| Newsletter instruction versus neutral | 18 | 1.87743 |
| Spelling instruction versus neutral | 18 | 1.63295 |

These are not matched-strength interventions or pure-noise controls. Their similar
magnitudes show why magnitude alone does not identify the target. E-prompt mean
local E change is .053626, while N-prompt mean local N change is .001170. Removing
refusals in a separately labelled sensitivity does not fundamentally change this
picture. Flash scores have their own scale and remain separate.

The post-result 18-cell E displacement-norm versus signed E-readout correlation is
.3005 (.3411 after linear length adjustment); N is -.1187. Neither endpoint is a
ground-truth trait detector. The cell map is routing evidence, not an identification
of opportunity versus reader failure.

## Native intervention results

The following correlations compare COMPUTATIONAL CHANGES, not predicted human
personality with questionnaire labels. All 72 application pairs, both generation
axes pooled; axis-specific and split-specific results are in metrics.csv.

| Method | E change r | E RMSE | N change r | N RMSE |
|---|---:|---:|---:|---:|
| Actual donor-head action | .46428 | .08307 | -.00456 | .04110 |
| Compiled four-coordinate action | -.26028 | .09693 | .02247 | .04049 |
| Direct four-coordinate Ridge | -.18245 | .11675 | -.05287 | .04258 |
| Equal-norm random action | .06441 | .09271 | -.09470 | .04057 |
| Predict zero change | undefined | .09285 | undefined | .04050 |

E compiled-minus-direct RMSE difference has block-bootstrap interval
[-.03065,-.00818], but compiled fails the zero-change reference and its direction
agreement is .514. Thus even a seemingly clear RMSE advantage is not a useful
construction success here. Against actual donor action, compiled E r=-.06780;
its RMSE disadvantage to random has interval [.00172,.00606].

The direct E mapping has source r=.4067 and new-task r=-.4107. This is a concrete
transfer failure of the specified estimator, not evidence that all text evidence
is absent or that only nonlinear models could work.

## One targeted follow-up: act on the remainder

For donor change d and compiled change a, define r=d-a. Existing actions give
f(0), f(a), f(d). One additional forward per application pair/query measures f(r).
Let A=f(a)-f(0), R=f(r)-f(0), I=f(d)-f(a)-f(r)+f(0). Exactly:

    donor_effect = A + R + I.

| Readout | R versus donor effect r | R RMS | Interaction I RMS |
|---|---:|---:|---:|
| E | .93760 | .0366475 | .00113775 |
| N | .96448 | .0078523 | .00008840 |

The high remainder correlation is partly expected because r is constructed as
d-a and a has small effect. Do not advertise it as an independent discovery or
personality validity. The extra forward establishes that this remainder really
acts on the reader, and quantifies the finite interaction rather than assuming
additivity. N's donor action still fails to explain the full N text contrast.
RMS and symmetric allocations are accounting measures, not information percentages.

## What the theory should change

Population covariance and reader-action geometry are different objectives. The
failed estimator minimizes hidden-state reconstruction error; that can prioritize
large state changes with little influence on the chosen measurement. A justified
next candidate is an action-weighted approximation, not another head/rank grid:

    L = sum_i ||J_(q,c,i) [delta_h_i - W delta_e_i]||^2 + regularization.

Here J is the score derivative with respect to the fixed head-state input, obtained
on SOURCE materials only. Equivalently the local metric is J^T J. A more general
active-subspace metric averages such matrices over a declared query/condition
reference; its weights define what the measurement preserves. It is not a universal
personality metric. Conditions and reference distribution must remain explicit.

This is a PROPOSED next estimator, not executed in this packet. If it still fails
with the same evidence, compare a text-computable richer representation, rather
than quietly retune the existing application set. The finite test above closes
with a negative promotion decision and an identified representation/objective gap.

The construction chain is therefore: target and context -> controlled evidence
contrast -> measured action -> action-preserving candidate coordinates -> frozen
text computation -> independent psychological interpretation. Intervention effects
alone leave the absolute score origin unspecified; retain population/reference
anchors. Simulation calibrates a computational instrument, not human trait truth.

## Predecessors and proposed contribution

- [DAS](https://proceedings.mlr.press/v236/geiger24a.html) learns distributed
  intervention alignments instead of enumerating neuron sets.
- [Constantine and Gleich, Computing active subspaces with Monte Carlo](https://arxiv.org/abs/1408.0545)
  derives subspaces from average gradient outer products. The proposed derivative
  metric is established dimension-reduction mathematics, not our invention.
- [Cui et al., Likelihood-informed dimension reduction](https://arxiv.org/abs/1403.4680)
  motivates distinguishing raw parameter variation from directions informed by
  observations; it does not establish a psychological interpretation for GRAIN.

Our candidate contribution remains a transferable open-text measurement
construction procedure with evidence provenance and population reference. This
packet tells that procedure what the current representation/fit does not preserve;
it does not establish novelty or complete a universal theory.

## Verification and artifacts

- 432 full-text/query state captures and 576 native actions: 1008 forwards, 502.33s.
- Remainder: 144 forwards, 62.75s. Total 1152 forwards; no model training or new API.
- All 432 cached scores reproduce original scores exactly. Source-only operator
  independently rebuilt with augmented least squares: max error 1.44e-15.
- Self interventions change zero. Four-corner accounting residual is zero.
- 23 targeted tests pass; code lint passes excluding EXE002 (external-volume file
  executable-bit reporting on non-entrypoint modules), formatting applied.
- r1 was the cache-only pre-format run; r2 is canonical. Pair CSVs are byte-identical.
- This is same-agent independent numerical replay, not third-party replication.

Key outputs: pairs.csv, cells.csv, evidence_control_contrasts.csv,
evidence_control_summary.csv, cell_endpoint_relations.json, action_model.npz,
action_arrays.npz, native_effects.jsonl, metrics.csv, paired_intervals.json,
remainder_effects.jsonl, remainder_summary.csv, audit.json and run manifests.
Local source-derived arrays remain ignored; no commit, push, tag or lockbox change.

## Reproduction

From the independent GRAIN repository, using the parent project's Python venv:

```sh
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
OUT=results/grain_vector_mechanisms/conditional_operator_v1/r2
"$PY" scripts/run_grain_conditional_operator.py --output "$OUT"
"$PY" scripts/run_grain_conditional_operator_native.py --output "$OUT"
"$PY" scripts/summarize_grain_operator_evidence.py --output "$OUT"
"$PY" scripts/run_grain_operator_remainder.py --output "$OUT"
"$PY" scripts/audit_grain_conditional_operator.py --output "$OUT"
"$PY" -m pytest -q tests/test_grain_conditional_operator.py tests/test_grain_persona_reverse_trace.py tests/test_grain_persona_generation_trace.py
```

Native/remainder runs resume existing complete records. Keep source fingerprints
unchanged. A new design belongs in a new protocol/output, never this frozen r2.
