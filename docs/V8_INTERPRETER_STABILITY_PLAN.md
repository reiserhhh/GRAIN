# GRAIN V8 Interpreter Stability Experiment

Status: `EXECUTED__PLANTED_PASS__PANDORA_RENDERER_ONLY`

## Pre-primary scoring amendment

Before any PANDORA API evaluation, the planted quick run exposed four
bookkeeping/evaluation ambiguities. Version 2 therefore freezes the following
corrections before the primary real-text gate:

- repeated empty interpretation sets are not counted as perfect informative
  agreement; stability is evaluated among profiles with a compiled candidate
  opportunity, with interpretation coverage reported separately;
- the multilabel reliability statistic is named correctly as Fleiss kappa,
  not alpha;
- observer-complete author coverage must be at least 0.95 so structured-output
  failures cannot silently select an easier real-text cohort;
- discovery associations must reproduce the same sign in the source-disjoint
  left and right halves before they can become registered links;
- dense real-text profiles are deterministically limited to eight candidates,
  matching the frozen output schema, ranked by observed support density,
  GRAIN support clarity, discovery association strength, and stable ID;
- the critic uses a literal pass/qualify/reject rubric and may not invent
  hypothetical counterevidence or state-trait concerns;
- targeted event deletion treats every atom sharing that event code as
  affected, so legitimate linked changes are not miscounted as non-target
  instability;
- neighborhood distance association must have the predicted positive sign,
  and same-author comparisons use strangers matched first on hashed condition
  overlap and then text volume.

These amendments do not alter GRAIN scores, prompts, event codes, reference
bands, or planted API outputs. The formal planted run is recomputed from its
frozen cache under the corrected summary rules before PANDORA starts.

## Objective

Test whether an LLM can repeatedly translate frozen GRAIN measurements and
source-bound behavior observations into stable, evidence-constrained
interpretations. GRAIN remains the sole scoring channel. The LLM is not
licensed to create person-level scores, personality types, diagnoses, or
construct names.

## Pipeline

1. The behavior observer reads raw, deidentified segments only and emits
   discrete event codes with exact span provenance.
2. GRAIN produces anonymous reference-relative dimensions, support status,
   and uncertainty without LLM modification.
3. Discovery data register dimension-to-event links. Calibration may compare
   at most two semantically equivalent interpreter prompts. Confirmation data
   cannot change prompts, links, thresholds, or gates.
4. The interpreter emits bounded interpretation atoms. Each atom binds an
   anonymous dimension, scope, relation, direction, qualifier, evidence,
   counterevidence, alternatives, and a controlled sentence.
5. An independent critic judges each submitted atom exactly once and cannot
   rewrite it.

## Experiments

- `E1`: five repeated full-pipeline runs on identical input.
- `E2`: source-disjoint halves from the same author versus nuisance-matched
  strangers.
- `E3`: agreement as a function of frozen GRAIN neighborhood distance.
- `E4`: irrelevant formatting and metadata perturbations.
- `E5`: targeted key-evidence removal versus random evidence removal.

The planted calibration uses 200 profiles in the formal run. Real-text
confirmation uses the frozen PANDORA V7 geometry first. Essays, MEPS, and
X-market panels are secondary and run only after the primary gates permit
continuation.

Execution stopped after the formal PANDORA same-author and neighborhood gates
failed. Secondary panels were therefore not run. The final metrics and
decision are recorded in
`reports/V8_INTERPRETER_STABILITY_REPORT.md`.

## Promotion Boundary

`V8_INTERPRETER_TECHNICALLY_STABLE` requires safety, repeated-input stability,
key-evidence sensitivity, PANDORA source-disjoint and neighborhood validity,
secondary-corpus support, and perturbation robustness. Evidence fidelity
without author/neighborhood stability yields
`V8_INTERPRETER_RENDERER_ONLY`. Human construct validity remains deferred.
