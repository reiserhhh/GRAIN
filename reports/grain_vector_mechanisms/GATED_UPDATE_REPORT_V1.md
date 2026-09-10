# NEXT43: a useful gated transformation compiles into a fixed reader

2026-09-10. **Finite experiment complete; whole-system goal ACTIVE.**

## Finding and mainline decision

The large block25 MLP response located in NEXT42 does not require finely adapting
its local gate/norm derivative to each current author. With both replaced by
paired source-author operators, new8 action r remains .996936 for N and .998451
for confirmation; SSE/zero=.011988/.002932. Both queries retain useful amplitude,
not just rank correlation. Keep this simpler mechanism rather than insisting
that every useful stage must have novel per-author nonlinear coefficients.

This is a local source-operator result on previously explored material, not
human personality prediction. The current upstream tangent d is still produced
by the fixed native model; the experiment locates where simplification works,
not where all context can be removed.

## Fixed four-way result

Same4 source authors,8 source-disjoint reader authors,2 queries,4 frozen GRAIN
directions:24 author-query units. Old4 use other3 same-query source contexts;
new8 use all4. Current d is identical across each unit's four alternatives.
Source B/N is same-query even when the final t readout is pooled-query.

New8, own-query t; target is the saved native block25 MLP increment contracted
with the same frozen t, not the full H16 action or a current downstream gradient:

| Local operator | N action r | N SSE/zero | Confirmation action r | Confirmation SSE/zero |
|---|---:|---:|---:|---:|
| Current B and N, analytic identity | 1.000000 | 6.18e-13 | 1.000000 | 5.23e-13 |
| Mean source B, current N | .996340 | .020103 | .998740 | .002247 |
| Current B, mean source N | .999773 | .001177 | .999952 | .000179 |
| Mean paired source B_s N_s | .996936 | .011988 | .998451 | .002932 |

The identity row checks arithmetic, not predictive discovery. SSE/zero is
sum(error^2)/sum(native_action^2), not centered R-squared. New8's32 actions
per query are8 people x4 directions, not32 independent participants.

Fully source-frozen RMS is .332905 versus native .308761 for N, and .221981
versus .225690 for confirmation. Sign agreement is31/32 and32/32. Factor-centered
r=.996645/.998563. Per-factor r ranges .984370-.999360 for N and
.992963-.999340 for confirmation, with all rows retained.

Pooled t produces fully source-frozen r=.996846/.998302 and SSE/zero
.012870/.003176. Old4 own-source fully frozen r=.997263/.985422 and
SSE/zero=.007816/.029540. There was no winner selection, new label fitting,
hyperparameter search or new source-person recruitment.

## What was decomposed, and what was not discarded

The actual derivative is

    dM = W_down[p * W_up dz + q * W_gate dz],

where p=silu(g), q=u*silu'(g), and dz includes the full RMS radial derivative.
Input and gate variation are signed terms of one product rule. They are not
separate content and personality channels.

New8 own-source current terms:

| Query | Input projected share | Gate projected share | Input-only SSE/zero | Gate-only SSE/zero |
|---|---:|---:|---:|---:|
| N | .554633 | .445367 | .234238 | .343504 |
| Confirmation | .591891 | .408109 | .200869 | .384652 |

Both terms are materially useful; omitting one is substantially worse than
retaining both with source-fixed coefficients. Signed cross-energy is positive
(.422258/.414479), although the two terms have opposite signs on9/32 and10/32
actions. Shares describe the computed increment, not psychological information.

Full-vector source-frozen relative L2 errors are .212876/.193736, while action
errors are much smaller. Therefore a useful query reading does not require
reproducing every hidden-vector coordinate. The same geometric distinction was
seen in earlier reader work; this pass finds it inside a specific gated block.

## Usable compilation

After the frozen four-arm result, the paired-source operator was algebraically
compiled into a fixed vector:

    wbar = mean_s N_s^T B_s^T tbar
    predicted_increment = wbar^T d_current.

Forty-eight addressed readout slots were exported; they include repeated fixed
source programs for new authors, not48 independent learned factors. Their dot
products reproduce the frozen source_BN predictions within7.772e-16. Two new
anisotropic-normalization tests verify the adjoint calculation. This post-outcome
step is an exact implementation conversion, not an additional blind experiment.
At this local read step current x and MLP matrices are unnecessary after wbar
is compiled; current d and its upstream encoding/transport remain dependencies.

The next useful question is whether query-specificity in this compact interface
resides mostly in wbar, in d_current, or their pairing. [NEXT44](../../docs/GRAIN_COMPILED_QUERY_TRANSFER_PLAN_V1.md)
tests both cross-query borrowing directions and exact query-contrast accounting
using saved numeric assets only. It is not another near-perfect precision grid.

## Execution and verification

- Primary code/analysis frozen in `010c1d3` before native collection, after
  `c2dc981` implementation. A test-only lint correction preceded collection;
  the earlier prepared-only folder is retained as `_preformat`, with0 native calls.
-96 upstream JVPs stop before block25 MLP;149.086s combined post-loading capture
  time,3.059s numerical prediction,0 new scoring/backwards/fits/human labels/API.
- A conservative forecast paused after8 JVPs because one-time export overhead
  was multiplied across units. A documented resume retained those8, the same code,
 96-call and300s caps. Two model loads total6.573s; no duplicate native units.
- All24 analytic current vectors match the original increment with maximum
  relative L2=2.055e-6 and maximum absolute error5.017e-6, within frozen tolerance.
-24 source exclusions checked,24 capture/24 prediction packets hash-verified,
 6 analysis CSVs byte-identical on replay. Scalar replay error0; independent
  metric discrepancy<=5.552e-16 and signed sums<=4.441e-16.
- Each active capture process and the numeric predictor passed4 target-denial
  probes with0 extra denied accesses. Native code did not execute block25 MLP
  or later blocks during boundary collection; synthetic hooks test this boundary.
-75 focused tests pass including the compilation tests;20 existing PyTorch/SWIG
  deprecation warnings. Ruff and whitespace checks pass; no full-suite claim.

## Reproduction and primary references

Private root `results/grain_vector_mechanisms/logic_gated_update_v1/` holds
manifests, fixed input/source assets, boundary captures, predictions, target-linked
analysis/replay, compiled readouts, `resume_decision.json`, and `replay_audit.json`.
Do not publish this root. Public tables contain aggregates only:
[comparison](gated_update_tables_v1/comparison.csv),
[factors](gated_update_tables_v1/factor_actions.csv),
[signed terms](gated_update_tables_v1/components.csv),
[vector errors](gated_update_tables_v1/vector_summary.csv).

```bash
PY="../project persona/.venv/bin/python"
# prepare needs a fresh output directory; capture resumes only unfinished units.
"$PY" scripts/run_grain_gated_update.py prepare
"$PY" scripts/run_grain_gated_update.py capture
"$PY" scripts/run_grain_gated_update.py sources
"$PY" scripts/run_grain_gated_update.py predict
"$PY" scripts/run_grain_gated_update.py evaluate
"$PY" scripts/run_grain_gated_update.py evaluate --analysis-name analysis_replay
"$PY" scripts/audit_grain_gated_update.py
"$PY" -m pytest -q tests/test_grain_gated_update.py tests/test_grain_compiled_gated_readout.py
```

[SwiGLU](https://arxiv.org/abs/2002.05202) supplies the architectural form;
[feed-forward prediction updates](https://aclanthology.org/2022.emnlp-main.3/)
motivate examining its role. Our contribution here is the GRAIN-aligned local
decomposition and source-operator test on the specified model/material, not the
invention of the product rule or evidence for a clinical construct.
See [the exact formulas](../../docs/GRAIN_GATED_UPDATE_THEORY_V1.md).
