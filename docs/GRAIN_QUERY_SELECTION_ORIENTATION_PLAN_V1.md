# NEXT73: separate query-conditioned selection from query direction

2026-09-11. PLANNED after NEXT72; no generic kernel or anchor accuracy search.
One bounded scalar-interface experiment, followed by source-operator development.

## Decision this changes

NEXT72 shows similar marginal spreads but opposite query-contrast responses.
Q changes both the probability measure a(Q) and directional statistic d(Q).
Which dependency must the next standalone numerical reader preserve: selection,
the direction paired with value, or their interaction? Do not choose another
regression head until that input relation is represented.

## Exact frozen design

Reuse audited NEXT50 pure K/V/dK/readouts/masks/RoPE and source-only Q bank;
source4LOO/new8all4, N/confirmation, four GRAIN directions, both readout modes.
Use only the fixed own/other source anchors, two by two, no new head/layer/dose
search, no human labels, MEPS, current Q, neural calls, fits or API requests.

For each recipient material/query calculate all four entries

    R_ab = Cov_{softmax(l(Q_a))}(v, d(Q_b)), a,b in {own,other}.
    S_ab(t)=E_{softmax(l(Q_a)+t*d(Q_b))}[v], S_ab'(0)=R_ab.

Diagonals must reproduce NEXT50 own/other key actions. Off-diagonals are defined
scalar-path derivatives, not ordinary same-Q interventions in the original model.
Do not evaluate their attractiveness and promote a mixed cell as a predictor.
Freeze all four outputs before native-action evaluation.

Use the exact symmetric two-channel account:

    Delta=R_11-R_00.
    Selection=.5[(R_10-R_00)+(R_11-R_01)].
    Direction=.5[(R_01-R_00)+(R_11-R_10)].
    Delta=Selection+Direction.
    Interaction=R_11-R_10-R_01+R_00.

Retain signed projections/cross-terms and source/new8 separately. Report each
query and N-minus-confirmation contrasts, not merely one pooled accuracy number.
This is algebraic allocation with a declared convention, not unique causation.

## Implementation / acceptance

- Add `query_selection_orientation.py`, staged runner and synthetic tests.
- Check pure input/source exclusions, diagonal identity, scalar finite differences,
  identical-anchor zero contrast and the exact decomposition. Reuse prior checks.
- One application+replay/account, max30CPU seconds,0fits/neural/Y. Old caches and
  primary NEXT72 files remain frozen. Private author/token/head rows stay ignored;
  publish only query/factor/contrast aggregates with code/input hashes.
- Record which query-conditioned signed operator the next source-trained
  function must produce. If neither part dominates, preserve both in the scalar
  generator; do not launch repeated attribution, capacity or rank sweeps.

This finishes the selection/orientation distinction, not the overall inference
goal. It must lead back to building the numerical reading function, not to a
fresh sequence of attention22 fidelity benchmarks.

## Frozen implementation before application

`query_selection_orientation.py` compiles an explicit vector-valued operator
Cov_a(v,delta K)/sqrt(h), then contracts it with the direction anchor. This
keeps the selection and direction interfaces separate without dropping either.
Private packets hold operators, four actions and source-derived rotated anchors.

`scripts/run_grain_query_selection_orientation.py` stages prepare/predict/evaluate/
audit, with `--folder replay` for independent replay and `--output` for fresh runs.
Application denies old/native targets. Evaluation checks all diagonal actions;
the fixed source author0, bothqueries/all4factors/all4paths supplies64 mode-wise
central differences at1e-5. No head or subject selected by the results.

Five explicitly named aggregate tables contain32query-cell/16contrast-cell/
8query-account/32factor-account/4contrast-account rows. OS sidecars never enter
the inventory. The native and swap projections have different denominators;
the interaction is reported separately, not added again to the symmetric account.

Five new tests (including two parameterized random worlds) plus the existing
relevant tests give32passed before real application; Ruff passes. Preserve all
four cells even if off-diagonal fidelity is attractive.
