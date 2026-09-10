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
