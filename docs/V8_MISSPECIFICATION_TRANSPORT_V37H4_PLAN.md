# V8 V3.7H.4 Misspecification and Transport Refutation Plan

Status: prospective synthetic discovery protocol.

V3.7H.3 identified registered components in a balanced additive world. This
protocol deliberately breaks that coordinate system and asks whether GRAIN
can refuse interpretation before a saturated observed-cell decomposition is
mistaken for a transportable response function.

No real text, external label, psychological construct, diagnosis, or clinical
claim is used.

## Core distinction

For observed author-condition cells, the H.3 response table is saturated:

\[
R_{uc}
=Y_{uc}-\hat\mu-\hat A_u-\hat H_c.
\]

It can reconstruct an arbitrary seen-cell interaction. Therefore exact
reconstruction is recorded but is never a promotion gate. H.4 requires:

1. transport to held-out conditions;
2. replication across independent opportunity panels;
3. explicit refusal when stable residual structure remains;
4. no causal attribution when response and persistent opportunity structure
   are observationally equivalent.

## Synthetic worlds

### W0 additive positive control

\[
Y=\mu+S_s+G_{g|s}+A_{u|sg}+H_c+U_{uckp}+\varepsilon_{uckrp}.
\]

\(U\) is independent across authors, conditions, opportunities, and panels.
Both Gaussian and standardized heteroskedastic \(t_5\) modes are used.

### W1 nonlinear author-condition interaction

For latent author and condition vectors \(a_u,h_c\in\mathbb R^3\):

\[
z_{ucj}=a_u^\top M_jh_c,
\qquad
Q_{ucj}=\gamma\,
\operatorname{dc}\left[
\tanh\left(
1.25\,z_{ucj}/\operatorname{sd}(z_{\cdot\cdot j})
\right)
\right].
\]

`dc` denotes author/condition double centering followed by unit-RMS
normalization. Effect strength is registered by stable variance share
\(\omega\in\{.05,.20\}\).

### W2 non-ergodic opportunity process

An author-correlated regime persists within a panel and usually persists
between adjacent panels:

\[
Z_u=.5A_{u1}+\sqrt{.75}\xi_u,
\qquad
P(R_{u,p+1}=R_{up})=.95.
\]

The persistent condition-patterned opportunity term is:

\[
L_{ucp}
=\lambda\,
\operatorname{dc}\left[
\left(
\sqrt{.5}Z_u+\sqrt{.5}R_{up}
\right)\ell_c
\right].
\]

It does not disappear as \(K\) increases. Its stable part is observationally
equivalent to an author-condition response. The only valid classification is
`MODEL_INADEQUATE_CAUSE_UNIDENTIFIED` or `CAUSE_UNIDENTIFIED`.

### W3 latent hierarchy

Each author belongs to an unobserved four-level subgroup crossed with the
registered society/group hierarchy:

\[
Q^{latent}_{ucj}=\delta D_{q_u,c,j}.
\]

\(D\) is double centered and generated from a rank-two subgroup-condition
surface. The registered estimator does not receive \(q_u\).

## Frozen design

- 8 societies;
- 4 registered groups per society;
- 8 authors per group, 256 authors total;
- 16 conditions: 8 train, 4 calibration, 4 test;
- 4 independent opportunity panels;
- \(K=2,4,8\) using nested prefixes;
- 2 technical streams;
- 6 score dimensions;
- author cross-fitting within every registered group;
- rank candidates 1, 2, 3, 4, and 6;
- 100 repetitions per world/noise/effect cell;
- 1,400 discovery cells total.

## Cross-fitted transport

Authors are split into anchor and target halves inside each registered group.
The split is swapped and predictions are concatenated.

Rank selection:

- panel 1 supplies anchor condition structure;
- panel 2 target train conditions select rank on calibration conditions.

Final evaluation:

- panel 3 supplies anchor factors for every condition;
- panel 4 target train+calibration conditions estimate author loadings;
- panel 4 target test conditions are opened once.

The additive and structured predictors share the same author baseline and
anchor-derived condition profile.

## Observation-only diagnostics

### Cross-panel residual covariance

\[
T_{\mathrm{CRC}}
=
\frac{
\langle R^{(3)},R^{(4)}\rangle
}{
\|R^{(3)}\|_F\|R^{(4)}\|_F
}.
\]

Author-block permutations within registered groups provide a one-sided null.

### Residual low-rank concentration

The selected-rank singular-energy ratio of the cross-panel residual average
is compared with a null that independently permutes condition blocks within
each author.

### Held-out-condition gain

\[
T_{\mathrm{GEN}}
=R^2_{\mathrm{structured}}-R^2_{\mathrm{additive}}.
\]

Author-block sign flips test whether per-author held-out error reduction is
positive.

The three p-values use Holm familywise \(\alpha=.05\). Any rejection produces
`MODEL_INADEQUATE_CAUSE_UNIDENTIFIED`. The diagnostics detect omitted stable
structure; they do not identify its psychological cause.

## Additional boundaries

- `operation_gap` compares opportunity-first and pooling-first linear
  projections. For H.3 these operations should commute to machine precision;
  this is an implementation check, not an empirical theorem.
- A paired response/opportunity construction must have identity error
  `<=1e-12` and return `CAUSE_UNIDENTIFIED`.
- W2 reports the residual-energy K=8/K=2 ratio. A persistent floor must not be
  described as removed by additional opportunities.

## Frozen discovery gates

W0:

1. simultaneous 95% lower K=4 main-component recovery \(R^2\ge.90\);
2. simultaneous 95% lower split-panel correlation \(\ge.85\);
3. familywise `MODEL_INADEQUATE` one-sided CP upper `<.05`;
4. 90% CI for \(T_{\mathrm{CRC}}\) inside `[-.05,.05]`;
5. 90% CI for \(T_{\mathrm{GEN}}\) inside `[-.02,.02]`;
6. maximum operation gap `<=1e-10`.

W1/W3 at \(\omega=.20\):

1. detection one-sided CP lower `>.90`;
2. mean recoverable-gap closure lower bound `>.60`;
3. mean \(T_{\mathrm{CRC}}>.10\).

W1/W3 at \(\omega=.05\) define sensitivity only.

W2:

1. refusal one-sided CP lower `>.90`;
2. specific author/personality causal attribution rate CP upper `<.05`;
3. paired observational-equivalence error `<=1e-12`;
4. persistent residual K=8/K=2 ratio `>=.50`, so it cannot be reported as
   eliminated by opportunity replication.

## Decision

- `PASS_MISSPECIFICATION_AWARE`: W0 is calibrated, strong W1/W3 are
  detected, and W2 is refused without causal attribution.
- `PARTIAL_DETECTS_NOT_LOCALIZES`: omitted structure is detected but
  opportunity dependence and stable interaction are not safely bounded.
- `REFUTED_OVERCONFIDENT_ADDITIVE`: false alarms exceed the W0 gate, strong
  W1/W3 are missed, or W2 is assigned a specific author/personality cause.

A pass remains a synthetic misspecification result, not evidence that the
same diagnostics are calibrated for real text.
