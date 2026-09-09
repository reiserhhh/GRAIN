# NEXT15: conditional remainder, measured factorial, and allocation order

2026-09-10. Exposed-source mechanism follow-up, fixed before new derived accounts.
No new endpoints, raw text, models, human-label fits, or reference selection.

## Decision changed by this packet

NEXT14 locates a measured conditional anchor outside all seven current scalar/
component proxies. The useful next construction depends on whether that part is
conditional mean/background, mean-preserving evidence configuration, or query
interaction. M15 already measured the required four cells; do not collect them
again or report its old raw associations as a new discovery.

## Fixed source and functions

- `configuration_query_factorial_v1/private_cells_before_outcomes.csv`, verified
  against `cells_frozen.json`:48 author-query rows,24 T/F and24 N, three-view means.
- `logic_scalar_component_v1/r2_replay/private_author_predictions.csv`, verified
  against its prediction freeze: all seven fixed proxies, primary `raw_f`.
- `logic_conditional_target_gap_v1/r3_replay/` supplies the previous complete
  crosswalk and aggregate account. Verify prediction/anchor/criterion provenance;
  do not import the old outcome-linked M15 file for construction.
- Same author/query join; f11 must reproduce current native; f10 must reproduce
  NEXT14's measured anchor. No author exclusions or new sign/orientation choices.
- Human Y is the already exposed M29 source criterion, verified against the
  reading-level manifest. Join only after model-only contrasts are frozen.

## Exact contrasts

`C(E)` replaces token states within each comment by that comment's mean while
preserving layout. x is the current P/Q state, x0 the common reference.

    f11=F(E,x); f01=F(C(E),x); f10=F(E,x0); f00=F(C(E),x0)
    B=f00; E_main=f10-f00; X_main=f01-f00
    I=f11-f10-f01+f00
    F=B+E_main+X_main+I.

For one fixed proxy p and its affine projector P_p onto span(1,p), Q_p=I-P_p:

    F=P_p F+Q_p B+Q_p E_main+Q_p X_main+Q_p I.

Use the same projector for all components; fitting it uses no Y. Each covariance
has the same root denominator sd(F)*sd(Y) on the same24 authors/query. Verify:

    previous remaining conditional anchor = Q_p B + Q_p E_main
    previous remaining transport = Q_p X_main + Q_p I - Q_p(f-b_ref).

These are signed covariance identities, not independent variance fractions or
deployable model corrections. Never residualize each component against different
covariates and then claim the total still has its original meaning.

## Ordered allocations

Keep the baseline separate and report both exact paths:

- E then X: E allocation=E_main, X allocation=X_main+I.
- X then E: E allocation=E_main+I, X allocation=X_main.
- Symmetric two-order average: E=E_main+I/2, X=X_main+I/2.

Subtracting one E allocation from the other gives I; corresponding X change is
-I. Every order preserves the same F. This is an allocation property, not an
empirical claim about the actual execution order of the network or human thought.
The symmetric result is the two-player Shapley allocation for this fixed local
four-cell game. It is not a unique causal importance measure or a global
conditional-expectation SHAP computation. The baseline C(E) is material-specific.

## Analysis and stop rule

- Publish all2 queries x7 proxies x5 common-root components; primary reading is
  raw_f, other proxies check whether the arithmetic conclusion changes.
- Publish both ordered allocations and their fixed symmetric average. Do not
  choose the order with larger Y association. A component's marginal correlation
  and its remaining contribution are distinct quantities.
-1999 paired whole-author resamples/query, seed20260910+query index in sorted
  query order; recompute each projector, pointwise intervals, fixed predictions.
- Report component RMS/SD and interaction-driven allocation differences. Confirm
  reconstruction, projection orthogonality and the NEXT14 bridge to1e-12.
- Create synthetic tests for pure additive/pure interaction functions, shared
  projection, cancellation and a case where raw association vanishes after p.
- Finish when the remaining measured term is localized and the next input/target
  contract is identified. No polynomial/rank/alpha search, large model run, or
  public release promotion. Existing readers remain untouched.

## Prior work and subsequent branch

[Sundararajan, Dhamdhere & Agarwal (2020)](https://proceedings.mlr.press/v119/sundararajan20a.html)
distinguish feature attribution from interaction attribution and explicitly
specify how interactions are allocated. Our two-player calculation is elementary,
not a reproduction of their full Shapley-Taylor algorithm. It prevents double
counting, while preserving the measured interaction as its own target.

[Hooker (2007)](https://doi.org/10.1198/106186007X237892) treats dependent-input
functional decompositions and their support weighting. Here we use already
measured internal interventions, not an independent-input ANOVA variance claim.

Before NEXT16, inventory M12-M23/MNAV rather than repeat attention means, the
quadratic parity failure, or endpoint-constraint experiments. Use the located
remaining term to define one new representation/learning target if warranted.

Output: `results/grain_vector_mechanisms/logic_conditional_factorial_v1/` private
freeze and links; public aggregate report/tables, updated theory/atlas/goal and
local commits. No model score, identity or questionnaire value in public files.
