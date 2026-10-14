# F13 recovery 2 protocol v1: dynamic missingness, variance components, MDC95

Frozen 2026-09-26 01:48 JST before computation. Extends the completed linear
F13 arms (missing_nested_full_r1, cold_contract_full_r1), which are reused and
not rerun. Original protocol: `GRAIN_GPTPRO_ATOMIC_PROTOCOLS_V3.md` section F13.

## A. Dynamic targets under missingness (F13-04 for F05 T2/T3)

- Source: the 200 frozen F05 core source models per seed (`F16/t2_cold_full_r1/<seed>/bundle.json`);
  estimator = conditional-Gaussian marginalization of (a,b,state) with 64-node
  Gauss-Hermite over eta (the numeric-dynamic integration version used by the
  dynamic transport packet), generalized to an arbitrary observed time subset.
  No source refit. Application: 200 F05 core authors x 3 seeds, prefix t=0..47.
- Masks (drawn per event with SeedSequence([seed,13,7,mask_id,author_index])):
  complete; MCAR .2; condition-dependent (.2 when z=-1, .6 when z=+1); MNAR
  (.2+.4*sigmoid(y)). The estimator always assumes ignorable missingness.
- Targets: T2 = s_47; T3 = mean(y_48..51) under plan (+1,-1,+1,-1), evaluator
  corrected exactly as in the text-transport packet. Truth is read only after
  all scores are written.
- Metrics per seed x mask x target: MSE, bias, coverage with Wilson95, width;
  paired author bootstrap (2000, seed 20260915, seeds stratified) of MSE
  differences mask - complete. COMMON decision rule.

## B. variance_components.csv

Per world: F04 L (Delta), F05 core (state/forecast), text channel. Columns:
world, facet (person, condition, person:condition, state, reader, residual and
their covariances), true value where the simulator declares it, source
estimate (mean over the 200 source fits) with 2.5/97.5% bootstrap quantiles,
and note. Covariances are kept (never zeroed to make totals add). Facets the
design cannot separate (material vs reader in the text route) are marked
NOT_IDENTIFIED rather than filled.

## C. Minimal detectable change on actual runs (T1)

For each F04 L application author (200 x 3 seeds) split the 32 events into two
occasions (t<16, t>=16; 4 events per condition each). Score each occasion with
the frozen T1 source mixture (200 models). Under the static simulator the true
change is exactly 0. Predictive SE of the observed change
d = s2 - s1 = sqrt(V(s2-s1)) computed per source from the linear kernels
(noise and source spread; person terms cancel only when the two occasions have
identical designs) and mixed over sources; MDC95 = 1.96 SE. Calibration check:
fraction of authors with |d| <= MDC95 (Wilson95). Report the MDC also in
reference-SD units. MDC is a measurement-error threshold, not a clinical MCID.
