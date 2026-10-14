# F13 B2/M history-refit comparison

## Completed experiment

Using the frozen F08 source models, ran both online and joint-history-refit B2/M
on all 2,400 application author-panel groups, six world/seed cells, three reader
channels and 36 next-event targets. Each history reconstruction uses 200 complete
dependent paths. Outputs: 518,400 head-level rows, 14,400 author/head/channel/panel
aggregates, seed-specific paired 2,000-author-bootstrap summaries. Runtime 137.97s.

Canonical roots under `results/grain_gptpro_foundation_v3/F13/`:
`path_heads_full_r1`, `path_heads_summary_r1`, with smoke in `path_heads_smoke_r1`.
Every group stores its RNG recipe and checkpoint. No new reader or source fit,
private text, neural training, API request, lockbox action or publication.

## Findings

Probability MSE, descriptive mean across seeds:

| World / panel / reader | Head | Frozen online | History-refit | Difference |
|---|---|---:|---:|---:|
| XOR / mixed / soft | B2 | .093931 | .094191 | +.000259 |
| XOR / mixed / soft | M | .075075 | .076770 | +.001694 |
| XOR / mixed / oracle | B2 | .056838 | .055910 | -.000928 |
| XOR / mixed / oracle | M | .045980 | .046255 | +.000275 |
| IID / mixed / soft | B2 | .042795 | .041777 | -.001018 |
| IID / mixed / soft | M | .010975 | .010199 | -.000776 |

Full-history reconstruction is not a universal improvement. Its effect depends
on target/head and observation channel; no model is selected using these results.
In XOR mixed soft, mean absolute score shifts are .023613 (B2) and .015118 (M).
The shift is not merely a new error bar on the same estimator. B2 sampled raw
predictions require clipping in 5.51% of draws, versus .245% for M; clipping is
recorded rather than treated as evidence of a calibrated probability model.

The strongest uncertainty counterexample is complete hard/oracle data. Every
sampled path equals the observed history, so imputation variance is zero and
both methods agree exactly. Yet XOR MSE remains .020360 for B2 and .011844 for M
(IID .028078/.010576). Exact reconstruction of observations therefore does not
eliminate finite-event estimation error. History uncertainty must not be reported
as total parameter uncertainty.

## Calculation and verification

For each path, weighted transition counts give B2's original unit-ridge normal
equations and M's original source-frozen rank-two projection. The prediction
uses that path's final state; values outside the operational probability range
are counted and clipped before working probability moments are calculated.

Eleven binary/history/head tests passed, including scalar online versus batch
path-head agreement on known hard histories. FFBS was checked against exact
six-event enumeration in the preceding packet. Ruff checks passed. Full new
path-head application independent replay remains separate from these unit tests;
the preceding 259,200-row B1 audit does not certify this new arm.

## Scope control and next step

The original F13-02 protocol requires cold uncertainty for Delta, state and soft
next-p; it does not explicitly demand every historical B2/M estimator become a
Bayesian model. This diagnostic prevents a false uncertainty claim, but does not
justify adding an unbounded 200-source-by-all-paths search to the critical path.

Keep the B1 source/history mixture as the implemented finite-state next-p arm;
keep the B2/M imputation arm explicitly partial, with finite-event parameter
uncertainty unimplemented. Do not release its history-only variance as total CI.
Next execute the original missingness, correlated-error, correction/withdrawal,
prior-shift and cold-interface requirements, carrying these limitations in the
interval contract. No parent-card or full-goal completion is claimed.
