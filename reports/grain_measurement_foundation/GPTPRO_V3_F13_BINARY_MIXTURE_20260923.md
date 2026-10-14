# F13 B1 source/history uncertainty application

## Execution

The six F08 world/seed cells now use all 200 actual source bootstrap fits on
application authors, with complete/mixed panels and oracle/soft/hard channels.
The source models and reader outputs were frozen before application. Prediction
occurs before consuming each next event; hidden outcomes are used for evaluation
only. No new API calls, text generation, model training or human data access.

Canonical outputs under `results/grain_gptpro_foundation_v3/F13/`:

- `binary_mixture_full_r1`: 259,200 target rows, 2,400 author-panel groups.
- `binary_mixture_summary_r1`: 7,200 author-panel-channel aggregates and
  seed-specific 2,000-resample author-bootstrap summaries.
- `binary_mixture_smoke_r1`: two-group smoke retained.

The full run took 20.55 seconds. For each source fit and prediction, 200
categorical history states were sampled through their exact multinomial count
representation. The equivalent 10.368 billion categorical draws are not that
many independent observations or model executions. B1 has no personal parameter
posterior; its inner integration concerns finite hidden history states only.

## Formula and interpretation

For source fit b, filtered state s with working weight w_bs, and transition
probability p_bs, mu_b=sum_s w_bs p_bs. The decomposition is

\[
V_{history}=E_b\sum_s w_{bs}(p_{bs}-\mu_b)^2,\quad
V_{source}=\operatorname{Var}_b(\mu_b),\quad
V_{event}=E_b\sum_s w_{bs}p_{bs}(1-p_{bs}).
\]

Their sum is mu(1-mu). Probability variation excludes the event component.
The filtered weights retain the existing F08 observation approximation; reader
scores are not newly calibrated likelihoods. The moment-based probability bands
are not automatically confidence intervals for the unknown predictive mean.

## Findings

Descriptive averages across the three seeds, XOR mixed panel:

| Channel | Probability MSE | History variance | Source variance | Working band coverage |
|---|---:|---:|---:|---:|
| oracle | .039620 | .039120 | .000050 | .971296 |
| soft | .063790 | .039913 | .000055 | .889583 |
| hard | .070019 | .033106 | .000056 | .853426 |

The uncertainty induced by source estimation is much smaller than uncertain
history. Nevertheless, soft/hard reader errors exceed the variation represented
by that working history law. More source resampling cannot repair a biased or
miscalibrated observation channel. This is a mechanism distinction, not an
argument for deleting the mixed panel or adjusting interval width to pass.

Complete-panel oracle coverage is .914213 in XOR and .920417 in IID. These
descriptive rates also do not warrant blanket calibration acceptance. In IID
mixed soft, probability MSE is .000059 and source/history variances are
.000039/.000054: with weak history effects, apparent state variation itself
partly reflects fitted transition fluctuations.

## Verification and limits

- Two new unit tests compare every sequential ensemble distribution with the
  existing scalar QualifiedHistory in both observation modes.
- Seven binary/source/mixture tests and Ruff passed.
- The full Bernoulli total-variance identity error is <=2.50e-16 after CSV reload.
- Internal Monte Carlo mean RMSE against exact finite-state means is .000486.
- Output keys are unique. Descriptive bootstrap resamples authors within each
  seed, not the 36 correlated events as independent calibration trials.
- Application-wide independent scalar replay has not yet been performed; these
  tests and identities do not substitute for it. Author bootstrap here conditions
  on the frozen source ensemble and does not account for new-source sampling.
- B2/M parameter uncertainty, observation-channel calibration, prescribed
  missingness/shift/history conditions and full F13 acceptance remain open.

Next: replay application calculations against scalar references, then implement
the B2/M working sampling law without fabricating conjugate posteriors from
fractional reader counts. No parent card is promoted; the full goal stays active.
