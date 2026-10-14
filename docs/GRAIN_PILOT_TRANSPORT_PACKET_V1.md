# M11/M12: independent pilot, measurement units and uncertainty

2026-09-13. Freeze this packet before execution. No neural calls or human-label fit.

## Mechanism question

Can a new-source pilot repair reference mismatch without using the scored author's
latent truth? Which shifts require population moments versus a changed observation
function? One personal target and one current-state target must both be retained.

Use inherited Gaussian mixed-model REML, OU covariance, paired affine calibration
and total variance. Primary references:
- [Bates et al. 2015](https://www.jstatsoft.org/article/view/v067i01): profiled REML.
- [Driver et al. 2017](https://www.jstatsoft.org/article/view/v077i05): noisy continuous-time observations.
- [Millsap 1995](https://pubmed.ncbi.nlm.nih.gov/26790049/): measurement and predictive invariance differ.

These are inherited ingredients, not exact software replications or new theorems.

## Frozen experiments

1. Scalar state with random level/condition/partner coefficients. Each source and
   independent pilot author has 24 irregular occasions, eight independent reports
   per occasion. Fit population mean/covariance, OU rate/variance and within-occasion
   noise from pilot observations only. Common schedule is a deliberate identifiable
   design; it is not assumed available in arbitrary PANDORA conversations.
2. Fixed worlds: unchanged, population shift, dynamics shift, affine meter shift,
   condition-dependent meter shift. All parameters/worlds are fixed in the runner.
   Compare unchanged source, same-unit global anchors, independent local pilot,
   condition-dependent anchor+pilot, and oracle evaluation-only reference. Test
   latent outcomes never enter fitting, calibration, model selection or stopping.
3. Source-person bootstrap and paired-anchor bootstrap propagate construction
   variation. Report conditional and between-construction uncertainty separately.
   Approximate mixture intervals need empirical coverage, not a guarantee.
4. Existing public text bank: learn moments on independent source/pilot persons,
   score unseen simulated persons at two assigned conditions. Same/new wording and
   response-distribution shifts. Targets are the frozen reader's conditional event
   expectations, with latent numeric truth reported separately if used. No claim
   that a scalar calibration resolves open-language interpretation.
5. Existing PANDORA short-comment coordinates: split authors by a frozen hash,
   estimate author-balanced observation reference only on source; apply unchanged
   to held authors. Bootstrap source and within-author comments separately. This
   does not replace original GRAIN scores or certify a common psychological scale.

## Measurement interpretation

Paired anchors are the same observation under two meters, not two populations'
quantiles. y_new=c(z)+d(z)*y_old requires nonzero known-or-learned gain at use. Level
absorbs offsets, state/response differences absorb scale; a nonlinear or contextual
meter cannot in general be corrected by one population mean and variance.

M11 uncertainty targets: conditional material/state noise, source construction,
paired-reader calibration. No unobserved human biological variability decomposition.
M12 transport rule must distinguish reference shift, meter shift and model-structure
shift. Prior mean repair cannot remove partner/state shared causes found in M09.

## Acceptance

Archive predictions, truths kept separate from fits, source/pilot/evaluation IDs,
calibrations, optimization failures/boundaries, bootstrap fits and all comparisons.
Verify analytic REML derivative numerically, fixed-score batch independence,
affine equivariance, uncertainty accounting and independent score replay. Keep
negative outcomes. No performance-based tuning or claiming the whole TODO closed.
CPU budget300sec; preserve stages if exceeded. No MEPS read, release or new dependency.
