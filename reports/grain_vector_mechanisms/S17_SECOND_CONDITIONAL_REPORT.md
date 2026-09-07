# S17: conditional curvature generalizes beyond the pilot

2026-09-08. COMPLETE. Reused24 S16 authors/72prefixes; collected189 further
source authors/567new prefixes in520.33seconds. All213 authors have three views.
No additional model fit, calibration, suffix call, backward or outcome selection.

## Fixed-formula result

| Cohort | Formula | r with exact reader | MSE/exact author variance | Mean error |
|---|---|---:|---:|---:|
| New189 | Conditional covariance |.854969|.708774|+.004604|
| New189 | Covariance plus second-order term |.920711|.318792|-.001294|
| All213 | Conditional covariance |.847179|.701136|+.004556|
| All213 | Covariance plus second-order term |.925021|.301820|-.001251|

The pilot's improvement extends to the remaining authors. The formula was chosen
from internal fidelity on the pilot and frozen before the continuation. The
new189 are additional formula measurements on the same exposed development
source, not a newly recruited or untouched psychological validation population.

All rows retain exact comment masses and the same author-out common query and
S12 affine reader. The added term is
E_G[(V-E_G V)(q^TK-E_G(q^TK))^2]/2. It is second order in query strength and
third order as a joint cumulant. No Gaussian assumption, refitted slope, sign
flip or bias correction is included. Internal r=.925 is not personality r.

## Source connection after score freeze

| Formula | Target-C r | Target MSE/train-mean null | Source N r | Delta N r vs exact, paired95%CI |
|---|---:|---:|---:|---|
| Exact reader |.315627|.908661|.133706|Reference|
| Conditional covariance |.287099|.995844|.104370|-.029336[-.100256,.043057]|
| Plus second-order term |.307056|.975468|.125430|-.008276[-.062050,.046117]|

The fixed second-order approximation is closer to the exact reader in both
internal shape and retained source association than first order. It is not a
prediction-accuracy gain over exact reading. The existing exact pathway remains
the reference; second order is an interpretable finite approximation to it.
Source N r for second order has Fisher95%CI[-.009156,.255553], p=.06769.
Paired intervals resample authors conditional on frozen cross-fit predictions;
they do not include retraining uncertainty or undo previous source exposure.

## What this contributes to the theory

An author supplies a joint distribution of contextualized keys and values. A
shared query reads it through three distinct operations: total body gain,
relative comment allocation, and conditional response shape. S16 located all
three; S17 shows that one low-order curvature term improves conditional shape
outside the discovery pilot. This is a concrete mathematical mechanism rather
than a new verbal factor label or a universal noise-removal rule.

Do not enlarge the Taylor order grid now. The next question is whether the
population-trained reader direction participates in the actual decoder's
judgment. S18 converts extraction weights into training-covariance patterns and
intervenes on the selected and reader-null components separately. That changes
the experimental question from approximation quality to functional pathway.

## Reproduction and resource accounting

Protocol/code commit a306320; S16 source code0c8ed6b. New runner
`scripts/run_grain_second_conditional.py`, modes prepare/run/evaluate.
Private output `results/grain_vector_mechanisms/second_conditional_s17_r1/`:
manifest,213packets,run_status,internal_fidelity,private_scores_before_outcomes,
scores_frozen,metrics,source_paired_intervals,summary. Hashes bind reused pilot
scores and source packets. Scores were persisted and hashed before source Y join.
No bridge/MEPS questionnaires or lockboxes opened; T/F route unchanged.
No process remains from S17. S18 is a separately bounded experiment.
