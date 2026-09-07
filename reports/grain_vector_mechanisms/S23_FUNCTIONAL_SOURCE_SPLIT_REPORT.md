# S23: the N source association survives the control-orthogonal component

2026-09-08. COMPLETE, zero new model calls, no human-label fitting. Analysis
.268seconds excluding source verification. Preserves S20's24-author g_N exactly.

## Source connection

| Cohort | Total N r | Content-aligned N r | Content-orthogonal N r |
|---|---:|---:|---:|
| Primary189, outside gradient reference | .199990 | -.034438 | .194560 |
| Discovery24, secondary | -.007903 | -.383739 | .066251 |
| All213, secondary | .174895 | -.066849 | .179480 |

Primary orthogonal r=.19456,p=.00730, bootstrap95%CI[.05286,.32977].
Its delta versus total=-.00543[-.04722,.03459]. The aligned part gives
r=-.03444,p=.6380. This locates the observed source association primarily in
the part distinct from this fixed quantitative-content reader; it is not a
prediction improvement or a declaration of a psychologically pure direction.

All213 orthogonal r=.17948,p=.00866,bootstrapCI[.04339,.30315].
Its delta versus total=.00459[-.03222,.04199]. Both total and both components
were frozen before this outcome join. The source has been used in earlier
development, and the chosen query definitions are limited. Do not call it a
new untouched PANDORA benchmark or generalize to all kinds of textual content.

## Exact retained-channel accounting

The S22 reference metric applied to the original S20 g_N gives lambda=-.4810823.
Keep z_total=z_aligned+z_orthogonal with no recalibration or axis flipping.
S20 score replay error9.99e-17; maximum component-sum error3.04e-18.

On primary189:
Cov(Y,z_total)=.0000610554
=(-.0000032820)+.0000643374, i.e. aligned covariance plus orthogonal covariance.
The source covariance identity error is1.36e-20. The negative aligned term
is retained, not dropped. Source channel covariance is not zero: reference
orthogonality does not force orthogonality after domain/support changes.

On all213: .0000549426=-.0000062193+.0000611619, exactly in floating precision.
Covariances refer to signed model-response contrasts, not calibrated scale scores
or percentages of human personality variance.

## Mainline implication

S19-S23 now support a shared-across-author, query-conditioned functional reader
with a distinct response component that carries real source N association.
A universal digit-answer direction does not explain the whole result.
The final formula keeps evidence, query-specific covectors and reference-defined
shared/distinct functions separate. Existing T/F remains a successful different
case; no release scorer is replaced.

The useful S9 body-collapse contrast C still has a different intervention from
the small pairing effect E. Next investigate how much C actually travels through
the measured layer17 final-query head, using a bounded native/collapsed factorial
restoration rather than another source-label-selected readout.

Code/protocol40930c2. Output:
results/grain_vector_mechanisms/functional_source_split_s23_r1/
contains manifest, scores_frozen.json, private_scores_before_outcomes.csv,
private_components.npz, private_source_connections.csv, metrics.csv, summary.json.
All private artifacts ignored. Run scripts/run_grain_functional_source_split.py
with parent venv Python and a new output directory. Full61 focused tests pass;
no protected bridge/MEPS questionnaire, lockbox or remote transmission.
