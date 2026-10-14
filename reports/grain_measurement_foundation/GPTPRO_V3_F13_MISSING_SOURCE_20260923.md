# F13 learned-source covariance preparation for missingness

## Completed

Actual independent source data: 300 authors, 32 events, three seeds, IID and AR=.6
worlds. Each was fitted under two declared residual-kernel shapes, rho=0 and .6.
All 12 source cells have 200 author-bootstrap refits: 2,400 total, zero invalid
variance fits. Each fit retains sampled author indices, RNG recipe and raw
covariance estimates. No eigenvalue clipping or rejection-based renormalization.

Canonical: `results/grain_gptpro_foundation_v3/F13/missing_source_full_r1`;
smoke: `missing_source_smoke_r1`. The source streams are independent of the
previous application reference streams. Future prior-shift application uses the
old IID source model, not the new application's latent parameter values.

The declared estimator regresses source sample covariance onto four bases:

\[
S\approx v_a11'+v_{ab}(1z'+z1')+v_bzz'+\sigma^2 C_\rho.
\]

It learns covariance magnitudes and source mean, but **does not estimate rho**.
This is an explicit operational source-moment arm, not REML and not evidence
that an unknown residual kernel has been recovered. Every fitted coefficient
vector was independently checked by reduced normal equations.

## Diagnostic result

Plugin component averages across seeds:

| True world | Assumed rho | Level variance | Slope variance | Residual variance |
|---|---:|---:|---:|---:|
| IID | 0 | 1.009246 | .245918 | .246934 |
| IID | .6 | 1.001555 | .256177 | .130931 |
| AR=.6 | 0 | 1.036441 | .236068 | .225571 |
| AR=.6 | .6 | 1.014878 | .242014 | .243143 |

The wrong correlation shape changes fitted variance magnitudes. Refitting source
parameters therefore cannot be assumed to repair structural misspecification.
Application calibration under these actual fitted parameters is the next step;
the table alone does not establish its magnitude or direction.

## Application interface prepared

`f13_masked_conditioning.condition_source_fits` conditions all fitted source
models using the actual masked observation kernel. It factors the kernel once,
then computes each two-parameter precision update. Irregular observed times keep
their original gaps: rho raised to a gap of four is not replaced by rho.
Empty evidence returns prior moments; an unsupported source fit is rejected,
not silently removed from the ensemble.

Nine related tests passed, including exact constructed source covariance,
unidentifiable design rejection, and batch versus observation-space conditioning
on irregular time points. Ruff passed. Source fits are complete; application
mixtures, 200 inner parameter draws, coverage and uncertainty components are
not yet executed for this packet. No F13 parent-card completion is claimed.

No human data, API, neural training, commit, push or lockbox action.
