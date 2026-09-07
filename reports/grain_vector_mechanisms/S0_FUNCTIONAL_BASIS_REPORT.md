# S0: functional coordinates recover response information missed by variance

2026-09-08. COMPLETE; no new neural forward or questionnaire access.

## Finding

On the same cached96 development authors, supervised PLS coordinates substantially
improve prediction of finite block16 query-position intervention responses.

| Query | Authors | SVD+Ridge r | Full Ridge r | PLS+Ridge r | PLS MSE reduction vs SVD |
|---|---:|---:|---:|---:|---:|
| TF | 60 | .4586 | .8029 | .8021 | 54.5% |
| Neuroticism | 36 | .3883 | .7663 | .7882 | 55.6% |

Paired author-bootstrap PLS-minus-SVD MSE intervals:
TF [-.000312,-.000172]; N [-.000066,-.000026]. Both favor PLS. Full Ridge also
improves both tasks; the signal is not exclusive to a particular PLS construction.

## Protocol and interpretation

- Five outer/three inner deterministic author folds; all views, two edits and
  two doses stay with their author. This yields480 TF and288 N response rows.
- Origin-preserving estimators; ranks2/4/8, Ridge1/10/100, minimum inner-CV MSE.
  Directions and RMS scales fit only outer-training authors for each evaluation.
- Existing D2 and D3B users are development material. The reported intervals
  describe paired OOF errors conditional on these fits, not a fresh confirmation.
- TF/N are query names. These r values target **internal intervention responses**,
  not Big5 scores or MBTI questionnaire labels. No claim of .8 personality accuracy.
- RMS scaling is specific to each coordinate system. The comparison concerns
  complete estimators, not a coordinate-only intervention isolating every cause.

H1 of the seminar receives direct computational support: low-complexity,
function-informed coordinates can preserve substantially more effect information
than the previous variance-first compression. Next test the actual local gradient,
not another increasingly large SVD rank search. Preserve the previous measured
effects; improve their mathematical approximation.

## Next implementation choice

S1 uses fresh source-only authors, three common queries and shared edits. Because
the Mac has24GB unified memory, retain the primary block16 gradient using a frozen
suffix graph. Optional comment attribution is gradient-times-state at block16,
with a zero hidden-state reference; it is not input-embedding attribution.
This preserves the central experiment without retaining all28 layers' gradients.

Private data and exact metrics: `results/grain_vector_mechanisms/functional_basis_s0_r1/`.
Script: `scripts/analyze_grain_functional_basis.py`; targeted synthetic tests:7 passed.
S1 preparation is frozen; no new questionnaire or bridge/MEPS outcomes are used.
