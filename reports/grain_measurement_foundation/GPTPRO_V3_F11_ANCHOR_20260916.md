# F11 independent source anchor: identical text observations, different targets

## Mechanism

The paired construction has two worlds with exactly the same text observation:

    A: y = L f + c*1 + M m + e; target f
    B: g = f+c*1; y = L g + M m + e; target g

Var(c)=.25. Since L repeats the three trait coordinates, both y expressions are
identical. World B has trait covariance Sigma_f+.25*11' and no common method
component. Text covariance alone has rank eight for nine requested covariance
components; adding independent source anchors raises the design rank to nine.

The anchor is z=f+eta in A and z=g+eta in B, with unit loading and known independent
noise variance .25. These are substantive assumptions of this experiment, not a
claim that a real questionnaire automatically supplies such an anchor. Anchors
are supplied ONLY on source authors to fit population covariance. Application
scoring receives text alone and no anchor or hidden target.

## Execution and results

Three seeds, each 300 source /100 validation /200 application: 1,800 unique
simulated authors, shared across paired worlds. Source moment fits retain raw
coefficients before nonnegative/PSD repair. No application selection or fitting.
Source estimates of the shared component in A: .228924, .252400, .227696
(truth .25). In B: .001766, 0, .011651 (truth 0; raw negative values retained
in fit artifacts before clipping).

Application mean MSE and plug-in 95% coverage:

| World | Method | MSE | Coverage |
|---|---|---:|---:|
| A: shared method | Unanchored effective trait | .274978 | .788333 |
| A | Source anchored | .222378 | .949444 |
| A | Oracle parameters | .217799 | .959444 |
| B: all common variation is target | Unanchored effective trait | .116412 | .952222 |
| B | Source anchored | .116290 | .955556 |
| B | Oracle parameters | .112958 | .957778 |

Paired anchored-minus-unanchored MSE intervals (2,000 application-author bootstrap,
stratified by seed, frozen source fits): A -.052600 [-.071758,-.034024];
B -.000122 [-.000525,.000228]. No universal gain is claimed: B does not need
the same correction. These are conditional-on-source uncertainty intervals, not
source-refit or human-population intervals.

## Meaning for construction rules

More precise measurement of y cannot alone decide whether the common direction
belongs to the intended construct or to a shared method component. An independent
calibration relation supplies information absent from y. Once that relation is
learned in a source pilot, it can alter a text-only scoring rule for new authors.
It does not reveal each author's realized method component exactly: oracle A MSE
remains .217799. Population identification and individual recovery differ.

## Verification and retained failures

- Smoke r1 and initial full r1 stopped in world B because application-loop variables
  overwrote source f/g. Partial files retained; corrected source variables are
  distinct. Canonical smoke/full revision is r2, with an end-to-end smoke regression
  test added. No scientific settings were changed to rescue results.
- Four new tests: source covariance rank, exact population moment recovery,
  text-only application interface and both-world smoke execution.
- 12 F11 related tests and Ruff pass.
- Alternate scalar covariance/inverse implementation replayed all 16,200 score
  rows and 36 metric rows, maximum error 1.40e-14.
- Exact population moment recovery passes. Finite-source parameter refits still
  need alternate reconstruction; source estimation uncertainty and optimizer
  audit in the separate MTMM panel remain open.

Artifacts: `results/grain_gptpro_foundation_v3/F11/anchor_full_r2`,
`anchor_smoke_r2`, `anchor_audit_r1`. Observations and hidden parameters are saved
separately; all fitted models and scalar predictions are retained. Paired world
construction is explicit and is not counted as independent sample replication.

This closes the finite independent-anchor demonstration, not F11 packaging or
the overall goal. No new human data, API, neural training, remote job, lockbox,
commit or push; no background experiment remains live.
