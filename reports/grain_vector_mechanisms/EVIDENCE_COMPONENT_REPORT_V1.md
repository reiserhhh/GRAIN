# Which evidence term supplied the gain? NEXT71

2026-09-11. COMPLETE; primary account frozen at `fb97a12`. No new fit/neural/API/
human-label/MEPS access. Finite pass60; overall research goal ACTIVE.

## Main finding

NEXT70's raw+logmass new8 gains are **joint changes with substantial cancellation**,
not evidence that a single new selector supplied the improvement. Local readout
refitting and the derivative of kernel weighting help, direct content often
opposes them, and logmass makes a smaller positive contribution.

The exact frozen account preserves the prior3.602%/5.413% action-error reductions.
It now explains their algebra. Stop generic normalized-kernel development; return
to the source-anchored query/value mechanism, retaining mass and relative-value
effects in a common scalar formulation rather than dropping inconvenient terms.

## Source: untouched-B response

When only front BODY changes, local L and raw content C/selection S changes on
back BODY B are exactly0 in all source cells. Thus the raw reader's B response
comes entirely from M=logmass. This is a property of this function, not the data
having no cross-token content signal. Normalized reading adds cross-token C/S.

Signed projection is `<component,target>/<target,target>` across4authors and
fourGRAIN directions. It is not a correlation or an independent variance share.

| Query / reader | C projection | S projection | M projection | Total projection | Total SSE/zero |
|---|---:|---:|---:|---:|---:|
| N / raw | 0 | 0 | .020915 | .020915 | .969572 |
| N / relative | .020849 | .004811 | .013304 | .038964 | .948326 |
| Confirmation / raw | 0 | 0 | -.020489 | -.020489 | 1.045596 |
| Confirmation / relative | .023730 | -.012795 | -.028489 | -.017554 | 1.042535 |

For N the normalized content redistribution is the largest positive projection;
for confirmation, a positive content contribution is opposed by selection and
mass terms. Their energies do not add independently: all16 ordered Gram entries
are retained. The confirmation example makes the need for signed cross-terms
concrete rather than a generic warning.

Coordinate and hidden-field accounts are also saved. Their target inner products
average within each author, then average authors. Hidden-field SSE is a ratio
of these averaged energies, unlike NEXT70's mean of author-wise error ratios;
do not compare those two hidden summaries as if their weights were identical.
Coordinate and GRAIN-action totals reproduce NEXT70.

## New8: where the observed action-error improvement went

Relative to the frozen matched-local baseline, decompose the total prediction
change as `d=dL+C+S+M`, where dL is the change in local coefficients. Use the
declared symmetric error credits from the plan, including all cross-terms.
The table entries are **percentage points of relative baseline-error reduction**,
not percentages of personality information or unique causal responsibility.

| Query / reader | Local refit | Content | Selection | Logmass | Net reduction |
|---|---:|---:|---:|---:|---:|
| N / raw | +8.867 | -15.247 | +9.346 | +.637 | +3.602 |
| Confirmation / raw | +10.368 | -11.428 | +6.175 | +.299 | +5.413 |
| N / relative | +1.727 | -4.545 | +1.072 | +.772 | -.975 |
| Confirmation / relative | +1.258 | +1.931 | -1.504 | +1.536 | +3.221 |

Logmass credit is positive in all8 declared query/arm/full-vs-allocation cells,
range .299 to2.809 percentage points. These overlapping cells are not8 independent
replications. A positive credit is not proof that retaining only that term works.
In particular, C and S are generally not separate gradients of scalar scores;
their nonzero curls cancel in the sum. A two-arm synthetic check verifies this
post-account mathematical property without changing the primary fit or data.

These eight exposed development authors are disjoint from source fitting, not
fresh confirmation subjects. Targets are the frozen model's query actions,
not questionnaire labels. Full-score results remain the unchanged NEXT70 report.

## Theory and next step

The common scalar structure explains why arbitrary gradient-term deletion would
not be a valid scoring-system upgrade. Develop the selector and content together.
The next source-query formulation uses a log-partition potential:

    Psi(s,t)=log mean exp(logit + s*value + t*GRAIN_key_change).
    Psi_t = mean_attention(GRAIN_key_change)
    Psi_st = Cov_attention(value,GRAIN_key_change).

This reconnects the small mass finding with the real query-directed mechanism
already observed in NEXT50. NEXT72 will compile the mean/covariance geometry from
existing Qbar/K/V/dK caches, rather than fit another generic Gaussian reader.
It will explain amplitude and alignment separately and not claim a new end-to-end
personality predictor merely because a local identity is reproduced.

## Verification and artifacts

- 48component packets and6aggregate tables exactly replayed; component gradient/
  action reconstruction maximum1.6653345369377348e-15 before native target joins.
- Error-credit identity maximum2.5500435096859064e-16; unchanged raw C/S and L on B
  exactly0; frozen/matched-local GRAIN predictions reconstructed within1.14e-15.
- Primary account+replay+audit12.469320CPU seconds, below60seconds. No training.
- 73focused tests at primary freeze; final focused rerun70passed, including all
  7new component/geometry cases. Two inherited SWIG warnings; Ruff passes.
- Public tables: [evidence_components_v1](evidence_components_v1/publication.json).
- Private packets: `results/grain_vector_mechanisms/logic_evidence_components_v1/`.
- Runner: `scripts/run_grain_evidence_components.py`, stages `prepare`, `apply`,
  `evaluate`, `apply --folder replay`, `evaluate --folder replay`, `audit`.
  Use `--stage`; preserve existing outputs and pass `--output` with a fresh
  directory for reruns. The publisher intentionally uses the frozen default run.

See [theory](../../docs/GRAIN_EVIDENCE_COMPONENT_THEORY_V1.md) and
[NEXT72](../../docs/GRAIN_QUERY_PARTITION_GEOMETRY_PLAN_V1.md). No scorer/release/
lockbox changes, private data publication or remote push.
