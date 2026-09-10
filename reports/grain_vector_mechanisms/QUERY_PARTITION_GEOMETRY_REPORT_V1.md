# Joint evidence geometry, not marginal strength: NEXT72

2026-09-11. COMPLETE. Primary `79e2a4a`, inventory amendment `cfe4d9d`;
canonical R2, finite pass61, overall goal ACTIVE.
Zero fits, new neural/API/human-label/MEPS calls. Frozen source4 leave-one-out
anchors and exposed source-disjoint new8 application; both queries, both fixed
readout modes, all four GRAIN directions and all12 heads retained.

## What was learned

The source-query program's key action is exactly a **signed cross-moment** of
readout value and GRAIN-induced logit change under query-conditioned weights.
Compiling this joint geometry reproduces the earlier NEXT50 actions within
3.747e-16. This is a new numerical representation/account of an existing effect,
not an independently improved prediction result.

Changing the query anchor barely changes marginal value spread, yet reverses
the direction of the two-query contrast. New8 own-mode value-SD changes from
.114863 to.117322 for N, and .114616 to.112304 for confirmation: about2%.
Direction-SD changes +8.4%/-4.1%. These magnitudes alone do not encode the sign.

| New8 query contrast | Native-action r | Native projection U | Native projection B | Native projection R=U-B |
|---|---:|---:|---:|---:|
| Own-query anchor | .939345 | +.205872 | -.068884 | +.274756 |
| Other-query anchor | -.890468 | -.342670 | -.143197 | -.199473 |
| Shared anchor | .477162 | -.062292 | -.103705 | +.041413 |

U is uncentered E[v*d], B is E[v]E[d], R is their difference, summed across heads
before taking the N-minus-confirmation contrast. The projection denominator is
native squared action norm, not personality variance. Subtraction can *increase*
the correctly directed signal: own-query U+.205872 minus B-.068884 yields+.274756.
It is not an operation that necessarily deletes unwanted information.

## Where magnitude is lost

| New8 / own anchor / own readout | RMS value SD | RMS direction SD | Spread-weighted absolute alignment | Head-sign retention | Summed response RMS |
|---|---:|---:|---:|---:|---:|
| Neuroticism query | .114863 | .359819 | .087826 | .389114 | .026668 |
| Confirmation-seeking query | .114616 | .376975 | .070253 | .319030 | .013313 |

The two reductions are defined at different operations: sum|Cov|/sum(sd_v*sd_d)
describes within-head alignment; sum|sum_head Cov|/sum|Cov| describes signed
head aggregation. These are descriptive arithmetic ratios, not independent
psychological attenuation rates. They must not be multiplied by the separately
averaged RMS SDs to claim an exact total-RMS formula. Head/cell values are not
independent subjects. No rho was undefined in the current real packets.

Effective attention support averages3.65/3.74 tokens per head in these cells.
This covers the original full allowed sequence, including non-BODY positions;
it does not mean that three lexical words explain a person's personality.
Pooled-readout conclusions are similar and included, not selected away.
Other-anchor confirmation alone has higher action r(.965 vs.903), illustrating
why one-query accuracy ranking cannot substitute for retaining query contrasts.

## Mathematical and development consequence

The finite partition Psi(s,t)=log mean exp(l+s*v+t*d) produces mean-v/mean-d
as first derivatives and Cov(v,d) as its mixed second derivative. A synthetic
example has identical marginal distributions/variances but opposite covariance.
Therefore a standalone reader preserving only marginal moments cannot in general
recover this action. It must preserve signed pairing, not just activation size.

The next unresolved dependency is concrete: swapping Q changes both attention
weights a(Q) and directional logits d(Q). Current results do not uniquely allocate
the swap effect between them. NEXT73 will build the two-anchor crossed scalar
interface and separate these changes, then specify what the next source-trained
operator must retain. No more generic Gaussian/alpha or attention accuracy sweep.

## Verification and artifacts

-12paired private packets and3aggregate CSVs exactly replayed;24query,96factor,
 12contrast rows. Old-action difference3.747e-16; U-B-R maximum4.025e-16.
-1.917672CPU seconds including prepare/application/replay/account/audit, below90.
-34focused tests passed, including6new cases: finite first/mixed derivatives,
 covariance PSD, zero variance, masks, logit shifts, GQA/RoPE, old actions and
 scalar key derivative and sidecar exclusion. Inherited SWIG/PyTorch deprecations
 only; Ruff passes. Initial glob also listed AppleDouble `._*.csv` metadata;
 publisher rejected it. R2 uses explicit table names and an exact-set audit.
 The first run is retained; all three real numerical tables match byte-for-byte.
-Outputs: [publication](query_partition_geometry_v1/publication.json); private
 `results/grain_vector_mechanisms/logic_query_partition_geometry_v1/r2/`.
-Runner `scripts/run_grain_query_partition_geometry.py`: `--stage prepare`,
 `compile`, `evaluate`, then `compile/evaluate --folder replay`, then `audit`.
 Fresh reruns require `--output`; the publisher uses the default audited run.

Targets remain a fixed neural model's local actions. Current K/V/dK still come
from its prefix/transport; no new end-to-end author questionnaire predictor or
psychological naming is claimed. Scorer, release, lockbox and old assets unchanged.
See [theory](../../docs/GRAIN_QUERY_PARTITION_GEOMETRY_THEORY_V1.md) and
[NEXT73](../../docs/GRAIN_QUERY_SELECTION_ORIENTATION_PLAN_V1.md).
