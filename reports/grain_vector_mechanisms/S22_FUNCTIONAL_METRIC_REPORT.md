# S22: shared functions and shared coordinate directions differ

2026-09-08. COMPLETE; no new neural calls or questionnaires. Numerical analysis
.021seconds, excluding source validation. Uses frozen S21 reference/transfer12.

## Main finding

The empirical response-function metric changes the interpretation of angles:

| Query pair | Native gradient cosine | Reference functional correlation |
|---|---:|---:|
| N / confirmation | .8497 | .7150 |
| N / Feeling numeric | .3160 | -.5775 |
| N / quantitative content | .3714 | -.3787 |
| Feeling numeric / quantitative content | .7274 | .8343 |

Functional correlation compares different reader functions applied to the same
reference support, with recipient-query centering and equal author/query weight.
It is neither a human scale correlation nor the correlation of each query's
native responses on different evidence. The12 authors remain12 people, not48
independent observations. Negative values preserve the original scoring signs.

K_rs=E_ref[(g_r^T D-centered)(g_s^T D-centered)] is positive semidefinite.
Under D'=A D and g'=A^-T g, scalar scores and K are unchanged. Its empirical
spectrum is .64735/.30021/.03693/.01552; the first two query-function combinations
account for94.76% of reference response variance. These are not personality factors.

The perspective is informed by the metric distinctions in
[Park et al.2024](https://arxiv.org/html/2311.03658v2) and covariance-aware weight
interpretation in [Haufe et al.2014](https://pubmed.ncbi.nlm.nih.gov/24239590/).
This empirical L2 seminorm is not the papers' identified causal metric.

## Keep both components

Using the fixed content reader c, lambda_r=K_rc/K_cc:
N=-.45256, confirmation=.24772, Feeling=1.42659, control=1.
The functions split exactly into lambda_r*g_c and g_r-lambda_r*g_c.
Pooled reference covariance of the orthogonal part with the anchor is below3e-22;
maximum transferred score-addition error2.17e-18.

| Recipient | Total effect r | Control-aligned r | Control-orthogonal r |
|---|---:|---:|---:|
| N | .9307 | .0716 | .9247 |
| Confirmation | .9486 | .4750 | .9243 |
| Feeling numeric | .8015 | .2074 | .7840 |
| Content control | .6374 | .6374 | undefined, zero component |

The distinct components retain useful response reconstruction. They are
orthogonal only in the pooled reference metric, not necessarily within each
query or on transfer authors. No component is declared noise or removed.
Whether the N source association lives in the aligned or orthogonal component
is the next direct question, answered separately by S23.

Code/protocol203f5c5. Three synthetic tests verify PSD, invertible-coordinate
invariance, component equality, reference orthogonality and singular-support
handling. Output: results/grain_vector_mechanisms/functional_metric_s22_r1/,
including manifest, summary, component_metrics.csv, private component arrays
and private transfer scores. Run scripts/run_grain_functional_metric.py using
parent venv Python and a new output directory. Full61 focused tests pass.
