# All-nine frozen functions on human source labels: M29

2026-09-09. COMPLETE. Nine functions now have explicit source-label links,
paired cached-native comparisons, and signed mean/configuration/final-state
contributions. No human outcomes were used to construct these functions.

## Findings

The fixed functions retain positive Big5 associations across all five domains
and a T/F ranking signal. Their mean Big5 r is .1406 versus cached native .1671
on these same cue-masked source materials. This is a mechanism-transport result,
not a new official five-fold benchmark or an improvement over r=.196.

| Query | Authors | Function human r [bootstrap95% CI] | Cached native human r | Function AUC | Cached native AUC | Function/native r |
|---|---:|---:|---:|---:|---:|---:|
| Extraversion |213| .1767 [.0523,.2997] | .2109 | - | - | .7851 |
| Neuroticism |213| .1008 [-.0275,.2304] | .1765 | - | - | .6775 |
| Agreeableness |213| .1855 [.0422,.3132] | .1685 | - | - | .9118 |
| Conscientiousness |213| .0634 [-.0632,.1916] | .1294 | - | - | .8499 |
| Openness |213| .1764 [.0629,.2819] | .1501 | - | - | .8507 |
| EI (E positive) |64| -.0102 [-.2136,.2036] | .0471 | .4968 | .5240 | .5550 |
| SN (N positive) |64| .0550 [-.1639,.2671] | .1556 | .5625 | .6917 | .6389 |
| TF (F positive) |64| .2072 [-.0505,.4226] | .1866 | .6196 | .5906 | .7322 |
| JP (J positive) |64| .1052 [-.1463,.3385] | -.0853 | .5317 | .4658 | .6682 |

All paired function-minus-native r/AUC intervals cross zero. The native column
is the cached D1 observer, not a newly run identical-precision suffix control.
The [S9 audit](S9_SOURCE_REMAINDER_REPORT.md) established material backend/precision
effects despite exact prompt tokens. Thus these differences combine readout
transport and observer/runtime differences; they do not isolate a causal loss
from replacing the suffix. No new control run was added merely to improve the table.

The64 MBTI authors contain12 E,60 N,27 F and28 J positives. SN has only four
negative cases: function AUC95% CI [.1905,.9206],1972 valid bootstrap samples
of2000. Other function AUC intervals are EI [.3190,.6704], TF [.4708,.7500],
JP [.3856,.6757]. The full2,328-author D1 discovery is a separate larger native
screen, not the sample size of this functional experiment.

## What carries the external association

For the fixed function f=b+M+C+Q, the following are signed contributions
Cov(Y,part)/(sd(Y)sd(f)). They sum to its external Pearson r, rather than summing
individually standardized correlations. C is the centered configuration of
context-conditioned QUESTION-region token states, not a bag of raw text words.

| Query | Mean M | Configuration C | Final state Q | Total r |
|---|---:|---:|---:|---:|
| Extraversion | .03920 | .12812 | .00936 | .17668 |
| Neuroticism | .01979 | .03435 | .04671 | .10084 |
| Agreeableness | .02976 | .12384 | .03192 | .18552 |
| Conscientiousness | .02513 | -.00310 | .04134 | .06337 |
| Openness | .01912 | .12280 | .03446 | .17639 |
| EI | -.01154 | .02889 | -.02751 | -.01016 |
| SN | .08781 | .00014 | -.03296 | .05498 |
| TF | -.00938 | .24418 | -.02761 | .20719 |
| JP | -.02694 | .12848 | .00367 | .10521 |

Exact unrounded accounting error is below8.75e-16. TF illustrates why a weak
or opposing component must not erase a useful path: configuration contributes
.244, offset by mean/final contributions. C and N have a different allocation.
The full205 query-by-criterion cells remain available; no sign flipping or
target selection follows the outcomes. These are association allocations under
a fixed coordinate partition, not causal percentages of human personality.

## Execution and outputs

Use the frozen213 Big5-only and64 MBTI-only S9 authors, disjoint from the M28
construction panel. Each has three exact cue-masked views. Reuse old N/TF
prefixes and collect the other seven questions:1,044 new author/query units,
3,132 prefix-only calls,2,806.81seconds total. All1,321 units completed over
three bounded invocations. No suffix, generated response, new training, bridge
sample, MEPS labels or old lockbox was used here. Predictions were saved before
joining existing exposed source outcomes. Affine scores were not clipped into
probabilities or represented as calibrated questionnaire scores.

Local ignored outputs under `results/grain_vector_mechanisms/`:

- `all_factor_source_m29_v1/`: frozen private predictions,45 component/native
  external-metric rows, source covariance and COMPLETE external summary.
- `all_factor_source_atlas_m29_v1/`: nine paired comparisons, signed external-r
  contributions,205 full cross-target links and COMPLETE atlas summary.
- `all_factor_function_analysis_m28_v2/nine_query_functions.npz`: frozen
  coefficients, SHA256
  `b7c21a141af6d2b68b8e740c3ded95f8bc9525f12adc579c1230f97344699f1f`.

Reproduce the cached association analysis from the independent repository:

```bash
"/Volumes/mobile3/projects/project persona/.venv/bin/python" scripts/analyze_grain_all_factor_source.py \
  --source-dir results/grain_vector_mechanisms/all_factor_source_m29_v1 \
  --output-dir results/grain_vector_mechanisms/all_factor_source_atlas_m29_replay
```

Next decision: retain the common construction, but use target-specific allocation
and evidence opportunity to explain differing transport. Do not select a new
Ridge/LightGBM grid on these already exposed human correlations.
