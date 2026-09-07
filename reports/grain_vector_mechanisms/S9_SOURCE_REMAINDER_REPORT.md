# S9: configuration response connects to real source neuroticism

2026-09-08. COMPLETE.213 N source authors and64 fixed-hash T/F controls,
three unchanged cue-masked views per author.1662 forwards,0 backwards,
2431.4seconds; separate16-forward observer check57.2seconds. No model training.
Scores were frozen before joining the already authorized source outcomes.

## Mainline finding

The mean-preserving token-configuration contrast C=S_new-M has a positive
neuroticism association: **r=.2116**, Fisher95% CI[.0794,.3364], Pearson
p=.00191. Its label covariance is .001046, paired-author bootstrap95%
CI[.000284,.002054]. This connects the internal intervention to human source
measurements, rather than only showing that model scores can be moved.

The stronger paired finding is that preserving the native configuration improves
the model's association relative to its projected endpoint: N r=.1741 versus
.0969, delta=.0773, paired bootstrap95% CI[.0236,.1330]. The corresponding T/F
comparison is AUC=.7037 versus .6657, delta=.0380 [.0049,.0774] on64 controls.
These are fixed model interventions on existing development material, not
independent external replication or causal manipulation of people's traits.

## Scores on exactly matched authors

| Score | N,213 authors: Pearson r | T/F,64 authors: AUC |
|---|---:|---:|
| Old CUDA/BF16 teacher | .1765 | .5906 |
| New paired MPS/FP32 teacher S_new | .1741 | .7037 |
| Projected suffix endpoint M | .0969 | .6657 |
| Frozen S4 student f | .0673 | .6747 |
| Configuration contrast S_new-M | .2116 | .6026 |

The64-author T/F subset is a control sample, not a replacement for S4's2328-author
result (student AUC .7173, paired old teacher .6591). Do not compare the subset's
numbers against the full population as if only the algorithm changed.

C is a signed difference, not a calibrated probability or predicted0-1 trait
level. Its correlation and ranking are meaningful; generic MAE/RMSE and .5-threshold
fields emitted for this row by the shared metric helper are not used for claims.

C's N point estimate exceeds S_new by .0374, but the paired interval is
[-.0771,.1678]. Its difference from the frozen student is .1443 [-.0133,.3058].
Thus the result establishes a useful label-related component, not a statistically
settled claim that using C alone is the best personality predictor.

## Locate the old teacher's remainder

The three-part identity is exact on the frozen author-level scores:

S_old-f = (S_old-S_new) + (S_new-M) + (M-f).

| N component | Covariance with Y | Bootstrap95% CI |
|---|---:|---|
| Observer-condition difference | .000742 | [-.000385,.001925] |
| Configuration response | .001046 | [.000284,.002054] |
| Remaining projected-suffix/student difference | .000181 | [-.001041,.001333] |
| Total old teacher/student remainder | .001969 | [.000099,.004012] |

The point estimates add exactly. The configuration term accounts for about53%
of this signed covariance remainder at its observed scale, **not53% of personality
variance or a causal mediation percentage**. The useful conclusion is the positive
measurable configuration covariance, not the ratio's apparent precision.

N total teacher covariance changes from .002663 (old) to .001921 (new), despite
similar Pearson correlations; the new score has lower spread. Student covariance
is .000693 and projected endpoint covariance .000875. This illustrates why
correlation and additive covariance must be reported separately.

## Controls and interpretation

Exact old prompt tokens were reproduced for all831 views. Original cached
candidate logits reconstruct all old scalar scores within2.7e-15. The bounded
observer check found FP32 implementation/padding differences below5e-6, while
precision/backend conditions change scores materially. The full S9 experiment
therefore uses paired FP32 origin/intervention scores and explicitly accounts for
the old/new difference. See [resource check](S9_SOURCE_RESOURCE_CHECKPOINT.md).

A2000-draw author-label permutation check on the frozen C gives N two-sided
p=.00050 (the Monte Carlo minimum1/2001), null mean r=.0013. T/F C gives p=.0735;
its positive covariance interval is not treated as unanimous significance across
every inferential method. The main T/F result here is the paired native-versus-M
AUC difference, not C as a new standalone classifier.

M retains the query state and other earlier-layer information. Hence this is
evidence for a useful **conditional suffix configuration route**, not proof that
all contextual information disappears under averaging or that E5 cannot encode it.
No bridge/MEPS questionnaire, lockbox, new LoRA, or external text upload was used.

## Next decision

The main question now changes from "is the internal contrast label-related?" to
"can that useful contrast be recovered by a cheap vector function?" Run a bounded
teacher-contrast student on the existing E5 cache before collecting more neural
data. Preserve C, M and S_new separately and use teacher-only inner selection.
The component choice is informed by S9, so this remains an exposed-source discovery
continuation, not a fresh confirmation. S10's independently frozen KV intervention
provides the elementary mathematical link beneath this configuration hypothesis.

## Reproduction and artifacts

Runners: `run_grain_source_remainder.py` (prepare/run/evaluate),
`check_grain_source_observer.py`, `analyze_grain_source_contrasts.py`.
All output paths are relative to ignored
`results/grain_vector_mechanisms/source_remainder_s9_r1/`:

- `manifest.json`, `private_material.json`, `private_source_hashes.json`.
- `private_packets/`, `private_components_before_outcomes.csv`, `components_frozen.json`.
- `metrics.csv`, `covariance_split.csv`, `summary.json`, `run_status.json`.
- `observer_check/summary.json`; `paired_analysis/paired_intervals.csv`,
  `permutation_control.csv`, `covariance_totals.csv`, `summary.json`.

Intervals are pointwise author bootstrap, conditional on frozen predictions;
the old OOF student's training uncertainty is not re-estimated. No fit was chosen
from these outcome comparisons.
