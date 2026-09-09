# GRAIN A/B programs: real versus synthetic reader references

2026-09-09. The previously missing NAV10 comparison is complete: four independent
candidates, two fixed reader sources, and three input families. The code is
frozen in local commit c1f0787. No weights or private materials are pushed.

## Result that matters

The two useful local paths survive the reader-source change. The token-only T/F
program gives AUC=.6657 with the real-source reader and .7047 with the synthetic
reader. The four-block free-chat low-state program gives r=.3515 and .3506,
respectively. Thus the existing state association is not just an accident of
using the synthetic rather than the old real-reference reader.

This is an actual reader-source factorial, not new independent participants or
a claim that the two estimates are statistically equivalent. All sources,
coefficients, chart origins and pooling rules were fixed before external labels.
No source, program, sign or threshold was selected from the new human results.

## What the factor changes

R_real/R_synth changes the **upper-state reader** G/g and its origin. Both B
programs use the same two synthetic upstream chart-origin texts per query, so
reader provenance is isolated from chart-origin choice. It is not a factorial
of every possible reader-source and upstream-origin-source combination.

A_linear uses the synthetic-only early-to-upper-state estimator. A_source uses
the fixed synthetic-plus-twelve-unlabelled-reference-author estimator. Both
estimators are unchanged across readers; only their compiled output functions
change. B_token uses static pretrained embeddings and semantic/order alignment;
B_context4 uses exactly four contextual blocks, aligned evidence and exact
query/end slots. Both B arms use mean pooling over two construction charts.

Eighteen new construction forwards took7.49seconds. The real-source B charts
were differentiated at six fixed synthetic origins; existing synthetic-source
charts were reused. Four-block evaluations executed288 controlled inputs in
15.58seconds,288 natural inputs in69.77seconds and923 external views in226.10
seconds. One early forward supplies all eight candidates. Token-only scoring
itself requires zero contextual forwards; joint batch cost includes the others.

Synthetic-control external replays versus prior frozen results: token and
context4 score error0.0, A_linear maximum2.11e-15. This guards the meaning of the
new comparator without rerunning the full teacher.

## Real-author comparison

All entries are from identical membership within an endpoint. T/F values are
AUC; E and low-state values are Pearson r. These are exposed exploratory data.

| Candidate | T/F real/synthetic, n64 | E real/synthetic, n213 | Free-chat state real/synthetic, n46 | Task-answer state real/synthetic, n46 |
|---|---:|---:|---:|---:|
| A_linear | .3634/.3043 | .0370/.0428 | .1310/.1880 | .1370/.1653 |
| A_source | .5255/.4745 | .0690/.0778 | .2343/.2607 | .1027/.1470 |
| B_token | .6657/.7047 | -.0674/-.0225 | -.0772/-.0214 | -.0424/-.0418 |
| B_context4 | .4615/.5355 | .1013/.0901 | .3515/.3506 | -.0266/-.0358 |

Relevant95% bootstrap intervals:

| Endpoint/program | R_real | R_synth |
|---|---|---|
| T/F B_token AUC | [.5216,.7887] | [.5594,.8361] |
| Free-chat B_context4 r | [.0474,.5945] | [.0382,.6013] |

The existing full-prefix fixed-reader controls are T/F AUC=.6196/.6116, E
r=.1767/.1661, free-chat state r=.5645/.5690 and task-answer state r=.1092/.1316.
The matching cached native controls are T/F AUC=.5906, E r=.2109, free-chat
state r=.5360 and task-answer state r=.0053. Source-native scores are historical
cached reads, not a new same-runtime benchmark. The larger historical MBTI
sample's native AUC=.660 is a different cohort and is not substituted here.

Human association and reconstruction of the original LLM function are different
columns. A token program can have useful T/F association without accurately
recreating all original-model evidence responses. Do not describe AUC=.705 as
70.5% classification accuracy or as comprehensive LLM distillation.

## Conditional response versus score levels

On96 edited observations from12 natural-source evaluation authors, B_context4
low-state function fidelity is r=.3607/.5136 for R_real/R_synth. Its matched
endorsement-response correlations are .7037/.6996 and time-response correlations
.3617/.6329. Directional association survives, but amplitude is compressed:
mean absolute endorsement change is about.000475 versus native.006955.

E B_token has natural level fidelity .6217/.5647, yet weak external E
association. Conversely, the T/F external association does not imply natural
function fidelity: that is -.0696/-.3469 in this edit packet. These contrasts
locate the next construction problem instead of inviting a global winner chosen
from human labels.

## Decision and artifacts

Keep token T/F and contextual state as two usable local routes. Keep both reader
sources as scientific comparators, rather than switching each endpoint to its
highest observed source. Source-span A is a coverage experiment, not the best
inference program. Future work should learn/derive conditional response rules,
not repeat this source comparison with an alpha grid.

The entire frozen packet is under
results/grain_vector_mechanisms/navigation_v1/NAV-10/reference_routes_r1/:

- construction_freeze.json: coefficient/chart hashes, source factor and18 calls.
- worlds/, natural/, external/: all eight candidate scores and prediction freezes.
- analysis/function_metrics.csv: model-function and native fidelity separately.
- analysis/response_metrics.csv: paired natural edits and author-clustered CI.
- analysis/external_metrics.csv: all32 candidate/endpoint rows,95% intervals.
- analysis/private_*: local joins and response pairs; never public artifacts.

Reproduce with scripts/run_grain_reference_route_control.py construct/evaluate/
analyze. Use a fresh --out for construction. Evaluation is resumable by --mode
worlds/natural/external under unchanged code and inputs; analysis reads labels
only after all three prediction freezes exist. The fixed reader-source comparison
closes the original NAV10 requirement without changing any previous result.
