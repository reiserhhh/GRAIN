# S25: actual configuration heads are efficiently measurable, not a stronger N score

2026-09-08. COMPLETE:213source N authors, three12-comment views each,
639prefix calls plus639local KV reprojections,590.43seconds after loading.
No upper decoder/LM-head scoring, backward calls, new fitting or source-label
calibration. Code b6d2a53; exact protocol frozen before collection.

## Main result

Keep the original S20 gradient. The actual head configuration readout is
z_config=mean_v g_N^T(h_native-h_projected). It has a positive but weaker
source N association than the existing pairing readout in this sample.

| Method | Primary189 N r | Bootstrap95%CI | All213 N r |
|---|---:|---|---:|
| New configuration-head score | .116028 | [-.042374,.253217] | .110449 |
| Existing S20 pairing score | .199990 | [.061117,.330554] | .174895 |
| Full S9 configuration contrast C | .198936 | [.081977,.290264] | .211552 |
| Native full model score | .163638 | [.016439,.294763] | .174134 |
| Projected full model score | .089303 | [-.052038,.231964] | .096869 |

Primary configuration p=.11185; paired delta versus pairing=-.08396,
CI[-.21308,.03150]. All213 delta=-.06445[-.18519,.05100]. Thus there is no
observed improvement and the paired interval includes no difference; preserve
the earlier pairing route rather than replace it. Discovery24 r=.05023 versus
pairing-.00790; the full cohort table is saved without selection.

Configuration-head score versus full C is only r=.08105 on primary189 and
.08277 on all213. S24's.9543 describes prediction of the head allocation on a
24-author/view0 intervention panel, not recovery of full C or source labels.
The head approximation can be good while other paths carry different effects.

## A verified execution shortcut

The native prefix provides block16 boundary H, final query and rotated keys/
values. Reapply S9's exact mean projection P to H, then only layer17's RMSNorm,
K,V affine maps and rotary position map. Reconstruct the full head by softmax
weighted values with its unchanged final query. This avoids repeating the
first17 decoder blocks for the projected condition.

On the first8authors/24views, collection cost25.40seconds. Stored actual S24
heads replay with maxerror3.02e-6 and linear scoreerror5.14e-8. Full213 collection
replays all24available S24/view0 heads: maxerror3.78e-6, same maxscoreerror5.14e-8.
Final-query boundary/vector changes and forbidden suffix calls are0. No dual-
prefix fallback was needed. All model weights still reside in memory; this is
a compute shortcut using pretrained representations, not removal of the encoder.

The formula uses tokenwise normalization from
[Zhang & Sennrich2019](https://papers.nips.cc/paper_files/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf)
and a transported functional covector motivated by
[Sundararajan et al.2017](https://proceedings.mlr.press/v70/sundararajan17a.html).
Actual source transport, not either paper, decides the usefulness of this score.

## Stability is not the same question as criterion association

Primary189 view-pair correlations (01/02/12): configuration .23665/.22491/.38081;
pairing .10036/.04804/.25739. More cross-text consistency does not automatically
produce a stronger N link. These random source views are not longitudinal
test-retest observations, and consistency is not a label for a psychological trait.

Next S26 was fixed during collection, before these source outcomes: decompose
the new configuration vector into the original pairing path plus a retained
remaining vector. Its exact covariance/variance account determines whether the
lower observed r reflects cancellation or extra criterion-unrelated dispersion.
Do not sum overlapping head routes or choose a source-label-optimized blend.

## Artifacts and replay

Private directory: results/grain_vector_mechanisms/configuration_source_s25_r1/.
Manifest binds code, local Qwen implementation, source membership, comparison
scores and frozen gradient. Files include213complete three-view head packets,
checkpoint_008.json, private_scores_before_outcomes.csv, scores_frozen.json,
private_source_connections.csv, metrics.csv, view_consistency.csv, summary.json.
Scores froze before current Y join; these source users are previously exposed
development material, not a new untouched PANDORA benchmark.
The generic metrics CSV also contains MAE/RMSE from the common helper. Signed
contrast scores are not calibrated questionnaire scores, so those raw-unit
errors must not rank these methods as personality prediction models; use the
declared association and paired-comparison columns for this mechanism question.

Run scripts/run_grain_configuration_source.py with parent venv Python:
prepare; run (eight by default); inspect; run --max-authors 213; evaluate.
Supply a new --output-dir to each command for a replay. No model download,
protected bridge/MEPS/lockbox use, T/F change, training or private cloud upload.
94focused tests pass after S26 and normalization-order math additions; Ruff passes.
