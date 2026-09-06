# VM05 fixed layers: external-label readability is not monotone in depth

2026-09-07, round32. **All45 three-arm layer sensitivity heads and225 saved
official-fold replays COMPLETE**, compared with all45 corresponding primary
heads. The full54-case execution also includes the9 separately reported
no-PCA heads:270 saved-fold replays, maximum error2.220446049250313e-16. It
does not select a replacement layer or imply that an LLM uses these statistical
readouts to generate a personality answer.

## Common source comparison

The same624 Big5-only and6,294 reference-excluded MBTI-only authors,36 complete
comments, official task-specific5-fold outer CV,3-fold inner selection and
declared scaler/PCA/Ridge-or-logistic protocol are retained. Qwen uses the same
1.5B model and pre-question token-weighted body mean/std at four fixed sites.
BGE/E5 use their existing comment mean/std pipeline, not the same architecture
or training objective. These are not controlled model-size comparisons.

| Raw source metric | BGE | E5 | Qwen04 | Qwen10 | Qwen16 | Qwen final norm |
|---|---:|---:|---:|---:|---:|---:|
| Big5 mean Pearson |.176915|.202323|.153841|.150330|.165173|.149203|
| EI AUC |.618367|.639579|.638162|.627869|.625020|.635103|
| SN AUC |.652762|.655146|.644299|.638585|.623809|.614987|
| TF AUC |.748528|.750560|.728028|.735316|.740350|.723842|
| JP AUC |.615172|.615935|.604415|.609213|.611225|.620903|

The largest descriptive Qwen values occur at04 for EI/SN,16 for TF/Big5 and
final norm for JP. Such a rank pattern is not a statistical interaction test,
a selected optimal model, a trait localization result or a universal depth law.

## Complete three-arm comparison

The full packet now covers raw/cue_mask/placebo_mask,90 observer-score rows and
270 paired rows. The same descriptive Qwen site pattern occurs in all three
arms. Pooled deltas against primary Qwen10:

| Contrast | Raw delta [95% CI] | Cue-mask delta [95% CI] | Placebo-mask delta [95% CI] |
|---|---|---|---|
| EI:04-10 |+.010293 [.002571,.017767]|+.013595 [.005955,.021714]|+.012757 [.005027,.020654]|
| TF:16-10 |+.005035 [.001924,.007778]|+.004595 [.001458,.007304]|+.003849 [.000566,.006670]|
| JP:final-10 |+.011691 [.003715,.019450]|+.014130 [.005555,.022308]|+.010736 [.002538,.018537]|

Their mean-official-fold intervals also stay positive in all arms. These are
the same previously highlighted raw contrasts carried through the complete
declared family, not three independent replications or a multiplicity-adjusted
claim. All other positive and negative contrasts remain in the full CSV.

No new layer has a positive pooled interval lower bound versus E5 on any task
or arm. Only EI04/final have positive pooled lower bounds versus BGE across
all three arms. No Qwen Big5 layer's point estimate exceeds BGE or E5 in any
arm. Thus the full results support an outcome-dependent accessibility profile,
not a general decoder-layer advantage or a new selected layer mixture.

## Paired evidence and uncertainty

All six observers share1,000 author-within-official-fold bootstrap draws per
task/arm, seed731. Nine declared contrasts (each new layer versus10/BGE/E5) give270
rows across five tasks, three arms and pooled/mean-fold estimands. These intervals are
pointwise, not family-adjusted, and conditional on saved fitted OOF models.
They do not include retraining uncertainty or prove equivalence when crossing0.
The analysis implementation followed some point estimates; layer conditions
were already declared, not newly chosen from their outcomes.

Pooled differences versus the primary Qwen10 readout:

| Contrast | Delta | Pointwise95% CI | Mean-fold direction |
|---|---:|---|---|
| EI:04-10 |+.010293|[.002571,.017767]|Positive, interval above0|
| TF:16-10 |+.005035|[.001924,.007778]|Positive, interval above0|
| JP:final-10 |+.011691|[.003715,.019450]|Positive, interval above0|
| SN:16-10 |-.014777|[-.026223,-.004579]|See complete CSV|
| SN:final-10 |-.023598|[-.038130,-.008129]|See complete CSV|
| TF:final-10 |-.011473|[-.017028,-.006272]|See complete CSV|
| Big5:16-10 |+.014843|[-.001483,.031207]|No clear pooled improvement|

All other contrasts and negative results remain in paired_layers.csv. EI04
also exceeds BGE by+.019795[.007889,.031453] under this readout, but its point
estimate remains slightly below E5. Final-layer JP has a positive point
difference from BGE/E5, not an established observer advantage. None of the
Qwen Big5 layers has a higher point estimate than BGE or E5 here.

The Big5 average hides different trait behavior: Qwen16 versus10 raises
Extraversion r from.213332 to.260525 and Openness from.170084 to.209744,
while Neuroticism and Conscientiousness decrease. Those are descriptive trait
differences; the mean-score interval is not a trait-specific delta interval.
Raw/cue/placebo reuse these authors and are not independent replications.

## Mechanism interpretation and next decision

The measured object is

```
R(layer, task) = score(y_task, fitted_source_head(layer, task)(pooled_state(layer)))
```

R depends on representation, pooling, compression, regularization, training
sample and external label. Changing layer can reorganize linear accessibility
without increasing all label signals together. Thus the current evidence does
not support a single increasing amount of personality information through
depth. It supports studying outcome-specific readability profiles instead.

It does not identify total information in a layer, the LLM's native decision
rule, causal necessity of a layer, or a unique psychological construct. The
no-PCA result also did not establish compression as the main bottleneck. Keep
the originally fixed10 primary comparison; all three arms are now complete,
do not build an outcome-selected04/16/final ensemble from this table.

Proceed with the already specified task-conditioned decision states, genuine
same-current own/matched-history contrasts, direct trait scoring and same-ready
GRAIN increments. Their comparison addresses whether changing what the model
is asked to compute matters more than selecting another generic body layer.
Bridge outcomes remain closed until source decisions are frozen.

## Artifacts and implementation

Entrypoint: `scripts/analyze_grain_fixed_layers.py`. `--allow-partial` permits
only whole15-head arms, not whichever task or layer happened to complete.
Without the flag, all45 layer heads must be available. The script validates
the full primary audit, sensitivity manifest/code, exact source memberships,
paired IDs/targets/folds, model hashes and all saved-fold replay records.

Full private packet: `results/grain_vector_mechanisms/vm05_fixed_layer_comparison_full_r1`.
Identity: `feabc914359c19f0f3b7fa9a1d7c040898b5558454a56c6a14e6c05d96ad30d7`.
Full `paired_layers.csv` SHA256:
`bd406663b1401bdea5c9d2017793ca804d97fd1ff08c2001cf969f4e3f2b4c75`.
All45 sensitivity source heads/225 replays and complete primary commitments
were revalidated without new fits or outcome-table reads.

Earlier raw-only packet is preserved: `results/grain_vector_mechanisms/vm05_fixed_layer_comparison_raw_r1`.
Identity: `72103af3e46bdb81eea9fcf4033ccc26e1ae72b04717eeeff2b14825f6e78eb2`.
`paired_layers.csv` SHA256:
`cf364026b6f148eaa61a65f2aacc8ab8db6c515b244242f86e97f15b49c9e1f9`.
The packet also contains all30 observer rows, trait-wise metrics, parent/input
commitments and a source snapshot. No new label tables, bridge values, fits or
neural forwards were used by this analysis.

Four focused tests independently reproduce pooled and mean-fold estimates and
bootstrap endpoints with SciPy/sklearn, including tied AUC predictions. They
also test identical-observer zero intervals, mismatched authors and incomplete
arm rejection. The related selection totals16 passing tests, not a full repo run.
