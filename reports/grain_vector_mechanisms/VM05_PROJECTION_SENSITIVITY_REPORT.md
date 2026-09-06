# VM05 Big5 projection sensitivity: original columns versus PCA

2026-09-07, round31. **All nine predeclared projection comparisons completed.**
No comparison has a pointwise95% interval excluding zero. This closes the
original-dimension Big5 sensitivity, not the45 fixed-layer comparisons, other
readouts, source controls, bridge evaluation or the whole research program.

## Matched evidence

All comparisons use the same624 Big5-only authors,36 complete comments per
author, task-specific official5 folds and source-inner3-fold head selection.
The candidate removes PCA from the same scaled feature columns and refits the
same Ridge grid. The baseline is the accepted primary PCA16 protocol. Thus this
tests the entire compression/regularization readout, not an isolated causal
effect of deleting vector information. BGE/E5 comment aggregation and Qwen
token-weighted body aggregation are unchanged within each comparison.

All nine candidate heads and45 saved-fold replays were verified. The complete
45-head primary audit is an immutable parent. Exact authors, outcomes, folds,
feature-input bytes, common/source material and projection flags must match.
There are no new label-table reads or model refits in this analysis: it reuses
the permitted source OOF outcomes, not bridge or MEPS psychological outcomes.

## Results

Raw-text pooled mean Big5 Pearson, with paired delta = original columns - PCA:

| Observer | PCA | Original columns | Delta | Paired95% CI |
|---|---:|---:|---:|---|
| BGE |.176915|.160954|-.015961|[-.048825,.021530]|
| E5 |.202323|.181133|-.021190|[-.053796,.015304]|
| Qwen block10 |.150330|.152780|+.002450|[-.031705,.041042]|

The mean-official-fold deltas are respectively-.025750[-.059574,.010539],
-.030359[-.063388,.006379] and-.002555[-.037826,.035689]. The small positive
pooled Qwen difference is therefore not even positive under both estimands.

Cue-mask pooled deltas are BGE-.015821, E5-.021457 and Qwen+.005997;
placebo-mask deltas are-.014668,-.018960 and+.002401. All18 intervals across
nine pairs and both estimands cross zero. No undefined bootstrap draw occurred.
Raw/cue/placebo are dependent views of the same authors, not three replications.

Intervals use1,000 shared author-within-official-fold bootstrap draws, seed731.
They are pointwise and conditional on the saved fitted OOF heads. They do not
include repeated training/selection uncertainty and are not equivalence tests.
The paired-analysis implementation followed observation of the point estimates;
the nine projection conditions themselves were specified before this analysis.

## Mechanism decision

The data do not support the simple rescue explanation that retaining all
columns makes this Qwen body readout substantially better than PCA. They also
do not prove that discarded components contain no personality-related signal.
Removing PCA changes estimator variance and selected regularization; weak
linear-readout performance is not a statement about all recoverable information.

Keep PCA as the declared primary protocol; do not select a new dimension or arm
from these results. Continue the already specified fixed layers, same-ready
GRAIN increments, task-conditioned states and own/matched-history contrasts.
The scientific distinction remains information present, information readable by
a specified head, and information used by the LLM's own output computation.

## Reproduction and scope

Entrypoint: `scripts/analyze_grain_projection_sensitivity.py`.
Private packet: `results/grain_vector_mechanisms/vm05_projection_sensitivity_r1`.
Identity: `1ed460b5b01f407849ad829eb088a6d943e7a137b630cecc37872eb693b8f558`.
`paired_projection.csv` contains all18 rows, SHA256
`7312785141506c8da8cc9512db01c04f2e0185d4ed4eb9ddad4661fa166dd0e5`.
Manifest binds the nine candidate packets, their models/replays, full primary
audit, exact sensitivity grid and implementation. Source snapshot is included.
The three focused tests and related primary/sensitivity tests pass:12 total.
These tests are not additional empirical evidence or a full repository test run.
