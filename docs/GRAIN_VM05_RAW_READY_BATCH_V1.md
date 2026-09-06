# VM05 raw same-ready primary execution stage

2026-09-07, round31. This is the first bounded execution stage of the existing
same-ready protocol, not a new feature search or a complete replacement of VM05.
The earlier geometry, source comparison and sensitivity definitions remain intact.

## Fixed comparison

Use the audited `raw_ready_reference_free` population:144 Big5-only and1,539
MBTI-only authors. All methods consume the same36 complete raw comments, with
task-specific official5-fold source CV and3-fold inner selection. No bridge
outcomes, MEPS scales, new LLM weights or additional text are used.

Fit the six already specified comparators for all five tasks:
TF-IDF word+char, BGE, E5, Qwen block10 body, [GRAIN,Qwen10], [GRAIN,BGE].
Fit the formal sorted-distance GRAIN profile on the four reference-excluded
MBTI source tasks too, since the older reference-inclusive heads differ in their
training cohort. This yields34 genuinely new statistical heads, not LLM training.

The Big5 eligible cohort contains no old reference member. Reuse its identical
previously fitted sorted-distance profile instead of duplicating it. Before
execution, its exact profile bytes, source identities, official folds and all
five saved-head predictions must independently replay. A live preflight has
verified144 authors, five folds, maximum error8.326672684688674e-17, zero refits.
The historical manifest retains its true `raw_ready` name; never relabel its
metadata to imply a new reference-free training event.

The formal GRAIN comparator is16 sorted frozen landmark distances, not the
entire present GRAIN4W theory. Dense scaling/PCA follows the existing fold-fit
protocol; on this small Big5 cohort the available components can be below16.
The fixed joint concatenation can privilege its larger block under joint PCA.
These are limitations to report, not reasons to quietly tune feature weights.

## Resource and artifact discipline

`scripts/run_grain_ready_primary_batch.py` waits for all54 sensitivity conditions
to complete/replay and for the owner process to exit. It then obtains the
sensitivity, primary and strong-source CPU leases. It does not interrupt or
resume any predecessor. Do not manually restart those batches while this stage
owns the leases. Waiting is at most3h; a fitting invocation is at most8h, with
each child bounded to1h/8GiB. Only one fitted child runs at a time.

The waiting supervisor retains only the small input registry, not the complete
text/profile matrices. Full source material loads only after acquiring the CPU
leases. The unchanged one-condition runner performs actual fitting and saved
model replay inside the owned child. Failed/partial outputs remain in place;
resume is explicit and identical, and never silently retries a partial fit.

Each condition retains its original `fit/` packet, source-only label audit,
47 target-support partitions, five saved-fold models, final source head,
selections, private OOF predictions, trait/fold metrics, replay and hashes.
The batch verifies exact task/feature/cohort/arm/projection/readout, eligible
authors/folds, full-head digest and all five replay results before counting it.
The manifest binds input files, original Big5 parent and scoring implementation.

## Acceptance and remaining scope

COMPLETE for this stage means34 accepted new heads plus one independently
replayed Big5 parent. It does not mean a positive GRAIN increment or final
scientific acceptance. Next compare paired [g,q]-q and [g,b]-b as well as their
differences from g, with pooled and mean-fold estimates. Adding q to a weak g
does not by itself show that g has unique information. Retain all tasks and
negative results; do not select axes, blend weights or a layer using bridge data.

Still separate: all-arms-ready raw/cue/placebo, reference-inclusive sensitivity,
the declared no-PCA/readout sensitivities, discussion/source controls, full nulls,
history/direct scoring, source freeze then bridge, and final VM07 synthesis.
The complete45 fixed-layer sensitivity analysis can run from saved OOF while
this independent, already specified source-fitting stage proceeds.

Executable inputs are the previously verified directories listed in
GRAIN_VM05_SAME_READY_EXECUTION_V1, plus `--predecessor-dir` pointing to
vm05_neural_sensitivities_r1, `--primary-dir` to vm05_full_neural_source_heads_r1,
`--resource-dir` to vm05_strong_source_controls_r1, and `--big5-profile-parent`
to vm05_profile_heads_raw_r1/big5_raw_sorted_distances_pca. Use a fresh private
output directory. Resume uses the identical arguments and `--resume-completed`.
