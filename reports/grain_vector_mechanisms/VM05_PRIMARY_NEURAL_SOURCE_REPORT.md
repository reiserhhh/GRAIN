# VM05 primary neural readouts: first complete source triads

2026-09-06, round29 extension. **Partial comparison:6 of45 primary heads accepted
in this snapshot.** These are complete source cohorts for two tasks, not six
authors, not the completed observer family and not final bridge acceptance.

## Actual source evidence

The strong sparse invocation naturally paused at29/84, with no active child;
the already-running successor acquired its CPU lease and began the frozen45-case
grid. Do not restart the strong batch during this lease. No live scoring file was
edited. The new script reads completed source OOF packets and completed saved-head
replays only; no new fits, label table reads or bridge outcomes are involved.

All observations below use the same36 comments per author, same task-specific
official outer folds,3 inner folds, fold-fit scaler/PCA16 and fixed head grids.
Big5 uses624 Big5-only authors; EI uses6,294 MBTI-only authors, excluding the99 old
reference authors. BGE/E5 encode comments and aggregate mean/std; Qwen block10
uses token-weighted pre-answer body mean/std. This is a comparison of complete
observer/readout pipelines, not a controlled change of model size alone.

| Task / metric | BGE | E5 | Qwen block10 |
|---|---:|---:|---:|
| Big5 pooled mean Pearson | .176915 | .202323 | .150330 |
| Big5 mean official-fold Pearson | .194453 | .218432 | .162815 |
| EI pooled AUC | .618367 | .639579 | .627869 |
| EI mean official-fold AUC | .618088 | .639926 | .629175 |

Each of the six heads has five independently replayed saved fold models; maximum
absolute prediction discrepancy is1.1102230246251565e-16. This establishes stored
model/output correspondence, not psychological validity or causal mechanism.

## Paired differences, not subtraction of separate confidence intervals

All three observers share1,000 author-within-official-fold bootstrap draws, seed731.
Intervals below are pointwise and conditional on these fitted OOF heads. They do
not include model-refit variation, offer simultaneous family coverage, or replace
the planned full selection-aware nulls. The comparison implementation was written
after seeing the first source results; no new confirmatory registration is claimed.

| Contrast | Pooled difference [95% CI] | Mean-fold difference [95% CI] |
|---|---|---|
| Big5 Qwen - BGE (planned primary) | -.026585 [-.058784,.002071] | -.031638 [-.063114,-.002410] |
| Big5 E5 - BGE (secondary) | +.025408 [.004469,.046188] | +.023979 [.002893,.043734] |
| Big5 Qwen - E5 (secondary) | -.051993 [-.079467,-.025745] | -.055617 [-.082170,-.027726] |
| EI Qwen - BGE (planned primary) | +.009502 [-.002748,.020852] | +.011088 [-.001228,.022704] |
| EI E5 - BGE (secondary) | +.021212 [.011933,.030662] | +.021838 [.012640,.031236] |
| EI Qwen - E5 (secondary) | -.011710 [-.023462,-.000794] | -.010751 [-.022600,.000071] |

Do not choose pooled versus mean-fold after seeing which interval excludes zero.
Different fold calibration/composition can change these estimands; both are shown.

## What this changes, and what it does not

1. These two source tasks do not establish an advantage for Qwen body pooling.
   E5 currently reads more outcome-related information under the fixed PCA16/head
   protocol. This is not a claim that E5 reasons better or that a larger LLM would
   have the same result. The Qwen checkpoint is the existing pinned1.5B observer.
2. The Big5 trait pattern is not uniform: Extraversion r=.269/.301/.213 and
   Openness=.222/.266/.170 for BGE/E5/Qwen; Conscientiousness=.067/.048/.019.
   These descriptive contrasts suggest trait/readout-specific accessibility, not
   a single scalar amount of personality information. They are not a new layer
   or trait-selection rule.
3. Body pooling, final-decision states, own-history effects and the model's direct
   trait answer are different measurements. The first cannot settle whether
   task conditioning/history retrieval improves the latter. Continue the frozen
   task/history/direct comparisons rather than declaring reasoning disproved.
4. The restored original-dimension Ridge and fixed-layer sensitivities remain
   needed. Weak PCA16 readability cannot show the original vector lacks a signal.
   The old PANDORA headline scores use different cohorts/budgets/protocols; no
   claim of breaking the historical .196/.246/.293 targets follows from this table.
5. Cue/placebo arms, remaining axes, same-ready GRAIN increments, stronger source
   controls/nulls and source-frozen bridge still constrain interpretation. Do not
   switch the production embedding, change prompts, blend heads or train a new LLM
   from this partial table. Continue collection and complete the registered grid.

## Artifacts and reproducibility

Input queue: `results/grain_vector_mechanisms/vm05_full_neural_source_heads_r1`,
identity `2dc7c3334582497d13257073b65025db61dd0892e38bbbcfdc3bb6f79196bd27`.
Paired packet: `vm05_primary_neural_comparison_partial_r2`, identity
`179723b32f8c5efd177d7e707a4269297a6da806306bcbf26b15403076e230dd`.
Its three declared CSVs are observer_scores, metrics_by_trait and
paired_observer_comparison. The latter SHA is
`c52b960b8cacb67350bb5011206dc493c3a957c4ca641370712a83616a4ca6bd`.

The initial r1 packet inadvertently included macOS AppleDouble sidecars in its
output-hash list. r2 uses an explicit three-file allowlist; all three scientific
CSVs are byte-identical to r1. r1 is preserved as historical, not overwritten.
Metric tests independently compare centered-dot Pearson/tied-rank AUC with SciPy/
sklearn, including exact identity intervals, folded estimands, missing families
and filesystem sidecars. These do not replace the saved-model real-data replays.

Entry: `scripts/summarize_grain_neural_source_primary.py --batch-dir ... --views ...
--preflight-dir ... --output-dir <fresh-output>`. Add `--allow-partial` only for an
explicit partial snapshot such as this one; without it, all45 completed/replayed
heads are required. No individual author IDs, raw text, vectors or scale values
are included in this public aggregate report. The whole research program remains
active and incomplete.
