# F08 reader and qualified-history recovery

## Corrections to old execution

The legacy runner gave oracle/soft/hard the same full-material reader Brier scores and did not compute their distinct downstream targets. Its prefix-contract test was a literal True. Some old correction templates denied saying a sentence without specifying the actual corrected behavior, yet assigned an actual pole. These outputs are retained but are not acceptance evidence for the repaired downstream mechanism.

The new material bank explicitly separates own actual behavior, another person's quoted behavior, hypothetical behavior and corrected own actual behavior. A correction denies the opposite action and affirms the actual action. Negativity is not disqualification. The renderer accepts only pole/status/family/variant, never author identity, world, seed or hidden parameters.

## Executed reader packet

96 unique materials cover three disjoint framing families x two poles x four statuses x four variants. The bank and SHA256 were saved before fitting. Source has32 materials; validation/application each32. Core polarity/ownership phrases are intentionally shared; this is a wrapper-shift synthetic test, not natural-language or unseen-concept generalization.

Source-only word1-2/char2-4 TF-IDF readers use Ridge alpha1 for pole and logistic C1 for qualification. Raw pole scores, clipped scores and qualification probabilities are saved. No API, LLM, external model or private text is used.

| Role | Pole Brier | Qualification Brier | Pole accuracy | Qualification accuracy |
|---|---:|---:|---:|---:|
| Source | .070669 | .032712 | 1.0 | 1.0 |
| Validation | .062190 | .029128 | 1.0 | 1.0 |
| Application | .062308 | .029439 | 1.0 | 1.0 |

Perfect classifications on these transparent materials do not imply perfectly calibrated probabilities. Soft histories can differ from hard histories even here. The next experiment must compute that difference rather than copy one reader metric into all arms.

Three ID/row permutation checks, ten equal-count calibration bins, two separate confusion matrices per role, word-order reversal and ownership contrasts are saved. Word reversal is a diagnostic, not an assumption of semantic invariance. Qualification zero means missing personal evidence, not a low-pole response.

## History mechanism built and tested

For a frozen source transition T(y|a,b), the repaired filter uses

L_t(y) = q_t [p_t if y=1 else 1-p_t] + (1-q_t)/2,

xi_t(a,b,y) proportional to pi_(t-1)(a,b) T(y|a,b) L_t(y),

pi_t(b,y) = sum_a xi_t(a,b,y).

Full enumeration of all hidden histories matches the recursive pair distribution. With full oracle observations it collapses exactly to the last two observed bits. With q=0 it ignores content, propagates time and adds zero empirical evidence; an entirely unqualified history is marked prior_only. This is a source-transition approximation, not identification of the author's private dynamics.

Six new material/filter tests and thirteen F07 tests pass (19 total). The actual full-cohort B1/B2/M downstream runner and weighted sufficient statistics are still open; no downstream performance claim is made here.

## Next tasks and artifacts

Use the frozen source transition and unchanged original B1/B2/M definitions. Complete-visible arms must replay the original oracle; mixed eligibility needs q-weighted statistics and cannot delete missing observations. Save actual before/after scores for eight status evidence cards, propagate observed reader errors into F07, and retain all failures without prompt repair.

- `grain_research/gptpro_foundation_v3/f08_materials.py`
- `grain_research/gptpro_foundation_v3/f08_filter.py`
- `scripts/run_grain_gptpro_f08_reader_recovery.py`
- `results/grain_gptpro_foundation_v3/F08/reader_recovery_r1`

No parent card is promoted; global goal remains active. No active background job, training of a neural model, MEPS use, lockbox, remote operation, commit or push.
