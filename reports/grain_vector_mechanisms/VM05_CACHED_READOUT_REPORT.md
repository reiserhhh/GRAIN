# VM05: separating content pooling, task readout and unobserved history effects

2026-09-06. **Instrument qualification complete; external increments pending.**
This implements the original equal-view and post-question sensitivities without
new model calls. It does not add a discovered personality direction, change the
primary block10 body candidate or replace the current GRAIN scorer.

## Completed and independently reproduced

The original12 technical bundles (four per source group, including four bridge
bundles without outcome values) contain three12-comment views and two questions.
At each of four frozen sites, raw/cue/placebo caches were transformed into:

- Existing token-weighted body mean/std and the equal-view counterpart.
- Separate equal-view mean, within-view RMS deviation and between-view deviation.
- Behavior-question, disposition-question and paired task-difference mean/std.
- One site-independent four-scalar Yes-minus-No logit-margin diagnostic per arm.

The questions concern concrete social initiation versus a general tendency to
initiate, not five Big5 or four MBTI questions. Native margins are not calibrated
trait scores. Three views do not create three independent participants.

All75 matrices/900 author-channel rows were independently reconstructed using
second moments and direct task arithmetic rather than the producer's centered
variance implementation. Maximum error1.95e-13. Existing token pooling replays
within1.17e-13; the total-variance identity closes within4.55e-13. Zero outcome
conversion, neural forwards and model refits. No numerical repair or relaxed
threshold was needed for this actual technical packet.

The full-cache reader and same-ready runner now accept explicit readout identities.
They verify completed observer caches and preserve exact author order. A real
preflight against the still-incomplete full queue was deliberately refused before
label conversion or any head: `All72 observer shards must finish before comparison`.
The stored FAILED_STOP in that guard test is the expected refusal, not a failed
personality experiment. Full-source secondary-readout orchestration remains open.

## A descriptive geometric clue, not a new hypothesis confirmation

After qualification, we examined the mean task displacement for each technical
bundle, D_u=mean_v(d_disposition-d_behavior). The standard identity

```
sum_u ||D_u||^2 = n ||mean_u D_u||^2 + sum_u ||D_u - mean_u D_u||^2
```

separates a common vector component from bundle-dependent residual energy.
This is an explicitly post-hoc technical diagnostic. All three arms/four sites
are saved, not only the favorable row. Raw-arm results:

| Site | Common task-displacement energy | Between-bundle residual energy | Between-view share of body variance |
|---|---:|---:|---:|
| block04 | 71.98% | 28.02% | 0.34% |
| block10 | 84.55% | 15.45% | 0.64% |
| block16 | 96.16% | 3.84% | 0.78% |
| final norm | 87.13% | 12.87% | 1.95% |

The large common component warns against interpreting total task displacement
as person-specific information. It can contain both question effects and the
average response to these excerpts; the calculation does not isolate a pure
prompt effect. The residual can contain content, context and measurement effects,
not just stable author differences. Energy ratios depend on the chosen coordinate
metric and reference set. They are not explained personality variance, reliability,
prediction accuracy or an estimate of irreducible psychological noise.

In particular, **do not delete the common component or christen the remainder a
personality factor**. A small geometric component may still carry readable target
information; large variance need not be predictive. Source-only standardization
already estimates its center on each training fold, not on these12 technical
authors. The existing no-PCA sensitivity remains separate from the PCA result.

Body variance is predominantly within the twelve-comment prefix, not variation
between the three prefix means. BGE/E5 cached deviations are over comment vectors,
whereas Qwen body deviations are over contextual token states. Identical text
budgets do not erase this measurement-level difference. The new variance-parts
readout makes it visible, rather than proving one architecture is superior.

## The mechanism still not observed in this full-source cache

Changing the question after the same evidence measures a task-conditioned state
change. Measuring the effect of history requires the **same current excerpt**
under own, matched other and no history. Those counterfactual forwards are not
in the full-source cache. Between-view variance cannot substitute for them.
Existing smaller VM03/VM06 packets retain their scoped functional findings, but
are not full-source context-effect predictors or source-frozen bridge heads.

Next complete the same-author feature comparisons; separately implement the
source-safe current/history contrast and the originally requested direct LLM
scoring control. Neither these two fixed questions nor passed instrumentation
tests close those requirements. No source/bridge labels choose a new site,
layer, common-component subtraction or narrative from the technical diagnostic.

## Reproduction and evidence

- Protocol `docs/GRAIN_VM05_READOUT_SENSITIVITIES_V1.md`, frozen92162dd.
- Operators/independent audit ba3adce; head integration and post-hoc diagnostic419a0df.
- `vm05_cached_readout_qualification_r1`, identity
  `f4cf9aefbd142c913535a5b64dc7ef71958098ec800d5072c63e8668ce2a7505`.
- `vm05_cached_readout_audit_r1`: independent75-matrix reconstruction.
- `vm05_readout_technical_geometry_r1`: all12 arm/site diagnostic rows, identity
  `e149fd04869c6565030017de6129153fb7e8e4098cc832439005a1f1bba71dc3`.
- `vm05_readout_incomplete_queue_guard_r1`: expected pre-label refusal.
-51 related regression tests passed; later14 focused tests include the additional
  energy identity/translation-dependence case. Two historical SWIG warnings.
  Ruff/whitespace checks passed. Not the entire repository test suite.

Private packets are under`results/grain_vector_mechanisms/`. Entrypoints are
`qualify_grain_cached_readouts.py`, `audit_grain_cached_readouts.py`,
`summarize_grain_readout_qualification.py`; pass the frozen technical, qualification
and audit parent paths explicitly and use fresh output directories. No private
author arrays, labels, IDs, text or vector caches should be published.
