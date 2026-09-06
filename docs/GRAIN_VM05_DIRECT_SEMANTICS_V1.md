# Direct trait semantics and encoding control

2026-09-06. Frozen before any model observations on these cases. Research control
only; no change to GRAIN's deterministic scoring or historical direct prompts.

## Question

Technical replay and answer compliance do not show that a model understands a
score's direction. Can the already-qualified Qwen observer order clearly contrasted
invented behaviors, and does that ordering survive a change in answer encoding?

## Fixed panel

- Nine tasks: five Big5 and four MBTI axes with positive E/N/F/J.
- Two invented low/high behavioral pairs per task, plus a no-evidence prompt.
- Primary wording exactly reuses the frozen direct question and native template.
- Counter: reverse Big5 numeric code meaning and recode the output back; for MBTI
  reverse the order in which preferences are listed without changing their meaning.
- Total90 cases,72 anchor cases and18 no-evidence cases. No-evidence is an output
  prior, not a person with a known midrange score. Do not impose a0.5 target.
- Capture each case twice:180 real forwards, no cross-case caching. Same pinned
  CUDA BF16 observer, native3072-row padding, no truncation or free explanation.
- TF pairs contrast logical criteria with personal values/relationships, not
  intelligence with emotion or moral good with bad. These are intentionally simple
  capability examples, not representative items or new psychometric labels.

## Reports and decisions

Save all logits, full-vocabulary normalizers, candidate probability mass, entropy,
first-token compliance, recoded scores, every pair's high-minus-low difference,
no-evidence priors and absolute recoded counter changes. Treat <=1e-6 as nonpositive
for descriptive counting; it is not a psychological significance threshold.

Two positive pairs only support these two invented examples, not population accuracy.
No p-value, equivalence claim, model selection or new scale follows from this panel.
Numeric-code inversion and MBTI answer-listing reversal are different manipulations;
their absolute effects are not directly comparable. Short invented inputs are not
a matched-budget replication of natural12-excerpt views.

Stop for numerical replay, missing/changed cases, model identity or resource failure.
Record negative or encoding-sensitive semantic results without changing prompts,
flipping axes, discarding tasks or choosing a winning variant. Such a result limits
the interpretation of the existing direct control; full source results are still
required and are not rescued by this panel. Any later redesign is a new protocol.

## Execution boundary

Token-only preflight may run while history extraction is active. Real forwards
wait until that queue is COMPLETE or PAUSED_BUDGET_BETWEEN_SHARDS, its lock absent,
and both owner and last-child PIDs dead. No second loaded GPU model, no training,
model download, paid API, bridge outcomes or MEPS data.

Entry: `scripts/qualify_grain_direct_traits_semantics.py`. The module and invented
panel are archived and hashed with the already-qualified direct code. Full direct
source collection and external evaluation remain separate required work.
