# P1B: role, opportunity, and evidence-location mechanism panel

2026-09-07. Exploratory synthetic mechanism experiment, not natural personality
prediction. Materials and source snapshot are fixed before the neural run.

## Scope and split

Use the existing pinned Qwen2.5-1.5B-Instruct checkpoint, Mac MPS FP32/eager,
block16 only. No training, API, downloads, private text or psychological labels.
Six new development settings and six new evaluation settings are disjoint; they
remain a small authored material family, not six independent human samples.

Development: 6 x self/other/behavior x high/low =36 prompts. Estimate each paired
high-minus-low direction at the **evidence endpoint before the question**. This
same direction is inserted at evidence-end or decision, so a position comparison
does not silently change direction. Normalize self/other/behavior to the median
development direction norm and apply signed0.5 dose. One deterministic equal-norm
random control is shared across positions. No post-evaluation orthogonal deletion.

Evaluation: two explicitly named people with opposite actions; target behavior
high/low swaps preserve the word bag. Balance target name and record order across
settings. Do not use nested first-person quotations. Main48 prompts cross six
settings x two target levels x question polarity x A/B answer convention. A=Yes,
B=No versus A=No,B=Yes are correctly recoded into the same target direction.
12 unrelated venue questions keep an affirmative true answer; they are a drift
control, not a complete factorial semantic-specificity test.

24 additional factual owner queries ask separately whether target or companion
initiated. Their expected direction follows the queried person, not just action
words. These fit within the seminar instrumentation reserve; never filter the
trait panel by their success.36 opportunity prompts hold the target action fixed
while stating free choice, assigned behavior, or a neutral session description.
Single actions operationalize narrow attribution, not true Extraversion.

## Measurement and checks

Score = log p(high-semantic answer) - log p(low-semantic answer). Save candidate
IDs, logits/full candidate scores, candidate probability mass when single-token,
prompt hash, evidence span indices, norms, call count and timing. A/B code understanding
and event ownership are reported separately from trait attribution. No silent truncation.

Main intervention48 x3 directions(self/other/random) x2 signs x2 positions =576.
Unrelated12 x2 directions x2 signs xdecision =48. Natural full-evidence donor and
self-donor12 x2=24 restore/coverage controls use only the main positive-question,
canonical-code cell. Swapped evidence must have equal token-span length and identical
outside-span tokens; otherwise retain explicit failure and do not interpolate it.
An evidence-only donor must not rewrite the question or answer. Span recovery and
single-position dose are different interventions, not equal-energy alternatives.

Full packet <=2000 actual forwards, <=1200 seconds and <=9GiB allocated MPS memory.
Conservative row cost<=3 baseline and<=2 intervention; group admission checks precede
complete paired blocks, and every actual `_forward` is counted. Native/material-only
modes are separate fresh packets, never presented as a complete intervention run.
Self-patch and restore maximum logit discrepancy must be<=1e-5. Any budget/technical
stop preserves outputs. No automatic restart or larger-model fallback.

## Readout and branches

Report all native high-low gaps, absolute expected-direction rates, owner accuracy,
four question/code cells, venue drift, self/other/random signed slopes and semantic
gap changes. Bootstrap unit is scenario; pointwise uncertainty is descriptive and
does not include direction fitting, selection or human sampling uncertainty.

- Native role fails: investigate explicit entity binding before calling a trait
  null meaningful. Preserve the failed material and change no labels/directions.
- Only decision score moves: test output preparation vs distributed evidence,
  not a universal personality axis.
- Natural donor recovers while endpoint does not: locate distributed evidence;
  do not stop the whole hypothesis on a single-position null.
- Opportunity changes high-low separation: compare contextual-gain and semantic
  relation explanations; a response difference alone does not establish Bayesian reasoning.
- Stable query-selective action supports the next bounded functional-map packet.

Research value is checked after the native panel and after each complete scenario.
Do not use a successful early p-value to stop or suppress later cases. A technical
failure can stop immediately; a prioritization stop retains partial status and
cannot claim full-grid acceptance or that all remaining outcomes would agree.
