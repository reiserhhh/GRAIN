# P1T: cue diagnosticity versus answer-code dependence

2026-09-07. Freeze before forwards. Decision: current waiting-vs-initiating
material may be an ambiguous behavioral-attribution contrast. Waiting is not
logically equivalent to being reserved, and neither behavior establishes a human
trait. Before fitting a larger subspace, ask whether a clearer observable contrast
changes native descriptions under both answer codes.

## Fixed factorial

Use P1P's four contexts and two positive-action wordings, with room=open only.
Retain the original waiting line or replace ONLY the noninitiator's line with a
prewritten explicit avoidance: declined/refused conversation and stayed silent.
Keep both people, both role assignments, both action-record orders, both existing
event/behavioral-attribution questions, and both A/B answer mappings. No question
optimization and no explicit personality word is added to the evidence.

4 contexts x2 contrast types x2 orders x2 roles x2 people x2 criteria x2 codes
=256 native prompts.32 waiting/original/positive-code prompts are exact P1P
replays. No patches, no fitted heads, no new model, API or private text/labels.
Save all candidate logits, exact token layouts and single block16 target anchors
for future cache analysis. Four clean repeat controls (criterion x code) save raw
logits.260 forwards; cap300calls/180seconds/9GiB allocated MPS; padded256 FP32 eager.
Each complete context contains64 prompts; checkpoint those, not favorable cases.

## Outcomes

Code-aligned margin is code_sign*(logit_A-logit_B). Preserve raw margins. Report
per criterion/contrast/order/code: positive and negative preference rates, all
paired role gaps (signed mean AND RMS), positive-gap counts, and exact code-pair
semantic preference agreement. Report all4 contexts; no pooling away sign flips.
Only event initiation has an explicit authored fact. For attribution, call the
endpoint preference aligned with the authored contrast, NOT accuracy, psychological
truth or Big5 validity. Retain original waiting cases and all native failures.

Independent audit checks unchanged positive lines, changed negative lines only,
role ownership, both orders and codes, real token/anchor offsets, all32 replay
cells, common-prefix identity and raw scalar calculations. No repeated whole run
just to collect favorable native answers. No statistical significance inference
from256 authored prompts as independent people.

## Decision consequences

If avoidance changes attribution ordering/preferences under BOTH codes while
event facts remain readable, weak original cues contributed to prior failures;
do not retroactively change P1P/P1R results. Code-dependent improvements remain an
instrument limitation. If clear cues still fail, further local direction sweeps
have low value for native psychological inference with this observer; document
the exact model/material boundary before considering any separately authorized
observer or real-author extension. Either way, output influence is not human
personality validity. Old queues and released GRAIN scorer remain unchanged.
