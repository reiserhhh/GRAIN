# GRAIN VM04: Natural-History State Interventions

Status: exploratory protocol fixed before the first natural intervention run.
This is a model-computation experiment, not a personality validity test.

## Material and Access

Use the existing VM03 discovery pilot: six authors, two current comments per
author, twelve earlier comments, and two previously fixed donor rules
(length match; condition/length match). There are 24 current-donor pairs and
48 task-pair jobs. Do not select pairs by logits or questionnaire labels.
The 48-author mechanism holdout remains unopened.

The separate VM03 evidence-scope audit has opened one current comment for
each of 48 calibration authors. It supplies source-semantic coding, not
psychological labels. Its thresholds do not automatically transfer to the
different history-bearing prompt used here.

## Questions and Interventions

The target is the existing social-initiative disposition question. A collateral
task asks whether the current comment itself offers guidance/support. Neither
task has a validated natural-personality ground truth in this pilot.
Both tasks have the same system prompt and evidence prefix; instructions differ
only after the evidence. Prefix activations must agree exactly.

Keep current-comment token identities identical under own/donor histories.
Map positions by token order, not global sequence index. Histories can have
different lengths: positional/context-length effects remain an alternative
explanation and are reported, not silently removed.

At fixed decoder blocks 10 and 16, intervene on:

1. All current-comment token states.
2. The last pre-answer decision state.

For each pair/layer let D be the donor-minus-own current-token matrix. Obtain
the top 2/4 right singular vectors Q. Patch effects are:

    H_patched = H_own + alpha * (H_donor - H_own) Q Q^T

Alpha is 0.25, 0.5 or 1. Use this same current-derived basis at the decision
endpoint. This is pair-specific compression, not a universal personality axis.
Controls include exact full replacement, self replacement, zero-effect sham,
two random hidden subspaces of equal rank, and restoration. Random effects
match candidate L2 norm separately at every token before BF16 casting.
Record actual post-cast effect norms and quantization differences.

## Outcomes and Boundaries

Primary outputs: absolute and natural-donor-oriented changes in Yes-minus-No
logit margin; candidate versus matched-random effects; collateral changes.
Recovery fractions are descriptive and only computed when the natural
own/donor difference has magnitude at least one logit. A zero or small
natural contrast must not produce an unstable headline recovery ratio.

Average repeated comments/donors within author before uncertainty estimation.
Six authors give only a pilot-level uncertainty estimate. The nominal number
of intervention forwards is not the independent sample size.

Full current-span replacement is not expected to reconstruct the whole donor
forward: the historical token states are not replaced. Intermediate hybrid
states can be off the natural-state manifold. Model intervention effects
cannot establish human psychological causation or improved trait prediction.

## Resources and Acceptance

Qwen2.5-1.5B-Instruct at the existing pinned revision; Windows .20 CUDA BF16,
eager attention, no KV cache. Preflight picks one fixed execution shape
(1536 if all prompts fit with reserve, otherwise 2048) before any forward.
No truncation, training, model downloads, or shared-service reconfiguration.

Limits: 4096 actual forwards, 30 minutes for the intervention loop.
Checkpoint after every task-pair. A budget stop is PARTIAL, not success.
Self/sham, replay/restoration and pre-question prefix errors must be exactly
zero under the same numerical instrument.

Deliverables: manifest and source archive; private paired-state cache;
interventions.csv, baselines.csv, geometry.csv, controls.csv, run_status.json;
author-clustered summary and a report distinguishing information presence,
functional use, task bias and psychological validity.

## Methodological Precedents

Encoded information and its functional use are separate questions:
[Elazar et al., 2021](https://aclanthology.org/2021.tacl-1.10/).
Activation-patching conclusions depend on corruption and metric choices:
[Zhang and Nanda, ICLR 2024](https://openreview.net/forum?id=Hf17y6u9BC).
Our task-specific donor, norm and collateral controls address part of that
problem, not a guarantee that the intervention identifies a unique mechanism.

## Post-Run Instrument Repair

The first complete run showed pre-cast norm matching below 3e-14 relative
error but endpoint post-cast mismatch up to 24.66% after averaging random
replicates. Current-span mismatch was below 0.05%. Therefore the original
endpoint candidate/random comparison is provisional.

Before repair readouts, fix the following: rerun every endpoint random row
(384), not an outcome-selected subset; keep its direction/rank/seed and scale
the whole row to match the candidate's actual BF16 displacement. Use bounded
40-step monotone search; choose the nearest discrete norm, not coordinate edits.
Desired relative mismatch is <=1% per row. If rounding makes this unattainable,
report NEEDS_RETRY_NORM rather than claiming exact matching.

Replay all 192 endpoint candidate outputs and all 48 original baselines exactly,
with restoration after each job. Budget <=1024 calls and 10 minutes. This repair
does not change target directions, prompts, authors, decision thresholds, or
the original complete/current-span results. Preserve both versions and identify
the repaired rows explicitly in downstream summaries.
