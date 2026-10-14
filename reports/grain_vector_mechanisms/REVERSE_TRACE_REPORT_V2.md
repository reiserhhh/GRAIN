# Persona-to-text and text-to-trait: reverse mechanism V2

2026-09-14. Codex execution. Status: COMPLETE for the finite packet below.
Protocol: `docs/GRAIN_PERSONA_REVERSE_TRACE_PLAN_V2.md`.

## Why this is a new link

V1's reading heads read a system persona while GENERATING. V2 reads an article in
a fresh context and scores a trait question. Thus the mechanism graph is:

\[
P \longrightarrow H_G \longrightarrow X
\longrightarrow H_R(X,q) \longrightarrow s_q(X).
\]

The user-supplied persona P, the writer state H_G, the article X, the reader state
H_R, and the scalar s_q are different objects. A compatible head number is not an
inverse function; a successful reader intervention establishes a local causal
dependence of that reader score on the replaced states.

## Mathematical object

For the frozen ten digit candidates, with logits z_j and v_j=j/9:

\[
s_q(X)=\sum_{j=0}^{9}v_j\frac{e^{z_j(X,q)}}{\sum_{k=0}^{9}e^{z_k(X,q)}}.
\]

This is a candidate-conditioned model readout; the full-vocabulary probability
mass of those candidates is saved separately. It is not a calibrated human
trait percentile. For donor/recipient text H/L and an intervention I_k:

\[
\Delta=s_q(X_H)-s_q(X_L),\qquad
d_k=s_q(I_k[X_L\leftarrow X_H])-s_q(X_L).
\]

The primary quantities are signed d_k and mean squared reconstruction error
E[(d_k-Delta)^2]. The pooled slope sum(Delta*d)/sum(Delta^2) is descriptive;
small pairwise denominators are never divided individually. Random perturbations
are matched in norm per patched row, not assumed to be pure semantic removal.

The second packet clamps a reader head on writer-intervention texts to its
low-text state. With s_before, s_after and s_low:

\[
\Delta_{writer}=s_{before}-s_{low}
=(s_{before}-s_{after})+(s_{after}-s_{low}).
\]

This exact arithmetic is a finite perturbation accounting identity, not an
identified natural mediation proportion. It connects two actual interventions
without claiming a unique inverse or a trait-specific neuron.

## Materials and comparability

- 18 paired articles from the completed expanded set: 12 E pairs and 6 N pairs;
  two reader queries per pair. Opinion chooses positions; advice and neighbor
  apply them. These materials were seen in V1, so this is exploratory application,
  not an untouched confirmatory sample.
- Whole original texts are rescored. Intervention inputs use identical reader
  prefix/suffix tokens and pair-matched first <=128 article tokens; no generation
  persona prompt, cache, labels, or condition IDs enter the reader.
- Quarter patches replace contextual states. Their ordinal alignment does not
  equate word meaning, and later-quarter states can already encode earlier text.
- Reader head probe layers 12/16/20 are chosen from the generator experiment.
  The best among this restricted set is not the globally best reader head.
- Endpoint connection reuses V1's generated intervention/reference/random arms;
  all four arms in a pair/seed use the same <=96 token prefix length. No new
  generation, API call, training, participant fitting, or lockbox access.

## Numerical closeout

### Main finding: the two interventions connect on full articles

The writer's block14 intervention produced an E readout difference of 0.255862,
exactly replaying V1 on the original complete articles. Resetting the READER's
L12.H0 question-suffix output to the matched low-text state removes 0.078365
[0.053813, 0.101958], leaving 0.177497. An equal-norm random reader perturbation
removes -0.006373 [-0.016281, 0.003535]; self replacement changes zero.
Ten of twelve low-clamp effects are positive. Bootstrap blocks are the four
family/task pairs, with three seeds kept together; these are small synthetic
material blocks, not independent human participants.

| Full-article generator arm | Original E difference | Removed by reader L12.H0 low-clamp | Random reader perturbation |
|---|---:|---:|---:|
| block14 high-state intervention | .255862 | .078365 | -.006373 |
| direct high-persona prompt | .229120 | .079602 | -.000401 |
| random generator perturbation | .094042 | .054345 | -.006116 |

The same reader site responds to expression changes produced by either direct
prompts or writer-state interventions. It also carries readout changes from
random writer perturbations: it is not a detector of the provenance or intent
of a persona instruction. This is a finite computational expression-to-readout
link, not a uniquely persona-specific neuron. The 30.6% ratio of mean removal
to mean contrast is NOT an information/psychological variance share; the pooled
slope is a different statistic (.218768).

On the advice application materials alone, writer contrast is .138992 and the
reader clamp removes .065474 [.042151, .088798]. Opinion was used for head
selection; advice was not. This application reuses previously observed task
families and is not a new confirmatory dataset.

### Reverse localization and generator comparison

The coarse replacement scan selected block00 by donor reconstruction on opinion.
Early full-article-state replacement nearly reconstructs the donor reader score;
this is expected when replacing almost all early text information, not a novel
personality mechanism by itself. The useful localization is the subsequent loss
of leverage at article positions and the causal role of later question outputs.
By block12/16 the E article-position effect in opinion drops from .2037 at block00
to .0495/.0226; in advice it drops from .1677 to .0139/.0032. Information can
already have propagated into the unchanged suffix before a late patch.

Among the fixed head candidates in layers 12/16/20, opinion selects L12.H0;
generator candidate L16.H7 ranks second. Selection was not changed after
application. Whole-article suffix-aligned results (four high-low pairs per task):

| Task | Whole-text E contrast | L12.H0 effect | L16.H7 effect | L20.H5 effect |
|---|---:|---:|---:|---:|
| opinion | .211543 | .0950 | .1056 | .0316 |
| advice | .142826 | .0534 | .0225 | .0350 |
| neighbor | -.003731 | .0094 | -.0052 | -.0004 |

For L12.H0 the matched random effects are .0205, -.0048, .0035. The strong-task
donor effects exceed these controls. Shared head identities do not establish an
inverse; generator and reader use different contexts, units and effect metrics.
These output patches carry all text-caused changes in the relevant head output,
including earlier changes to query states. They do not yet isolate direct
article-K/V -> question-Q edges within that head.

### A discovered context effect: prefixes are not the whole article

On the same four neighbor pairs, original full-text E contrast is -.003731
(AUC .3125), but the controlled first-128-token contrast is +.042353 (AUC .8125).
All full-text scores replay V1 exactly. These tiny-panel AUCs are not new model
benchmark results. The change demonstrates reader sensitivity to coverage and
context completion; it does not show that the omitted suffix is meaningless noise.
We therefore added full-context and full-endpoint packets, with separate
result-triggered protocols, rather than presenting prefix effects as V1 replication.

The matched-prefix endpoint remains saved: contrast .228964, L12.H0 removal
.114063 [.051781, .216530], random -.003097. It is a different material estimand,
not a replacement for the full-article .255862/.078365 result above. Cross-query
N removal is near zero (.000321) on that E endpoint, but the original N contrast
is also near zero; this alone does not establish trait specificity. The six
N-generated diagnostic pairs and all cross-query residual results remain in
the output without rescue or outcome filtering.

### GRAIN interface clue

Post-result descriptive correlations across the 12 E pairs compare complete-text
GRAIN coordinate differences against complete-text reader-head intervention effects:

| Reader head | coordinate 1 | coordinate 2 | coordinate 3 | coordinate 4 |
|---|---:|---:|---:|---:|
| L12.H0 | -.172 | .340 | -.192 | .506 |
| L16.H7 | -.092 | .096 | -.195 | .278 |
| L20.H5 | -.089 | -.077 | -.015 | .214 |

This identifies a candidate interface to develop, not a psychological criterion
correlation or a name for coordinate4. All twelve correlations are retained;
there was no source-only prediction fit or multiplicity-adjusted confirmation.
The context-mismatched full-GRAIN/prefix-reader comparison is retained separately
in audit.json and is not substituted for this table.

### Execution and verification

| Packet | Effect records | Model forwards | Seconds |
|---|---:|---:|---:|
| reverse scan / quarters / heads | 1416 | 1524 | 729.65 |
| matched-prefix endpoint | 432 | 792 | 317.86 |
| full-context application | 108 | 204 | 86.48 |
| full-article endpoint connection | 108 | 204 | 79.28 |

Total substantive model forwards 2724; smoke 103 additional forwards. No new
generation, API calls or model fitting. Twenty-two targeted tests pass, including
head-row scope, hook cleanup on exception, source-only selection and the
AppleDouble metadata regression. All seven new Python files pass Ruff.
Independent arithmetic code checks 2064 effect records, 36 token-address pairs,
selection replay and V1 full scores: maximum arithmetic error 1.11e-16, self
effect zero, old readout replay error zero. This is same-agent independent
implementation checking, not third-party review.

The first address-audit attempt encountered macOS `._*.json` metadata. The audit
now excludes only those sidecars; malformed scientific JSON still fails. No
scientific inputs or neural results were changed to resolve this error.

## Theory update and remaining next step

The finite bridge now has both a writer intervention and a reader intervention:
persona -> writer state -> generated expression -> reader head state -> trait
readout. The supports overlap in part but are context-dependent and not uniquely
invertible. GRAIN supplies an additional description of expression, not an
automatic psychological meaning for every coordinate.

Next is one bounded Q/K/V decomposition at L12.H0, with L16.H7 as a reference:
determine whether the effect is carried by changed article values, changed
attention allocation, or text-conditioned query states. Use a fixed input
address scheme and source-only direction construction, then application on
advice without reselection. This should test a shared low-dimensional action
operator rather than sweep more heads, traits, prompts or model accuracy.

Still open: direct edge/source-token attribution, generator-reader vector
transport (the present connection passes through actual text), a cold compiled
reader independent of the LLM forward pass, broader persona/real-author use,
and stable GRAIN-to-action mapping. This packet does not close those by renaming
its output as a complete theory. No queue remains running after closeout.

## Reproduction

Use the existing project-persona `.venv` Python from the independent GRAIN root:

```bash
python -m pytest tests/test_grain_persona_reverse_trace.py tests/test_grain_persona_generation_trace.py tests/test_grain_direct_traits.py -q
python scripts/run_grain_persona_reverse_trace.py --smoke --output results/grain_vector_mechanisms/persona_reverse_trace_v2/smoke --max-seconds 300
python scripts/run_grain_persona_reverse_trace.py --max-seconds 2700
python scripts/run_grain_persona_reverse_endpoint.py --max-seconds 1800
python scripts/run_grain_persona_reverse_full_context.py --max-seconds 900
python scripts/run_grain_persona_reverse_full_endpoint.py --max-seconds 900
python scripts/audit_grain_persona_reverse_trace.py
```

Resume requires matching code/input/protocol hashes. Primary outputs:
`results/grain_vector_mechanisms/persona_reverse_trace_v2/r1/` and `endpoint_r1/`.
Full-context companions: `full_context_r1/`, `full_endpoint_r1/` under the same
root. V1 raw outputs and code were not edited. No automatic commit/push/release.
