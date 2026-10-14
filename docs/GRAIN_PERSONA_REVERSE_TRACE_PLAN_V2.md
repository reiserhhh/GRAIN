# Persona reverse trace V2

2026-09-14. User authorized Codex to continue the missing mechanisms. Claude is
idle/done. V1 outputs remain unchanged; no training, API calls, private data,
lockbox access or publication. This packet is a finite experiment, not a new
accuracy search.

## Question and distinction

V1 read-side patches measured the GENERATOR reading its persona prompt. Here a
fresh READER sees only a generated article and the frozen trait question. Trace
article evidence -> reader residual states -> question-position head outputs ->
trait score. Known persona assignment is a synthetic treatment, not human truth.

## Frozen design before execution

- Reuse expanded V1 generations. Extraversion: named/described x
  opinion/advice/neighbor x seeds 11/23 = 12 high-low pairs. Diagnostic
  neuroticism: same families/tasks x seed 11 = 6 pairs. No outcome filtering.
- Opinion is discovery; advice and neighbor are application strata. These are
  previously inspected materials, not a new confirmatory holdout.
- Use BOTH E and N reader queries. Record original full-text readouts too.
- Intervention contexts use pair-matched first min(128, high length, low length)
  article tokens inserted into one fixed reader template. Prefix/suffix IDs and
  positions are identical. This is a controlled-prefix estimand, not V1 full-text
  replication; report how much readback it preserves. All raw/token addresses
  are saved. Token-position matching is NOT semantic correspondence.
- Coarse article residual replacements at blocks 0,4,8,12,16,20,24; high-to-low,
  equal-norm random, and self check. Both query scores, answer-alphabet mass,
  raw signed changes and donor contrast recorded. No small-denominator ratios.
- Select one residual block by minimum squared distance to donor readout on
  E/opinion discovery only. Test four contiguous article quarters separately at
  that site on all pairs. These patches are contextual-state interventions:
  a later token state already contains preceding text, not isolated word effects.
- Independently probe downstream question/suffix head outputs at layers 12,16,20
  (fixed from generator V1, not claimed exhaustive reverse localization). Swap
  each of 12 heads, whole layer, self, and random for E-generated pairs/E-query.
  Head selection uses opinion only; retain every head and apply to advice and
  neighbor without reselection. Compare generator L16.H7 and L20.H5 explicitly.
- Bridge V1 endpoint: read block14-generated texts with the selected reader
  head patched from the matched low-text context (identical prefix length).
  Ask whether the reverse head mediates part of the generation-induced change.
  This is a finite intervention comparison, not an additive mediation fraction.

## Math and decision

For fixed query q, s_q(x)=sum_j softmax(logits_j(x,q))*j/9.
Delta=s_q(x_high)-s_q(x_low), d_k=s_q(patch_k(low,high))-s_q(low).
Report mean(d), mean(d-Delta)^2 and slope sum(Delta*d)/sum(Delta^2),
with pair-block bootstrap (family x situation, seeds kept together). Slopes are
relative reader effects, not personality variance or accuracy. GRAIN geometry,
generator effects and reader scores remain distinct coordinate spaces.

Success at this stage is a nonzero directional reader intervention on application
materials, followed by a generator-to-reader endpoint connection. A mismatch
with generator head identities is informative, not a reason to tune head grids.
No exact inverse or universal persona encoding is assumed.

## Deliverables / budget

New runner and helper/tests; results under persona_reverse_trace_v2/r1;
REVERSE_TRACE_REPORT_V2.md with separate full/prefix, discovery/application,
E/N and strong/weak task results. Incremental JSONL, input/code hashes, score
replay and hook-cleanup tests. One bounded local run up to 45 minutes, no
generation queue. If the bound is reached preserve partial status and assess
decision value before resuming. Do not silently change selection or materials.
