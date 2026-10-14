# Persona generation trace: authorized execution handoff

Task ID: CLAUDE-PERSONA-GENERATION-TRACE-20260914-01

The user authorizes Claude to complete the experiment designed in
GRAIN_PERSONA_GENERATION_TRACE_PLAN_V1.md and its three-round seminar record.
This supersedes the seminar's read-only/no-generation restriction, not privacy,
lockbox or publication boundaries. Claude owns implementation for this wave;
Codex supervises. Do not resume the old WP2b/c queue: Codex completed it.

## Execute, not another plan-only response

1. Read both design/seminar documents and current repository instructions.
2. Inspect existing generation, activation hooks, frozen GRAIN scoring and API
   utilities. Reuse verified components. Preserve unrelated work and old results.
3. Implement and test the new 88-text pilot, blinded readout, selected causal
   interventions and GRAIN endpoints. Run the pilot, inspect results, then follow
   the planned decision ladder. The pilot is not the entire experiment: proceed
   to the bounded expanded/held-out and causal phases where useful. Do not stop
   after preparing data or writing a report of intentions.
4. Keep per-condition results and all failure/coverage counts. No selecting only
   persona-compliant generations. Use task/prompt-family holdouts for fitting and
   site selection. Reader contexts must not contain persona prompts, filenames,
   condition IDs or generation KV caches.
5. Distinguish free generation from aligned teacher-forced interventions; use
   both poles' continuations. Report causal output changes separately from GRAIN
   coordinate changes measured on regenerated texts. Respect GRAIN minimum text
   requirements; do not invent full scores from unsupported short text.
6. Save artifacts, source/model identity, commands, tests and a concise final
   interpretation. On genuine quota/hardware/material blockage save a resumable
   checkpoint and exact remaining work; complete independent parts meanwhile.

## DeepSeek Flash permission

The user explicitly permits DeepSeek API Flash-model calls for this experiment.
Use /Volumes/mobile3/projects/project persona/.env for existing credentials;
inspect variable names/configuration without printing values. Reuse existing
client code where suitable. Verify the actual Flash model identifier through
provider documentation or the authorized endpoint; do not guess an identifier
or silently substitute a different model. Record resolved model and revision
information available from responses.

API use is restricted to authored/synthetic experiment material. Do not upload
PANDORA, MEPS, X private text, user IDs or psychological labels. Keys never enter
prompts, logs, reports or git. Record token usage, retries and cost if known.
Set finite request/token/retry limits per stage, cache outputs and avoid repeated
paid reruns. Expand only for the written experiment, not an unbounded search.

Flash may generate an additional synthetic corpus, help produce stimuli, or act
as a blinded evaluator. Keep its branch separate from the local writer/reader
and avoid selecting outputs using that evaluator. API-only calls do not expose
internal residual/MLP/head tensors and cannot satisfy the causal localization
phase. Use the existing open local checkpoint for that phase, preferably one
that both generates its own corpus and supplies aligned internal interventions.
Do not claim to have traced DeepSeek's internal heads from API responses.

## Resource and output boundaries

Begin with a smoke test and the 88-text pilot; then execute useful planned phases
with resumable, finite budgets. No new LoRA or large training queue. Default hosts
remain .20 Windows for heavy computation and .30 Linux/Intel for evaluation; verify
availability and hook support before using them. Existing Mac open-model execution
is acceptable for the bounded pilot. No dependency/model download beyond need;
document a missing dependency rather than silently replacing the scientific model.

Outputs: results/grain_vector_mechanisms/persona_generation_trace_v1/ with stage
manifests, samples, splits, readouts, selected activations, interventions, GRAIN
comparisons, API accounting and tests. Write a readable research report and update
the plan's actual execution status. Do not commit, push, tag, publish or open a
lockbox. Success may be a selective shared mechanism or a meaningful dissociation;
do not equate absence of exact inversion with failure of the entire study.
