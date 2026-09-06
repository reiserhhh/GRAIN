# Full-source neural comparison: bounded CPU handoff

2026-09-06, round29. This executes the already frozen VM05 comparisons; it is not
a new layer search, a new population or a reduced replacement for the whole plan.

## Work and priority

Run all45 primary conditions:3 raw/cue/placebo arms x5 tasks (Big5, EI, SN, TF,
JP) x3 observers (BGE-large, E5-large, Qwen block10). Within each arm, run all
three observers for a task together, beginning with Big5/raw. Exclude the99 old
reference authors. This leaves624 Big5-only and6,294 MBTI-only sources; bridge
outcomes are not read or used for head/feature selection.

Reuse `run_grain_vector_mechanism_external_links.py` unchanged: original source-
specific official5 folds,3 inner folds/seed731, fold-fit scaler/PCA16 and fixed
Ridge/Logistic hyperparameter choices. BGE/E5 aggregate36 comment vectors;
Qwen's primary readout is token-weighted body mean/std. Equal-view/post-question
readouts and same-ready GRAIN increments remain separate required comparisons.
No initial layer other than the predeclared block10 is promoted after outcomes.

## Resource handoff, not concurrent fitting

- The current strong sparse CPU batch retains priority until its current bounded
  invocation naturally stops. Do not interrupt a running condition.
- The new supervisor waits up to3h, polling the specific predecessor PID, verifying
  its command, then requiring its terminal COMPLETE/PAUSED_BUDGET status and dead
  child. A status file alone or an observation timeout cannot trigger a restart.
- Once the predecessor exits, acquire and hold its existing `.execution.lock`
  alongside the new queue's lock. This prevents manually resuming the strong
  batch while the new CPU fit is running. Do not bypass those locks.
- Each fit remains one owned child,4 numerical threads,1h/8GiB bound; follow it
  with an independent saved5-fold replay,10min/8GiB bound. No competing heavy
  Mac fit. Invocation budget8h starts **after** the verified wait.
- On budget pause, preserve completed conditions. Identical explicit resume
  verifies prior results and never overwrites/retries incomplete output. Failures
  publish FAILED_STOP; a second rejected lock owner must not overwrite status.

Windows history extraction continues independently. A current Windows hardware
check found only about7GB free RAM, so this phase does not add an8GiB CPU job or
another model there. No trading service is stopped or reconfigured.

## Evidence and remaining scope

Each accepted condition needs exact source membership/folds, matching observer,
mask, exposure handling and PCA flag, full OOF metrics, source-only full head and
all five replayed saved models. COMPLETE means all45 cases, not merely the first
positive Big5 result. Author-bootstrap intervals remain conditional on fitted
OOF heads; source-specific controls/full199 nulls, neural sensitivities, same-ready
comparisons, history/direct channels and source-frozen bridge remain open.

This queue temporarily takes the next CPU slot after the strong control budget
pause to deliver the primary full-source comparison. It does not cancel the
unfinished84-condition strong controls; resume those after releasing this lease.

Entry: `scripts/run_grain_vector_mechanism_neural_source_queue.py`.

```text
python scripts/run_grain_vector_mechanism_neural_source_queue.py
  --views results/grain_vector_mechanisms/vm05_qualification_r2/common_views.json
  --preflight-dir results/grain_vector_mechanisms/vm05_preflight_r1
  --parent-dir results/grain_vector_mechanisms/vm05_full_source_cohort_r1
  --prepared-root "/Volumes/mobile3/projects/project persona/data_sets/prepared/pandora_official"
  --vector-dir results/grain_vector_mechanisms/vm05_full_vectors_r1
  --predecessor-dir results/grain_vector_mechanisms/vm05_strong_source_controls_r1
  --predecessor-pid 54651
  --output-dir results/grain_vector_mechanisms/vm05_full_neural_source_heads_r1
  --max-wait-seconds 10800 --max-seconds 28800
```

Wrapped for readability; use one argv. PID54651 is the verified owner at this
handoff, not a permanent host identifier. If the owner changes, inspect actual
state rather than reusing this command blindly or overwriting the result directory.
