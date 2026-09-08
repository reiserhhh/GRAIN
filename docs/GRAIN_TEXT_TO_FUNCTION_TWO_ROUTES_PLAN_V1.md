# Text-to-function construction: two pending routes

2026-09-09. **PENDING / NOT STARTED.** User requested that both routes be
recorded, then paused research to discuss other questions. M27-M31 remain
complete. Do not create a goal, dispatch agents, run collectors, or resume old
experiments from this document without a subsequent instruction to proceed.

The user's subsequent planning request adds a controlled-world construction
platform for BOTH routes. The [full pending register and execution cards](GRAIN_PENDING_PUZZLE_EXECUTION_PLAN_V2.md)
and [22-source literature synthesis](GRAIN_NAVIGATION_LITERATURE_REVIEW_V1.md)
now specify NAV-01 through NAV-12. Literature/planning are complete; experiments
remain pending. Neither route has been replaced by the synthetic-world idea.

## Objective and starting point

Ultimate objective: explain and independently implement the useful computation
from author text to a specified psychological judgment, rather than retain a
permanent dependence on a full LLM inference pipeline.

Current implemented path:

```
(author text x, target q, condition c)
  -> frozen LLM prefix -> P_q, Q_q
  -> f_q = b_q + <G_q,P_q-P_Rq> + g_q' (Q_q-Q_Rq).
```

P is the ordered question-region hidden-state matrix; Q is the final input
position's hidden state. Their extraction positions are already known. What
remains to discover is the computation producing their useful information.
Attention heads are parts of the LLM, not an already independent substitute;
our Q coordinate is not the attention query matrix in the term QK.

## Route A: reconstruct the useful state-producing computation

Trace how context evidence is selected, transformed, combined and written into
P/Q. Start backward from the frozen G/g readers to locate influential attention
heads and paths. Examine QK selection and OV writing, together with relevant
feed-forward, residual and normalization operations. Attention weight alone is
not the proposed criterion for selecting a mechanism.

Build an explicit independent procedure:

```
T_state(x,q,c) -> P_hat_q, Q_hat_q -> unchanged f_q -> score.
```

Deliverables after resumption: an input/dependency graph, candidate necessary
operations, executable state constructor, and paired comparison with the old
prefix. Preserve contextual interactions rather than assuming heads operate
independently. A small circuit still requiring the full prefix to create its
inputs is an intermediate result, not runtime independence.

## Route B: reconstruct only what the fixed function needs

Do not require recovery of every coordinate in P/Q. The current affine score
uses only these two projections:

```
z_q1 = <G_q,P_q-P_Rq>
z_q2 = g_q' (Q_q-Q_Rq)
f_q = b_q + z_q1 + z_q2.
```

Seek a direct procedure:

```
T_projection(x,q,c) -> z_hat_q1, z_hat_q2 -> unchanged sum -> score.
```

This is an algebraically sufficient reduction for this fixed scoring function,
not a claim of statistical sufficiency for personality. Derive the procedure
from evidence selection, token/utterance interactions and target-directed
transforms identified in Route A, or test independently motivated explicit
operators. Do not replace mechanism discovery with a human-label accuracy grid.

Deliverables after resumption: explicit projection operators, evidence-change
responses, score reproduction and external-link retention. Retain M/C/Q
bookkeeping where feasible; collapsing to a scalar does not by itself explain
which relations produced it.

## Shared progression and decision

Both routes are retained. A offers state-level explanation and potential reuse;
B targets a smaller task-specific independent computation. Neither is already
proven to be possible for all targets. They can share localization evidence
without forcing identical intermediate coordinates or a single universal scalar.

Use controlled event/role/opportunity worlds to identify local computations.
Compare the existing real-text reference functions with separately constructed
synthetic-only reference functions before transport to real authors. Synthetic
event facts are known; psychological attributions are model responses to explain,
not generator-supplied human personality ground truth. After each useful local
map, test real-text connection rather than finishing all targets first.

After the user resumes: choose a bounded set of already useful target functions
and independent reference materials; freeze their existing readers. Locate a
useful upstream operation, implement it, then test changes in evidence and
scores. Compare both routes before expanding target coverage. Exact budget,
targets and numerical operating tolerances are to be set before execution,
not retrospectively from human-label correlations.

Use three distinct result columns: native/function behavior, response to natural
evidence edits, and retained human external association. Record compute and
runtime dependencies. A new-text program that independently supplies the needed
information and retains useful relations closes an end-to-end computational
path for those targets/materials, not all personality inference universally.
LLM-assisted construction can remain an offline research tool; embeddings,
tokenizer tables and copied model weights must be declared rather than hidden
under the term "mathematics only".

## Existing open work remains pending

Narrow-target interventions, E/EI wording-versus-coding, shared-context caching,
fixed-reference expansion and other M31 open questions remain PENDING. They may
support the two routes later, but are not automatic prerequisites or running jobs.
No new training, API upload, private-data use, or old lockbox opening is authorized
by recording this plan. Current coefficients and completed evidence stay intact.

## Method pointers

- [Current construction and equation atlas](GRAIN_ALL_FACTOR_MATHEMATICAL_ATLAS_M31.md).
- [Completed all-factor round](../reports/grain_vector_mechanisms/ALL_FACTOR_ROUND_DELIVERY_REPORT.md).
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): distinguishes attention routing (QK) from information writing (OV); motivates analysis, not proof that attention alone replaces the full prefix.
