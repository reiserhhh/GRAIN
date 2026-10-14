# Persona trace: from local mechanisms to measurement construction

2026-09-14. Codex / Claude read-only seminar. Claude model: claude-fable-5-1,
effort max; resumed existing session 3a2ae5ad-4b1d-4e46-9b28-669b1b3b7495.
No new experiment, API judgement, training, release or lockbox access.

## User objective

Develop the theoretical foundation and construction procedure for interpretable,
context-specific text-based psychological measures. Neither finding more heads
nor reproducing model judgements is the endpoint. GRAIN itself remains revisable.
The question is whether a finite set of informative interventions can teach a
reusable construction rule, instead of enumerating traits, tasks and components.

## Evidence that motivates the change

Source: reports/grain_vector_mechanisms/REVERSE_TRACE_REPORT_V2.md.

- Writer intervention creates a full-article E readout contrast .255862.
  Resetting reader L12.H0 removes .078365 [.053813,.101958]; random reader
  perturbation removes -.006373. This connects writing and reading through text.
- That reader head also carries changes from random writer perturbations.
  Its identity is not a personality label, nor evidence of a unique inverse.
- Task and phrasing matter. On four neighbor pairs the full-text contrast is
  -.003731, while its first-128-token version is +.042353. Coverage is substantive.
- GRAIN coordinate 4 versus H0 effect r=.506 is a twelve-pair, post-result
  interface clue, not a psychological validity estimate or a factor name.

The local causal link is useful experimental leverage. By itself it is not the
new psychometric construction theory sought by the user.

## Seminar and corrections

Round 1: Claude proposed a shared low-rank reader action model and a synthetic
instruction-responsiveness profile for selecting measurement conditions.
Codex agreed these are potential construction objects, but challenged inference
from a task null to absent expression, and from assigned persona to known latent
personality. Shared versus specific actions can coexist, rather than form a binary
fork. Subspace overlap alone is insufficient without score-changing intervention.

Round 2: Claude accepted those corrections and required the proposed evidence
coordinates to be computable from text. Otherwise the program becomes an
LLM-internals investigation rather than a GRAIN measurement method. Claude
recommended an existing-cache two-endpoint comparison before expensive alignment.

Codex's final corrections to that proposal: GRAIN displacement is independent of
the reader but not specific to the target trait. Both endpoints being small cannot
establish absence of expression. A large GRAIN displacement with a null reader
does not establish reader error. These patterns select a next intervention; they
do not identify the explanation. Low dimensionality is still a hypothesis.

The expanded JSONL was counted directly: 2 axes x 3 wording families x 6 tasks x
2 poles x 3 seeds = 216 trait texts, plus 54 controls = 270. Each trait cell has
THREE paired contrasts, not the nine pairs proposed in Claude's second reply.
No cell-wise covariance inversion or precision claim is justified by that count.

Round 3: Claude explicitly accepted the count correction, triage-only reading,
surrogate remainder, shared-plus-specific model and one withheld-condition action
test. Its concluding phrase "two independent endpoints" means separate computation
channels only: both derive from the same synthetic text and are not statistically
independent evidence or independent human validation.

## Candidate methodological advance

Change the unit of discovery from a named head or isolated factor to a
**conditional evidence-to-response function**. A candidate local model is

\[
e=\phi(X,c),\qquad
\widehat s_q(X,c)=b_q(c)+[a_{q,\mathrm{shared}}(c)+a_{q,\mathrm{specific}}(c)]^T e.
\]

Here X is the observed text, c its condition, q the target query, and e a frozen
text-computable representation. The shared/specific parameterization requires
explicit constraints and interventions; it is not a uniquely identified set of
psychological axes. The nonlinear alternative is F_q(e,c).

For fixed c, a differentiable F gives the exact finite-path identity

\[
F_q(e+\Delta e,c)-F_q(e,c)
=\int_0^1 \nabla_e F_q(e+t\Delta e,c)^T\Delta e\,dt.
\]

Approximating the integral by a local linear action is a testable approximation.
Explaining the actual LLM contrast additionally requires a residual:

\[
\Delta s_q=\Delta F_q+\Delta r_q.
\]

This separates observed evidence movement from the scoring function's sensitivity
without assuming the current four GRAIN coordinates preserve everything. It does
not identify human personality or the generator's latent state from two endpoints.

The proposed contribution is a procedure that learns such actions from controlled
contrasts, makes them text-computable, and predicts which new evidence/conditions
will change a measurement and why. Actual gain over ordinary regression and
existing representation methods must be demonstrated, not inferred from this
standard calculus identity.

## Relation to predecessors

- [DAS, Geiger et al., 2024](https://proceedings.mlr.press/v236/geiger24a.html):
  learns distributed causal alignments with gradient descent rather than brute
  force over neuron sets. Reuse this idea; do not claim learned alignment is new.
- [Representation Engineering](https://arxiv.org/abs/2310.01405): places
  population-level representations rather than single neurons at the center.
  Population representation analysis alone is not our unique contribution.
- [Explanatory Item Response Models, 2004](https://link.springer.com/book/10.1007/978-1-4757-3990-9):
  models person, item and interaction predictors in repeated observations.
  Condition dependence and measurement-design effects are established ideas.
- [pyvene](https://stanfordnlp.github.io/pyvene/): existing intervention tooling;
  use or adapt rather than rebuilding a general interpretability platform.

Candidate distinct value: linking intervention-calibrated, open-text evidence
coordinates to a transferable measurement-construction procedure with population
reference and interpretable evidence provenance. This is a research claim to test,
not a completed novelty assessment or a promise of publication.

## Minimal next sequence, not an exhaustive queue

1. Use the 270 existing texts and cached GRAIN/local-reader/Flash endpoints to map
   signed within-cell differences. Preserve original texts and all refusals;
   length/refusal analyses are separate sensitivities, not silent deletion.
   Use a fixed source reference metric, not a three-pair cell covariance inverse.
   Output identifies evidence/reader agreement and disagreements, not true trait
   presence. This is a cheap routing step, not another flagship experiment.
2. Select one informative mechanism contrast, with selection recorded. Fit one
   bounded text-computable action model on source task/wording groups. Test on
   withheld groups; apply learned versus random/equal-rank actions to actual
   reader computation. Compare the same materials/budget to direct response
   regression. Cache availability determines whether new bounded extraction is
   necessary; do not promise zero model calls before checking.
3. The decision is transport of signed effects and finite responses, including
   residuals, not a personality-AUC threshold. If the action transports and gives
   leverage beyond regression, develop the construction procedure. If only the
   teacher score is reproduced, retain a distillation result without calling it
   a new measurement theory. If transfer fails, distinguish missing evidence from
   a condition-dependent/nonlinear action using one targeted contrast, not grids.
4. Only after that, connect frozen evidence responses to authorized natural-author
   data and appropriate external/behavioral interpretation. Simulated persona is
   an instrument-control variable, not a human label. No new MEPS training.

## What changes and what stops

GRAIN retains population reference, conditional author profiles and evidence
provenance. The path branch supplies intervention-sensitive interpretation and
potentially a compilable scoring function. A future method should tell a researcher
what evidence is useful under a stated condition, rather than require a new head
search for every scale. Choosing informative experiments remains necessary;
exhaustively filling all combinations does not.

Stop expanding the head/layer/trait/seed grid, optimizing the twelve-pair coordinate
correlation, or naming a head as a trait. Do not launch synthetic-to-human design
transport solely on an assumed interpretation of a model task null. No automatic
experiment launch follows this seminar document.
