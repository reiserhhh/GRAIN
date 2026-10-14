# HM10B1: Native Response Content and Link Geometry

2026-10-08. Codex executes this finite packet under the active parent goal.
This is a new observer experiment on frozen HM10A texts, not new text replication.
HM10 author parameters and the original 18/100/44 scope remain open.

## Scientific Decision

Do actual content queries show a joint effect beyond the scoring link, and can
source-supported joint natural coordinates act on unseen lexical pair families?
Report coexposure in HM10A is not enough: its declared min/max law is nonlinear.

Reconcile all 25 source-referenced narrow functions without reading private
raw posts or questionnaire outcomes. The source coefficient arrays and exact
query-token coordinates are reused. Observe three fixed queries, selected on
the materials rather than scores: adapted_na8 (anxiety-relevant),
adapted_dejected5 (adjacent but not fully represented), pre_trust_positive
(direct AI trust evidence absent). An absent target is unobserved; a native
prior/default score must not become a personal zero or a measured trait.

## Fixed Observation Design

Reuse all 1,024 document records / 768 unique HM10A texts. Preserve the same
16 lexical pairing families (source/validation/application 8/4/4), four state
frames, self/other trigger actors and report incidence. The writer remains
the state experiencer. There are no human author IDs or personality labels.

Run 2,304 unique text-query forwards with the pinned local Qwen2.5-1.5B,
float32 MPS, no generation, gradients, training, network or API. Each text's
actual question-token P and final Q are observed at block16. Frozen affine
functions remain unchanged. Preserve exact target-weighted token projections
and Q, not the entire P; preserve the normalized final state and output-head
rows so logits can be independently reconstructed. Record candidate mass.

## Mathematical Objects

For conditional digit probabilities p=softmax(l), set theta=l-mean(l).
This removes only the exact additive-logit gauge. It does not remove text
context, population means, conditions, state or author information.

For cells ordered 00,10,01,11, theta*=theta10+theta01-theta00, and

    d=theta11-theta*
    Delta S=[S(theta*)-S10-S01+S00]+[S11-S(theta*)].

The first term is link curvature even when d=0. The second is the action of
actual logit interaction. Signed terms may cancel; they are not variance
percentages, unique causal allocations, or psychological information shares.

The second term also equals the line integral of
grad S=p*(answer_values-S) along theta*+t*d. The categorical Fisher matrix is
diag(p)-p*p', positive semidefinite with a gauge null. KL is checked against
the log-partition Bregman form. The local Fisher energy is not exact KL.

## Source-Only Completion and Controls

All six declared arms remain: scalar additive, additive logits then link,
source global logit correction, source conditional logit correction,
wrong-condition-key logit correction, source conditional scalar correction.
Conditional keys are frame/trigger actor/incidence. Average only source joint
contrasts; never fit on validation/application. Do not clip scalar completions.
The cold 11 cell is withheld from completion but used by the evaluator.
This requires cold 00/10/01 observations; it is not single-text prediction.

Source and application lexical families are independently bootstrapped 2,000
times for the conditional-logit arm versus the link-only baseline. Independent
application families number four, not 64 cells or 1,024 documents. No result
selects a prompt, threshold, rank or query. Frozen old affine response fidelity
is described separately, not claimed as no-prefix independence.

## Evidence and Next Action

Require actual calls and complete native units, exact raw-text null checks,
logit/output-head replay, score-link and integral closure, source-only cold
completion and a same-agent alternate-formula verifier. Preserve all failures.
If relation actions are target-specific, retain separate operators. If generic
or no-evidence queries also move, distinguish default inference from observed
evidence before any author parameter. No arbitrary simulated coefficient
recovery substitutes for actual author response content.

## Sources and Reading Scope

[Wainwright and Jordan (2008)](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf),
Section 3.5.1 and Propositions 3.1-3.2, provides exponential-family cumulant
derivatives and nonminimal parameter equivalence. This packet implements the
finite categorical specialization, not their full variational algorithms.
[Aitchison (1982)](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x) provides
compositional-analysis background; only its publisher record/abstract was
checked in this packet, not the full article. The exact clr identity here is
derived and tested directly.

OpenAI/math's pinned entropy-localization proof was reread in full. It concerns
compact dynamical systems with a specified attracting set. Those hypotheses
are not established for text, so its entropy theorem is not imported. Its
useful methodological reminder is to keep an observation/compression property
separate from the underlying system's property. HM10A's finite Boolean
contrast machinery is retained, not a claim of new Fourier mathematics.

## Run

From the independent GRAIN repository:

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
"$PY" scripts/run_grain_math_native_joint.py prepare
"$PY" scripts/run_grain_math_native_joint.py collect --seconds 1800
"$PY" scripts/run_grain_math_native_joint.py evaluate
"$PY" -m pytest -q tests/test_grain_math_native_joint.py
```

No private MEPS reads, API, new neural training, Claude/automation restart,
commit, push, tag, lockbox or old queue. Output is a fresh ignored local packet.
