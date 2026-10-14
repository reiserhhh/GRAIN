# Evidence units and author conditional profiles

Executed 2026-10-07. Canonical finite results:
`results/grain_math_response_v1/evidence_r2/`. Preserve `evidence_r1/` as the
initial implementation, including its qualification-changing revision bug.

## Construction recipe

1. Declare the semantic target, eligible action/report types, author slots,
   event identities, context endpoints and reference distribution. A high vector
   score does not establish any of these fields.
2. Select natural candidate spans without using predicted scores. Preserve exact
   source offsets and full originals locally. Candidates are not verified labels.
3. Use a frozen vector readout to test controlled content/ownership/modal changes.
   Separate natural observations, extracted spans and synthetic reframings.
4. Include non-target positive/negative statements. Generic negation response
   cannot alone establish target specificity.
5. Resolve claim revisions, aliases and withdrawals BEFORE routing active claims
   to the selected measurement channel. An ineligible correction must invalidate
   its previously eligible parent. Report/event/author identities are distinct.
6. Aggregate independent events within conditions. Retain observed condition
   choice separately from the fixed reference average. Missing coverage is null,
   not zero or a renormalized reference distribution.
7. Learn working moments from source observations, without application truths.
   Distinguish sampling dependence, target units and observation-law transport.
8. Attach psychological meaning later using independent behavioral/scale evidence;
   this package does not label a simulated expression parameter a human trait.

## Mathematics

Let A_u be the current versioned claim set and Q_u its eligible subset. For
independent declared events e and context c, first combine reports of the same
event, then define m_u(c) as the event-balanced mean. With a balanced reference:

\[
a_u^\nu=(m_u(-1)+m_u(+1))/2,\qquad
b_u=(m_u(+1)-m_u(-1))/2.
\]

Choice pi_u(c) remains a separate observation, not a nuisance automatically
subtracted from text. Undefined endpoints leave the complete profile unobserved.

The existing source-learned working model uses level, response, block, event and
report covariance components. With X=[1,c], prior mean mu, covariance V and
report covariance R, its linear conditional moment is:

\[
C=(V^{-1}+X^TR^{-1}X)^{-1},\qquad
\hat\theta=C(V^{-1}\mu+X^TR^{-1}y).
\]

The audit computes this precision form separately from the production covariance
form. These are working moments, not a claimed exact Gaussian text likelihood.

For the stipulated binary-event simulation, controlled-cell means give
E_D[T|a,b,c]=alpha_D,c+beta_D,c(a+bc). The resulting profile map is
theta_readout=d_D+R_D theta_event. Population/material changes can shift d/R;
more events do not automatically repair this deterministic unit mismatch.

## Results and current design decision

Thirty natural anchors preserve strong assertion/denial responses, but fresh
non-target controls also show strong response. Multi-subject contamination is
principally mediated by the compiled program's global ownership coordinate.
Do not promote whole-comment joint scores directly to a natural personality scale.

The usable next construction object is an event-local, target-qualified unit.
Local event binding and global contextual/choice information coexist in separate
ledgers. The new event/conditional interface is implemented and tested; a general
natural target-relevance/role observation law is the next research packet.

## Commands

Run from the implementation repository with the existing offline environment:

```sh
python scripts/run_grain_math_evidence.py all --device mps
python scripts/analyze_grain_math_evidence.py --device mps
python scripts/verify_grain_math_evidence.py
python -m pytest -q tests/test_grain_math_evidence.py tests/test_grain_math_evidence_diagnostics.py
```

Use `--device cpu` if needed. New output roots must also be passed consistently
to the analysis and verifier. Existing frozen source bundles and licensed local
pair cache are required. No model installation, generation, paid API, remote
training, private MEPS reading or publication is performed.

Raw text, identifiers and per-anchor scores stay inside ignored `results/`.
The complete Chinese report is in the parent repository:
`reports/openai_math_review_20261007/execution/NATURAL_EVIDENCE_AND_EVENT_REPORT_ZH.md`.
