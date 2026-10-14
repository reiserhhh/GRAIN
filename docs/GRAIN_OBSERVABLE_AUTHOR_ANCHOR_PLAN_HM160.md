# HM160: observable author anchors under an unknown latent law

2026-10-11. Freeze this plan before collecting its new source/application
records. No reader optimization, class-count fit, hidden-label fit or precision
search. Existing numerical public-source alphabet only; no private arrays,
raw text, API, new encoder forward, training, publication or old queue.

## Missing construction, and prior work not to repeat

HM159 compiles author functionals without recovering G, but fixes the behavior
kernel. HM142 learns a three-class kernel with imposed k and independence.
HM143 connects anonymous local classes. F15/R08 already implement PSR,
bisimulation and observable equivalence; those are not new discoveries here.
C3/C4 have real-author/private numerical evidence and reader transport. This
packet does not access those private arrays or re-run those precision protocols.

New capability: compile a reference-relative, condition-specific future
expression profile from qualified observed history/endpoint joint frequencies,
without recovering a latent kernel, a population G or any named trait. Meaning
is an explicit observed endpoint and collection regime, not an arbitrary latent
coordinate. This is a standard conditional-law construction newly connected to
the GRAIN vector/event and cold scoring interfaces, not a new probability theorem.

## Competing explanations

1. A unique numerical latent position is required before author comparison.
2. Latent descriptions can be nonunique while target-anchored author functions
   remain observable. A specified endpoint anchors a function, not an entire
   psychological theory or every possible behavioral quantity.

Second discriminator: a self-selected condition's observed response kernel
either determines or does not determine response under an imposed condition.
Keep the useful associational kernel rather than delete choice or refuse all
scoring. Do not silently promote it to an interventional kernel.

## Registered finite world

Alphabet: first four vectors in
`results/grain_math_response_v1/relative_choice_r1/dictionary.npz`, `menus`.
Address is 2*C + (C xor R), so vectors mark BOTH choice C and response R.
The simulator emits dictionary vectors exactly: this is not natural-text
recognition. History is one qualified event; endpoint is a DIFFERENT event of
the same simulated source author. No same-event carrier duplication.

Main population: p in {1/20,9/20,19/20}, mass {1/2,3/10,1/5}.
History C,R conditionally independent Bern(p). Endpoint opportunity A is
randomized Bern(1/2); endpoint response probability is p for A=0 and p^2 for A=1.
Constructor receives only observed table/history vectors, not p, G, component
labels, latent functions, reader answers or application outcomes. Independence
of the history indicators is a simulation property, NOT required by the table
constructor. Target is future expression in registered condition q, not Big5.

For a history h with positive support, randomized A identifies

    psi_q(h) = P(Y_endpoint=1 | H=h, A=q)
    reference_q = sum_h P(H=h) psi_q(h)
    delta_q(h) = psi_q(h) - reference_q.

Assignment design and same-population/occasion scope are supplied metadata,
not inferred from a low correlation. Natural-selection tables instead return
an ASSOCIATIONAL_CONDITION_PROFILE. No unspecified do-effect is reported.
One future response probability is not an exact individual p or stable trait.

## Exact new witnesses

- Latent reparameterization lambda=p^2, with all event kernels composed with
  the inverse, preserves every possible observed joint history/endpoint law.
  Compare E[p|C=1,R=0]-E[p] and E[lambda|C=1,R=0]-E[lambda]. No formula
  assertion about their signs is used as a test acceptance condition. Midrank
  coordinates are also checked, not supplied to or claimed recovered by scorer.
- Compute all four histories' two observable target functions and exact Brier
  improvement over the SAME fixed population reference: Var_H(psi_q(H)).
  This is a mathematical discriminator, not a hyperparameter objective.
- Self-selection alias: two equally weighted author types p={1/4,3/4}, A=type.
  Response rows (type x imposed A): W0 [[1/5,3/10],[7/10,4/5]],
  W1 [[1/5,9/10],[1/10,4/5]]. Actual selected responses are identical.
  Enumerate their full observed joint tables and randomized anchor tables.
  The latter can differ. Preserve both worlds, do not fit one to the other.
- Null endpoint: condition rates {3/10,3/5}, independent of author/history.
  Its conditional relative profile and exact history-based Brier gain should
  be zero; otherwise the implementation is wrong.
- Support failure: a missing history-condition cell returns an unobserved
  status, not zero/neutral or fabricated shrinkage. Reference calculation is
  unavailable for a condition missing any positive-history support.

## New records, bounded execution and cold scoring

Three fixed seeds 160,161,162, 12000 source authors and 4000 new application
authors per seed. One source history + one randomized endpoint. Each application
has one history and two separate endpoints in q=0 and q=1 for evaluation only.
No grid, seed choice, repeated fitting or rescue if an endpoint gain is negative.
Use the same random draws to replay the latent reparameterization; do not count
that coordinate readback as a second sample. Oracle p arrays are separate files
and excluded from both bundle construction and the cold CLI.

Save observed vectors/IDs, source bundles, evaluation-only arrays, all four
history profiles, every seed's endpoint scores, exact tables, hashes and report.
Cold CLI takes one source bundle, frozen alphabet and a history vector, with
no outcome/latent/teacher fields. Exact decoder rejects out-of-alphabet vectors.
No user-facing psychological trait naming, percentile or new latent axis.

Before collection: unit tests and Ruff. After collection: read-only replay all
counts/scores; independently enumerate exact joint laws and posterior quantities
using a separate direct-Fraction implementation. Same-agent alternate arithmetic,
not independent blind validation. Do not claim full goal completion.

## Primary literature actually read for this design

- He (2023), *Extended Asymptotic Identifiability of Nonparametric Item Response
  Models*, introduction, equations 1-2, Conditions 1-4 and Theorem 1:
  https://arxiv.org/html/2310.20165v1 . Reparameterization with a changing latent
  distribution is prior art; its long-assessment theorem is NOT applied here.
- Dawid (2020), *Decision-theoretic foundations for statistical causality*,
  pp.2-4 regime/stability discussion and pp.16-19 idle/ignorability definitions:
  https://arxiv.org/pdf/2004.12493 . A collection regime is part of the target.
- Dawid (1984), publisher summary of the prequential approach: predictions of
  observables versus parameter statements. Full PDF was retrieved but not
  text-extracted; do not claim its proofs have been read.
  https://academic.oup.com/jrsssa/article/147/2/278/7106293 .
- OpenAI/math pinned adc7f124, metric/unitary-connection one-boundary-patch
  manuscript, introduction lines 1-170 re-read. Only observation invariants
  versus coordinate/gauge choices inspires this design. No PDE theorem imported.

Stop this packet after its finite construction/discriminators. The next mainline
question is how source-qualified natural evidence can acquire and transport an
endpoint anchor while retaining choice, conditional response and state. Do not
turn latent naming, probability RMSE or these seeds into a new precision branch.
