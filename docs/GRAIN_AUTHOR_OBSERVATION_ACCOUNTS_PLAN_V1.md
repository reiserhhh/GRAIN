# HM69: Author Choice, Conditional Expression and Observation-Law Accounts

2026-10-09. Freeze before numerical execution. No accuracy optimization.

## Decision and Historical Crosswalk

M03 already established free-choice / fixed-condition accounts and source-learned
design. HM65 separated common reader Tq, author evidence Pu and human response Bu.
HM68 found generic and ownership-dependent cross-target composition. This packet
joins those existing pieces; it does not rediscover two-phase design or Shapley.

Question: can differences of actual vector-reader observations be accounted for
as context selection, conditional expression and material/reading-law change,
without discarding choice or confusing a reader change with author change?
Also construct differing author distributions with identical six-target scores:
does retaining choice/conditional profiles expose what those scores compress?

## Objects and Formula

Use all30 ordered distinct HM68 behavior pairs, both lexical families and all
six captured stages. F(x,y,g) is a frozen observation law on eight designed
accounts. C=(y,g) has four cells. Write the positive profile

    P_u(x,C) = pi_u(C)*(1+x*mu_u(C))/2,
    Theta(pi,mu,F) = sum_C pi(C)*sum_x (1+x*mu(C))/2 * F(x,C).

pi is observed condition/ownership composition, not preference corrected for
unknown opportunity supply. mu is conditional expression of the focal behavior,
not a causal human response Bu. Theta is an observation functional, not a
psychological ground truth. Both pi and mu are retained; no centering deletion.

Three factors may change between endpoints: pi, mu, F. For each subset S of
the three factors, form the complete counterfactual hybrid G(S) using endpoint1
for factors in S and endpoint0 otherwise. All probabilities remain positive.
Save all8 G values, their anchored Boolean Mobius differences and all6 replacement
orders. Average each factor's six sequential marginal contributions. Equivalently

    phi_j = sum_{S containing j} Mobius_G(S)/|S|,
    sum_j phi_j = G(all)-G(empty).

This is established Shapley decomposition, used coordinatewise for vectors.
It is a symmetric accounting convention, not unique causal mediation. Individual
order allocations and all interaction terms remain visible; exact sum alone
does not identify a person's psychological mechanism.

## Fixed Complete Worlds, Not a Precision Grid

For C=(y,g) in the HM68 order, define uniform pi0, and
pi1 proportional to exp(.6*y-.2*g+.15*y*g). Set
mu0=.2*y-.1*g+.15*y*g; mu1=-.1*y+.2*g-.35*y*g.
Use these complete, interior archetypes without sampling, estimation or labels.

Eight cases for every ordered pair:

1. Choice-only, with each family F fixed, mu0 fixed (two cases).
2. Expression-only, each F fixed, uniform pi and mu=+/-.5*y*g (two cases).
3. Law-only, pi0/mu0 fixed while F0 changes to F1 (one case).
4. Joint pi/mu/F transition with the fixed archetypes (one case).
5. Six-score-equivalent profiles in each family (two cases).

For case5, compute the nullspace of [1,F_score]^T by SVD with the fixed
machine-precision rank rule. Use its first normalized null vector v, with largest
absolute entry positive, and Pplus/minus=1/8+/-.05*v/maxabs(v). These profiles
remain positive and normalized. SVD is algebraic construction, not fitting a
personality predictor. The six scalar scores coincide by design; check whether
choice/conditional-expression profiles and other observation stages differ.
No claim that a six-score quotient must distinguish every behavior distribution.

Both families are existing comparable coded alternative banks, not identical
wordings/facts or a new blinded replication. F0->F1 includes changes of material
semantics as well as reading; it is not pure instrument calibration or a real
person's longitudinal change. No source-to-held prediction/selection is performed.

## Scope and Acceptance

30 pairs x8 cases x6 stages =1440 complete numerical account rows. Retain
actual profiles, all games, all components/interactions and order ranges.
Pure cases must assign0 to unchanged factors; mixed cases must sum exactly.
All six-score null profiles must preserve score and probability units; verify
the distinct retained pi/mu profiles instead of treating a zero score gap as no
author information. Body and logit changes are different units, not mediation.

Numerical verifier independently reconstructs each hybrid by direct nested sums,
then all permutation marginals and subset finite differences, without production
account helpers. Check source hashes, positive probability laws, direct expected
scores, exact decomposition and same-score witnesses. Tests use new synthetic
tables as well as actual saved inputs. All summaries must be recomputed from
full tables, not literal PASS checks. Preserve negative/zero results.

CPU collector bound60 seconds,0 new forwards/training/fits/API/raw/labels.
Existing HM68 numeric arrays are the observation source. No Claude, automation,
Git, publication, lockbox or old queues. Stop after this battery. Original
18/100/44 scope and unseen-text/psychological/Bu/dynamic questions remain open.

## Reading and Applicability

Shorrocks, Decomposition Procedures for Distributional Analysis, 1999 draft,
section2 (PDF pages5--8, zero-based pages4--7): replacement-order accounting
and symmetric averaging. Only that section was read, not the entire draft.
https://www.komkon.org/~tacik/science/shapley.pdf
The method is prior art, not a newly discovered causal attribution theorem.
Kitagawa1955 metadata was located, not its paywalled full proof:
https://doi.org/10.1080/01621459.1955.10501299

OpenAI/math pinned Conditional-coordinate-sweeps-and-analytic-transfer,
introduction and section03-paths, were actually read. They illustrate why
conditioning changes local laws and why uniform marginals need not determine
joint paths. Their random-permutation independence, without-replacement costs,
representation/Schatten contraction and analytic-transfer theorem do not apply
to this reader or human authors. Here the probability accounting is explicit;
no shuffle contraction, entropy-to-zero or new psychometric theorem is claimed.

Canonical planned: results/grain_math_response_v1/author_observation_accounts_r1.
