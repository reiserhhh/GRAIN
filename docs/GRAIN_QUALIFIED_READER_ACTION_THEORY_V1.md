# Person-qualified reader action and contextual composition

2026-10-11. HM155 working theory. The prospective experiment is sealed
separately; this explanatory document is not an outcome-selection rule.

## 1. Two observation laws, not one reversed arrow

The writer experiment observes

    L_W(theta; x, c) = log p_model(x | writer-persona theta, task c).

The reader experiment observes

    L_R(theta; x, u, q, c)
      = log p_model(Yes | x, queried-person u, target-query theta,q,c)
        - log p_model(No | x, queried-person u, target-query theta,q,c).

Neither is a human personality posterior by definition. In particular, the
author of a report, the actor in its events, and the person being evaluated
need not be the same. Bayes inversion of a writer likelihood would require
an explicit data-generating likelihood and reference population, including
these roles; it is not obtained by swapping the names of the arrows.
The endpoint covector W_Yes-W_No gives an exact native reader observation
from the final normalized state. This single shared answer direction does
not make the upstream target-dependent evidence operations identical.

## 2. Qualification as an actual interaction, not a label

e and n select which person performed each opposite behavior pair; p selects
the person asked about. At fixed wording, opportunity and block order,

    F(e,n,p) = beta0 + betaE*e + betaN*n + betaEN*e*n
               + betaP*p + betaEP*e*p + betaNP*n*p + betaENP*e*n*p.

All eight terms are retained. betaEP/betaNP distinguish a response following
the queried actor's evidence from word counts or a generic Person A bias.
They are finite designed contrasts, not latent traits or population variance
components. A nominally target-specific prompt response with zero qualified
contrasts does not establish person-specific inference. Conversely, a
nonzero qualified contrast is useful mechanism evidence without needing to
maximize predictive accuracy.

The same expansion can be applied to h, W*h, log normalizers and log
probabilities. Linearity makes their accounting consistent; averaging a
softmax probability and then transforming it is a different operation.
Three query descriptions expose common qualification and conditional
departures. Do not discard the latter to manufacture a universal axis.

## 3. Local evidence transport versus distributed context

An all-block K/V interchange on registered historical rows changes only
inquiry-suffix access; native historical computation remains fixed. Replace
social rows with a naturally observed e-flipped donor and worry rows with a
different, n-flipped donor. The two disjoint address replacements commute
BY CONSTRUCTION. Their reader outputs need not add, and their union need
not reproduce the fully flipped natural historical representation.

    DeltaWhole = DeltaE + DeltaN + I_local + R_nonlocal
    I_local = F_localBoth-F_localE-F_localN+F_native
    R_nonlocal = F_wholeBoth-F_localBoth.

This exact decomposition is arithmetic, not itself a discovery. Empirical
magnitudes, signs, target specificity and expression/opportunity/order
dependence decide whether the local compositional hypothesis works here.
R_nonlocal is a remainder of this specified intervention, not psychological
noise. It can contain causal descendants in later clauses, record framing,
distributed relational information, or an intervention mismatch. A reverse
order comparison helps locate, but does not by itself identify, these sources.

We do NOT call this operation a newly discovered group representation.
Transforming the text, naturally re-encoding it, and locally replacing a
fixed set of carriers are three distinct operations. Their equality is
tested, not assumed from the symmetry of the material design.

## 4. Connection to the GRAIN measurement spine

A report about another person can carry two different kinds of evidence:

1. An event about that actor, eligible for the actor's response profile.
2. The reporting author's choice of topic, framing or expression, eligible
   for the author's selection/style profile under an explicit observation law.

Thus actor qualification is not a mandate to delete all other-person text.
Maintain typed channels rather than collapsing author, actor and queried
person into one identity. GRAIN's choice/condition response/state/history
and fixed source reference population remain available. New reader actions
would supply a target-specific observation function, not replace the
psychometric aggregation and opportunity model.

An intended construction interface is

    S_tau(u; R) = integral ell_tau(event, condition, role=u; R)
                         dP_u(event,condition)

with separate choice and report channels, not a free identification of an
LLM log odds with personality. Shared output vectors can induce correlations
between instruments; psychological covariance must be examined separately.
The current experiment knows clause addresses from its material construction.
It does not claim to solve natural-text evidence segmentation or cold human
calibration. These are distinct existing requirements, not relocated to
human-pending by this packet.

## 5. Prior art, retained contribution and boundary

Feng/Steinhardt2023 https://arxiv.org/html/2310.17191v2 already distinguishes
construction/read interventions and investigates binding vectors.
Feng/Russell/Steinhardt2024 https://arxiv.org/html/2406.19501v1 uses content
and binding with causal interchanges. The finite character transform,
weighted Hilbert projections and inclusion/exclusion identities are mature.
HM10B2/HM114/HM124-146 are this project's existing related experiments.
The new test is the evidence-qualified cross-target/cross-wording reader
interface and the actual local-union/whole-history discrepancy on new,
fixed-bag records, not another ownership classifier or a new math theorem.

OpenAI/math adc7f124 Fourier proof and random-frame permutation manuscript
motivate preserving joint laws instead of relying on marginal similarities.
Their specialized hypergraph/shuffle results are not text-measurement proofs.

## 6. Empirical findings

HM155 collected all164 batches/1968 inquiry sequences in334.650 seconds.
The original analysis wrote complete accounts and CSVs but failed to serialize
a NumPy int64 count. Its partial summary.json is retained byte-for-byte.
serialization_recovery_r1/summary.json is the explicit recovered entry point;
no forward, scientific code edit or fit was used for recovery. Alternate
formulas checked2,961,156 numeric entries (algebra error0, native head error
6.64e-5, six predeclared Gram spots <=8.80e-7);448 additional direct-endpoint
aggregate checks differed by at most2.22e-16. These are same-agent checks.

Qualification is positive in46/48 predeclared target/family/context cells:
E24/24, N22/24. The two negative N cells both use the named query with the
social block first and freely chosen behavior, one in each wording bank.
This count is NOT classification accuracy. Own-qualified mean coefficients
are1.381346 E and.816304 N log-odds units. N's named mean.194610 differs from
described1.522781; retain the query dependence rather than reweighting it away.
All16 secondary target/context field supports are positive, but actor-qualified
actions, not a shared Yes/No output direction, are the primary evidence.

Local transport has a strong order pattern in both wording banks and both
opportunity conditions. When the edited block is the later block, signed
projection onto its natural flip is.994023..1.014814 across both queried
targets. When it is first, the range is.191268...826594. Local union leaves
a relative discrepancy.229041...575760 from whole-history action; local
interaction RMS is.110235...445840 times whole-action RMS. These are vector
action comparisons, not proportions of personality information.

Post-outcome inspection of all64 single-flip natural K/V pairs finds preceding
carriers unchanged exactly, a changed later other block in32/32 applicable
pairs, and a changed record-ending representation in64/64 pairs. This is
consistent with ordinary autoregressive contextual propagation. It locates
candidate descendants but does not determine which descendants causally
mediate the omitted action. Generic propagation is prior art, not a new
personality mechanism by itself. The next discriminator is the target-specific
role of these descendants, not improving the local patch's numerical fidelity.

Opportunity modulates the qualified E effect: free minus required is positive
in all12 matched wording-bank/order/query-family comparisons, mean+.542944.
Cross-target mean coefficients are also nonzero (worry-to-E-.551582;
social-to-N+.116274). Hence separate targets cannot simply be treated as
independent evidence channels. These are finite controlled observations,
not human trait/state effect sizes or a population generalization.

## 7. What the theory gains, and what is already known

The empirical update is a role-qualified, query- and opportunity-dependent
observation interface with a measured local/nonlocal composition discrepancy.
It makes the next operation identifiable enough to test: compare frozen
actor-binding carriers, later unchanged-text carriers, and record-ending
carriers, while keeping both targets and all query families. A narrow patch
can miss real response because later states encode its causal descendants;
that mismatch is not automatically psychological noise or inaccurate sensing.

Geiger et al., Causal Abstraction (2025 revision), Sections2.2,2.3,2.5,
https://arxiv.org/html/2301.04709v4, supplies a mature formal criterion:
a proposed high-level explanation should preserve specified intervention
results, not merely fit ordinary outputs. This reference was reread AFTER
HM155 design seal; it is retrospective alignment, not prospective provenance.
Our disjoint address operations commute by construction; no newly discovered
intervention algebra or exact causal abstraction is claimed here.

For GRAIN construction, keep the actor-event response channel separate from
the author's selection/reporting channel and the source population reference.
The former may depend on distributed context rather than only the token span
that names its event. A future source-defined reader operator must preserve
these counterfactual responses on new expressions; a flexible fit with higher
accuracy alone does not supply that guarantee. Current normalized states and
K/V still require the frozen model. This packet does not complete LLM-free
raw-text measurement or the original18/100/44 scope.
