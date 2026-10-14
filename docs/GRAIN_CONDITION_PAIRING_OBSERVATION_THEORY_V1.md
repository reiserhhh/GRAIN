# Condition-Pairing Author Observation

## The object, not an improved predictor

HM186 supplies a natural numerical observation of **which expression occurs
under which recorded condition**, retaining the person's position and choices.
The canonical result is `condition_pairing_response_r1`. It changes the
observation family, not a regression's accuracy or a numerical tolerance.
The empirical decision and all numbers are in the parent HM186 report.

HM81 already retained joint moments on constructed balanced histories. HM171
calibrated actual source target laws; HM185 separated repeated authors from
cross-author target structure. This extension measures the natural
condition-expression pairing on the same source events, at unchanged vector
bags and condition counts. Covariance itself is not our invention.

## Finite law and group action

Author u supplies addressed observations (h_ut,c_ut), with h in R^1024 and a
recorded binary condition. The empirical law includes all events:

    P_u = (1/n_u) sum_t delta_(h_ut,c_ut)
    mu_u = E_u[h]; p_u = E_u[c]; m_u = E_u[c h]
    rho_u = m_u - p_u mu_u
    o_u = p_u(1-p_u).

Let S_n act on the condition addresses, holding the entire vector bag fixed.
The Reynolds average of the mixed observation is:

    (1/n!) sum_(pi in S_n) m(h,c_pi) = p mu.

Thus rho is the component sensitive to the **pairing action**. It is not a
definition of all personal information, nor proof that invariant information
is psychological noise. The complete tuple (p,mu,m,rho,o) remains available.
Simultaneous relabeling of vector and condition addresses leaves it unchanged.

For supported observations, elementary conditional expectation gives:

    rho_u = o_u delta_u
    delta_u = E_u[h|c=1] - E_u[h|c=0].

The normalized delta is also the observed vector regression slope on binary c
with an intercept. It is not the person's causal response to an externally
assigned environment. At p=0 or1, delta is unobserved, not0. Arithmetic rho=0
then records absence of mixed-condition opportunity, not absence of a trait.
This resolves a measurement-object error without deleting any text features.

## Target observation on source populations

For fixed source author fold b, disjoint text view v and binary inherited
target q, the two covectors are:

    d_(b,v,q,level) = E_source[mu_uv|Y_q=1,b] - E_source[mu_uv|Y_q=0,b]
    d_(b,v,q,pairing) = E_source[rho_uv|Y_q=1,b] - E_source[rho_uv|Y_q=0,b].

Every author has equal weight within each group. Other four labels are not
matched, so these are marginal target-linked contrasts, not unique effects.
The reference differs from HM185's uniform32-profile atlas. It must not be
described as an accuracy improvement over that bank.

With the unchanged map A:R^15->R^1024 and its QR basis Q, each d is read in
full, QQ^T d and complementary coordinates. Both components remain. A frozen
pooled source bank scores the following observations without source labels or
LLM answers at application time:

    s_(u,q,k,v,space,O) = <d_(q,k,v,space), O_u>,
    O in {mu,m,rho,delta if observed}.

Every source field kind reads every observable; the cold output is not reduced
to whichever coordinate later looks favorable. These are source-referenced
linear readings, not probabilities or validated trait scales. The exact
mixed-account identity survives every field:

    s(m) = p s(mu) + s(rho).

## Identifiability and evidence boundaries

Equal vector means, identical vector bags and equal condition counts can coexist
with opposite rho. Therefore no more precise function of those marginals alone
can reconstruct the conditional association in general. This is a standard
finite coupling fact; the toy-world test supplies an explicit witness.

The empirical primary instead asks whether source target-associated rho fields
repeat in disjoint author folds, averaging all four text-view pairings and all
five targets. Whole-label-profile exchange tests source target assignment;
within-author/view condition relinking tests the actual pairing at fixed
opportunity and vectors. Relinking is a relational null, not a valid new natural
text or causal situation intervention.

The registered all-target primary did not pass either reference. N retained a
positive descriptive pairing signal, while C and O had negative cross-author
products. Keep all of them. Within-author recurrence is not a substitute for
cross-author target structure. Neither the null result nor the N clue establishes
a universal personality path, and there is no sign/fold/weight rescue.

The observed condition is only an inherited TOP20-venue bit. It is not a
complete condition, situation, opportunity, state or personal-choice model.
The historical neural encoding and frame selection stay part of provenance.

## Mathematical sources and what was actually read

- Song, Huang, Smola, Fukumizu (ICML2009), pp1-2 Eq3-5, cross-covariance and
  conditional embeddings: https://icml.cc/2009/papers/538.pdf. These pages were
  read. We implement the finite binary identity, not their inverse-operator,
  RKHS membership, convergence or characteristic-kernel guarantees.
- Olver, *Classical Invariant Theory*, p74 Eq4.8, finite-group averaging:
  https://www.math.toronto.edu/~ila/ClassicalInvariantTheory.pdf. The primary
  indexed page excerpt was inspected; direct PDF opening failed. No full-book
  reading is claimed.
- Maithani, *Splitting the difference: Computations of the Reynolds operator in
  classical invariant theory*, BLMS2026, https://arxiv.org/abs/2412.18841.
  Abstract checked after the science freeze; its algorithms for classical
  algebraic groups are not implemented or claimed to justify our empirical result.
- OpenAI/math pinned adc7f124, Signed-tensor-densities, 03b-isotypic.tex170-265
  reread: prescribed action/support and retained components are design inspiration.
  No density bound, compact-type extraction theorem or novel psychology theorem
  has been transferred.

The original18/100/44 scope remains open. The next mechanism question is how a
qualified target observation relates to the same event's vector/query action,
not whether these arithmetic identities can be made more precise.
