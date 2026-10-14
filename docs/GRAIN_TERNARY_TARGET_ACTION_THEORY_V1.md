# Finite marginal-preserving target actions on three evidence roles

2026-10-11, HM180/HM180B. A source-qualified construction, not a new theorem
about high-order information or a completed psychological measurement system.

## Object and addition

The target observation is a full five-bit measured Big5 profile G, not a
latent author class. A/B/C are observations drawn from three disjoint four-
event roles of one source author. Each event cell retains the coarse venue
bit and two frozen anonymous GRAIN coordinate-cut bits. Both text views use
the same4+4+4 rule; their slot order is not a time axis. Define

    F(g,a,b,c) = (1/U) sum_{u:Y_u=g} P_uA(a) P_uB(b) P_uC(c).

This is the law of uniform source-author sampling and one uniformly drawn
event per role. The product is a declared sampling procedure conditional on
the actual author, not independence of natural human behavior given Big5.
Repeated triples are dependent observations of the386 authors, not49408 people.

Unlike fitting a larger predictor, the new operation keeps ALL GAB/GAC/GBC/ABC
marginals fixed. Thus the five-target single-/two-event meanings, full profile
prior, profile-conditioned venue choices and pooled three-event content law
stay unchanged. Only the assignment of three-event configurations to measured
profiles can change. This answers a specific construction question beyond
HM177's pair assignment; it is not new discovery of the existence of synergy.

## Failed global reference and different finite operation

The original HM180 attempted a maximum-entropy reference under these margins.
It stopped at its fixed10000-sweep limit, error2.2785961626423448e-8. No reference
bundle was produced. That scientific failure and its original seals remain.
HM180B is a separate post-failure plan: an explicit feasible action, NOT a
more accurate solution of the original optimization problem.

Within each fixed venue triple, take all profile edges differing in exactly
one target bit q and all2x2x2 state cubes. The16-corner sign is

    chi = (-1)^(profile_edge_bit + state_A_bit + state_B_bit + state_C_bit).

Only cubes whose16 original F cells are positive are used. For each cube j:

    ell_j = sum_i chi_ji log(F_i)
    m_j = min_i F_i
    tau_j = m_j ell_j / (32 + 2 abs(ell_j))
    D_q = sum_{j:target(j)=q} tau_j chi_j / 135
    H = F - sum_q D_q.

All admissible cubes are used ONCE. There is no step-size tuning, entropy
iteration, target selection, source-view choice or outcome fitting. The135
bound is5 profile-bit neighbors x3 state partners per role cubed. It does not
estimate a regularization parameter. No input zeros are filled or discarded.

## Finite proof and target-specific isolation

Summing an alternating cube over any of its four axes is zero, so each D_q
preserves all four required marginals. For a different target p, its bit is
identical at the profile edge's two endpoints; hence pointwise

    sum_{g:g_p=1} D_q(g,a,b,c) = 0, p != q.

Consequently every q component changes only that target's conditional marginal
readout. Joint profile dependence and signed cross-target geometry remain;
this algebra does not imply independent or pure human psychological traits.

Since abs(tau_j)<m_j/2 and cell incidence<=135, abs(F_i-H_i)<=F_i/2. H is a
normalized nonnegative law, H>=F/2 on support, with exactly the same zeros.
For S(F)=-sum F log F, Taylor's curvature bound and weighted Cauchy give

    S(H)-S(F) >= sum_j tau_j ell_j /135 - sum_i (F_i-H_i)^2/F_i
               >= sum_j tau_j ell_j /(2*135) >=0.

The second step uses incidence135 and16 tau_j^2/m_j <= tau_j ell_j/2.
This is a direct finite algebra/calculus bound, not a new general theorem or
a claim that the listed cubes form a complete Markov basis.

H is not maximum entropy. KL(F||H) is not the entropy gain, the full irreducible
high-order information, or a fraction of psychology. A small fixed operation
does not upper-bound how much high-order structure the source may contain.

## Source target function and cold author readout

At supported abc, all five target functions derive from the SAME joint law:

    psi_M,q(abc) = sum_{g:g_q=1} M(g,abc) / sum_g M(g,abc), M in {F,H}.
    psi_F,q - psi_H,q = sum_{g:g_q=1} D_q(g,abc) / F_ABC(abc).

The denominator is unchanged. This supplies five separate qualified actions,
not five independently fitted and potentially inconsistent classifiers.
For a current text subset X with original role addresses, the cold function is

    theta_M,q(X) = (1/64) sum_{i in X_A,j in X_B,k in X_C} psi_M,q(c_i,c_j,c_k).

It also returns the full32-profile law, opportunity mass and conditional mean.
No triplet opportunity gives mass0 but conditional mean unobserved, not neutral
personality. Any unsupported present triple marks that readout unobserved;
there is no source-prior fill. The fixed denominator retains what the subset
offers. The context contrast over R+S/R+C/S/C is independently enumerated as
triples contacting the designated disjoint roles, not a neural intervention.
No64-fold repeated-label product or whole-history author posterior is used.

The existing strict HM180 cold schema is deliberately reused. Its version and
code_hash describe the shared readout-law validator. Actual HM180B constructor
identity/hash and ipf_used=false are explicit in every solver_receipt and the
frozen manifest; the name pair_constrained does NOT assert a maximum-entropy fit.

## Actual findings, reuse and restrictions

SOURCE view0/view1 have2114/1889 supported cubes; all five target fields change
while lower constraints agree to1.39e-17 and cross-target isolation to8.60e-23.
The field RMS ranges1.27e-5--4.71e-5 in probability units. This is empirical
support for a nonzero declared action, not an independently confirmed trait
signal; target names entered through reused source labels.

Fixed author-disjoint two-fold checks have13/20 negative and7/20 positive
reference-minus-full Brier differences. The10 pooled view/target comparisons
have7 negative and3 positive, maximum absolute4.79e-7. Full empirical triples
are not promoted as more useful than this reference. Small differences partly
reflect the conservative one-step bound and do not reject every high-order model.
Only337/333 of386 authors have full64-triple support in views0/1; all remaining
49/53 authors and partial-support losses stay recorded. No complete-support
subcohort is represented as the full population. Historical representation/
label binning were reused, so this is conditional source-law crossfit, not
untouched psychological validation or full-pipeline cross-validation.

The24-author unlabeled panel has480 cold records,4720/4800 finite target values,
84 records without triplet opportunity and20/960 unobserved contrast values.
The existing encoder remains neural. Cold arithmetic is neural-free, not an
end-to-end LLM-free natural-text personality system. Probability-table moves
are not automatically realizable new texts or attention/KV interventions.

## Prior art, actual reading and next question

- Schneidman/Still/Berry/Bialek2003, abstract and initial connected-information
  construction: https://arxiv.org/html/physics/0307072 . High-order dependence
  beyond lower marginals is established mathematics, not this project's invention.
- Sugiyama/Nakahara/Tsuda2016, SectionsII-A/II-B:
  https://arxiv.org/html/1601.05533 . Mixed coordinates motivate preserving
  selected meanings, but their strictly-positive manifold result is not imported
  unchanged onto our sparse source support.
- Diaconis/Sturmfels1998 conditional-fiber ancestry: author bibliography
  https://math.berkeley.edu/~bernd/articles.html and Stanford1993 report metadata
  https://statistics.stanford.edu/technical-reports/algebraic-algorithms-sampling-conditional-distributions .
  Journal fulltext/Stanford repository were not accessible in this reading;
  we use explicitly checked elementary margin-zero cubes, not an unexamined
  complete-basis/connectivity theorem.
- Pinned OpenAI/math adc7f124 three-state reconstruction foundations175-232
  was reread: common input versus product marginal is a sampling-design lesson.
  No tree reconstruction threshold, new theorem or complete-repository reading
  is claimed for author measurement.

STOP IPF/tolerance/step/grid rescue and triple-accuracy promotion. The next
missing link is whether a source-supported target action on these finite
observation cells has a qualified lift to the SAME actual vector evidence,
without inventing vector corners or fitting LLM responses. Inventory HM170's
original event/coordinate interface, HM171 target law, HM144 configuration
operator and HM173/HM178 native12-event addresses before a new finite plan.
Distinguish target-isolation by coarse construction from natural vector
meaning/transport. Keep choice, conditions, state, joint profile and all negatives.
Original18 protocols/100cards/44requirements remain OPEN; GoalACTIVE/nojob.
