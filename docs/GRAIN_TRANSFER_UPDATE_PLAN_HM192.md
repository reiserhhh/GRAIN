# HM192: common target-update action versus a recurrent author catalogue

2026-10-12 JST. Finite plan before any new result. Original18/100/44 OPEN.

## Scientific decision

HM176 already measured cross-window author relations. HM177 separated their
target assignment; HM185 found that a source profile-mean joint atlas did not
transport across author banks. HM191 showed that a current target reading is
not a sufficient update state. None of those tests establishes that a
target-valued pair relation is the SAME update action across different people
and disjoint texts. This packet constructs and tests that object directly.

Not another within-author identification test, rank search, PSR counterexample,
prediction benchmark or attempt to fit the old24 native answers.

## Fixed observations and mechanisms

Only authenticated HM176 SOURCE arrays:386 authors,2 nonoverlapping windows of
17 identified events,8 frozen venue/vector cells,all5 inherited binary labels.
Keep every source author and all32 observed full-label profiles. Reuse the
HM185 hash split/salt (203/183 people); do not search author splits.
Inside BOTH windows split even/odd event addresses into9/8 events. Quarter0
pairs early-even with late-even;quarter1 pairs early-odd with late-odd.
These pair experiments have disjoint event identities. They are NOT iid
replicates, complete topic controls, clinical occasions or attention paths.

Let c be a full5-label profile,pi_c its frozen ALL-SOURCE frequency. Within
bank s and disjoint experiment v,let p_u0,p_u1 be8-cell event proportions.
Construct two legitimate profile/event laws:

    L_same(c,a,b)=pi_c * mean_{u in c,s} p_u0(a)p_u1(b)
    L_product(c,a,b)=pi_c * mean_{u in c,s}p_u0(a)
                                     * mean_{u in c,s}p_u1(b).

They have identical C,A and C,B marginals and full-profile priors. L_product
also includes its chance self matches; it is NOT a strict different-author law.
No products of overlapping events and no iid assumption within windows.
Sampling unit:one author then one uniform event in each prescribed subset.
The reference pi is fixed to keep bank profile prevalence from becoming the
operator difference;this is a standardization,not a natural new population.

For each target q,define

    mu_q(a)=P(Yq=1|A=a)
    psi_kq(a,b)=P_k(Yq=1|A=a,B=b)
    step_kq(a,b)=psi_kq(a,b)-mu_q(a)
    H_q(a,b)=step_same-step_product=psi_same-psi_product.

The next-event evidence update is also Gamma_q(a,b)/P(B=b|A=a),where
Gamma=E[Yq*1(B=b)|A=a]-mu_q(a)*P(B=b|A=a).
This is Bayes/covariance arithmetic,not a new theorem. H is5 target-indexed
conditional actions over64 pairs,NOT320 factors or a recursive full-history
model. Support0 stays missing;do not smooth or renormalize away missing pairs.
The measured delta is a change of OBSERVATION LAW,not a person's changed trait.

## One primary and diagnostic observations

Freeze rho(a,b)=mean_ALL_SOURCE p_full0(a)p_full1(b),without labels. On the
common supported domain of all4 banks/views and both mechanisms,compute

    T_q = .5 sum_ab rho_ab[H_00q*H_11q+H_01q*H_10q]
    T = mean_q T_q.

No rescaling each target or flipping any field. Report reference support mass
without renormalizing;unsupported cells remain explicitly absent. This tests
common target-action orientation,NOTprediction r,AUC or psychometric accuracy.
Also report within-bank/disjoint-text versus between-bank/same-text-pattern
accounts,all5 T_q,and ALL4 condition-pair blocks. These are diagnostics,not
replacement primaries if T is negative. Compute a signed cross-bank5x5 Gram
of target actions,not a PSD-clipped factor model or trait correlation.

199 frozen nulls permute late author windows within full profile AND bank.
Use the SAME permutation for both late quarters,preserving their own
cross-quarter author structure,full profile masses and singleton laws.
Reconstruct H each time;do not pretend the289 correlated event pairs are
independent. Small profiles may have fixed points;retain and report counts.
The resulting tail is an exploratory conditional reference,not a universal
causal/psychological p value. No trait,seed,rank,cutpoint or bandwidth search.

Decision:positive primary orientation above the matched reference supplies a
candidate shared update bank to connect to existing qualified vector actions.
Within-bank recurrence without across-bank support keeps the effect as an
author-configuration mechanism,not a shared target rule. Negative/mixed
condition blocks remain useful localization information;do not rescue the
grand primary with a selected block. Neither outcome makes natural LLM
mediation true automatically.

## Positive construction and prior art

Export an aggregate pair-update bundle that contains NO source author ID,
histogram catalogue,raw text or application outcome. Its query uses two
distinct qualified event IDs and window/cell addresses;repeated event aliases
are rejected rather than counted twice. It exposes before/after/increment/H
and support. It does NOT invent a streaming closure from a two-event law.
No new cold CLI or routine480-input replay just to enlarge the output table.

Song/Huang/Smola/Fukumizu2009 Sec3/Eqs5-6 already builds conditional
distribution operators;finite one-hot observations here need only exact
probability tables,not kernel regularization or a novel CME theorem:
https://icml.cc/2009/papers/538.pdf .
Song/Fukumizu/Gretton2013 reviews sum/product/Bayes operations in feature
space:https://www.gatsby.ucl.ac.uk/~gretton/papers/SonFukGre13.pdf .
OpenAI/math pinned adc7f124 three-state02-foundations110-184 highlights that
posterior sufficiency and product composition have DIFFERENT conditions.
This motivates the matched same-versus-product design,NOTtransfer of its
spin symmetry,independence,tree threshold or new reconstruction theorem.

## Execution boundary

New code/module/config/tests only;prepare/compute/verify each once after
preseal tests. One independent-form direct event-pair sum verifies laws and
update identities;no millions-of-decimal audit. Preserve old hashes/failures.
NoAPI/private raw/MEPS/neural forward/training/Git/Claude/automation/lockbox/
oldqueue/application labels. Record evidence and next decision in parent
report/checkpoint. Goal stays active,not full scientific completion.
