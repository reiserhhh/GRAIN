# Conditional evidence valuation from an explicit SOURCE reference

HM244, 2026-10-13. Result-after theory record. The fixed design is
[HM244 plan](GRAIN_CONDITIONAL_EVIDENCE_VALUATION_PLAN_HM244.md).

## What was added and learned

A cold numerical interface now explains a background-dependent evidence edit
with two exact accounts: changing the posterior composition of the reference
profiles, and changing the edit response inside a profile. Both accounts are
retained, not subtracted as noise. This is a SOURCE-defined explanation, not
an identification of the native model's attention mediation or human traits.

On ALL48 two-window groups of the SAME24 historical authors, the conditionally
independent I law already gives2/240 and5/240 background sign reversals in the
two banks. It has no within-profile companion effect by construction. Therefore
a direction reversal alone does not identify a same-author relational channel.

The J law retaining author pairing gives5/240 and9/240 reversals. Its
within-profile account RMS exceeds the composition account RMS for all five
targets in each bank. However, on the24 COMPLETE native groups, J reversals
overlap the model's26 reversals in0 cases for bank0 and1 for bank1. This closes
the proposed automatic rescue by conditioning the existing Gaussian law.
No scale/sign/kernel/prior was fit to improve that contact.

## Observation law and reference

The HM209 laws remain unchanged. For the32 inherited profiles g, paired
SOURCE prototypes z_iA,z_iB and fixed Gaussian K_s,

    J_g(a,b) = mean_(i:g) K_s(a,z_iA) K_s(b,z_iB)
    I_g(a,b) = mean_(i:g) K_s(a,z_iA) mean_(j:g) K_s(b,z_jB).

The reference is uniform over profiles and uniform over authors within each
profile; it is not empirical population prevalence. Banks contain203/183
authors, total386. Variance is the old0.05525111588037168. All profiles, all
five target bits and both banks remain; conditions are not independently
identified by this declared Gaussian channel. Upstream E5 remains neural.

The192 numeric corners use the original selected/control event edits and
own/cyclic-foreign background windows. The native experiment remains99/192
collected batches and INCOMPLETE. Native comparisons use only the separately
audited24 complete groups/12 authors, not the unobserved numerical half.
Foreign pairing is an audit reconstruction, not a natural human history.

## Exact formula: a fixed evidence edit, variable valuation

At the control-deleted endpoint a_C, condition the profile reference on
background b and target bit q=t:

    w_L,g,t(b) = L_g(a_C,b) / sum_(h:q(h)=t) L_h(a_C,b)
    r_L,g(b) = log L_g(a_S,b) - log L_g(a_C,b)
    D_L,q(b) = log sum_(g:q(g)=1) w_L,g,1(b) exp(r_L,g(b))
               - log sum_(g:q(g)=0) w_L,g,0(b) exp(r_L,g(b)).

This is ordinary Bayes and log-sum-exp calculus. In I the per-profile response
r_g(b) is background invariant, but w_g,t(b) need not be. Coarsening32 profiles
to one binary target mixes conditional laws, and can produce contextual
interaction even without a paired-author prototype index.

In J the companion also reweights SOURCE authors WITHIN each profile, so the
profile edit response changes. The two levels must be distinguished before
calling a contextual interaction an author-specific mechanism.

For Phi_q(w,r), retain the four numeric crossings of own/foreign weights and
own/foreign responses. Their symmetric accounts are

    I_q = Phi00-Phi11
    C_q = (Phi00-Phi10 + Phi01-Phi11)/2
    W_q = (Phi00-Phi01 + Phi10-Phi11)/2
    I_q = C_q + W_q.

These crossings are bookkeeping counterfactuals, not actual text interventions.
Their symmetry avoids choosing one path ordering; it does not give a unique
causal allocation. RMS values cannot be added; the signed C*W term remains.
The implementation can use normalized profile log probabilities: query-common
gauge terms cancel from the high-minus-low valuations. This is not denoising.

## Actual SOURCE accounts

| Bank/law | Mixed RMS range across5 targets | Composition RMS | Within-profile RMS | Sign reversals /240 |
| --- | ---: | ---: | ---: | ---: |
| 0 J | .000858-.001247 | .000398-.000496 | .000636-.001073 | 5 |
| 0 I | .000372-.000479 | .000372-.000479 | numerical0 | 2 |
| 1 J | .000932-.001391 | .000469-.000622 | .000731-.001062 | 9 |
| 1 I | .000472-.000638 | .000472-.000638 | numerical0 | 5 |

These are log-odds action units under specified laws, not personality effect
sizes or240 independent people. The largest I within-profile account is
2.44249e-15, an algebraic independence control, not a human discovery.

All SAME-target J/native centered contacts on the24 complete groups:

| Target | Bank0 | Bank1 |
| --- | ---: | ---: |
| E | +.026897 | -.192785 |
| N | -.174065 | -.120358 |
| A | +.100715 | -.027888 |
| C | -.227507 | +.295769 |
| O | -.432995 | +.049276 |

All5x5 contacts for total/composition/within-profile in both laws and banks are
retained. None select a method or provide psychological prediction r. Native
outcomes were known before this follow-up; no blind-test claim is made.

## Cold interface, evidence identity and reproducibility

`math_conditional_evidence_valuation.cold(bank, record)` accepts exactly a
frame key, focal-view orientation and four addressed window-mean corners.
It rejects outcome fields, duplicate/missing event IDs, changed focal identity
under background substitution, or a changed companion under focal editing.
Caller-supplied means are observations, not a proof that a raw-text encoder is
sufficient. Distinct IDs do not establish stochastic or biological independence.

The interface returns all five actions in both laws and the exact two accounts.
Two separate cold-process readbacks matched the frozen calculations exactly.
Direct-distance Gaussian enumeration verified all1920 SOURCE finite actions,
max error1.19887e-14; account closure0. This is same-agent alternate arithmetic,
not independent blind verification. No new model/API/private/label reads.

Canonical: `results/grain_math_response_v1/conditional_evidence_valuation_r1`.
Prepare/compute/analyze/verify/check each ran once; SOURCE compute.793732sCPU.
28 focused tests and3 Python lint/format checks pass. Pre-freeze EXE002
shebang errors were corrected before SOURCE registration; no scientific retry.

## Prior art and mainline decision

Bayes, finite mixture conditioning and symmetric two-factor decomposition are
standard. HM209 supplied the laws/fields; HM214 supplied the earlier one-window
finite bridge. [Kernel Bayes' Rule2013](https://jmlr.org/papers/v14/fukumizu13a.html)
is related probability-representation work, not the implemented RKHS inverse.
[Patching guidance2024](https://arxiv.org/html/2404.15255v1) already distinguishes
composition, calibration and redundant routes. No first context effect claim.

Pinned OpenAI/math adc7f124, three-state foundations98-125, was re-read for
conditional likelihood tilts and sufficiency WITHIN a declared experiment.
No tree/spin reconstruction theorem or posterior sufficiency result is
transferred to E5 window means, raw natural text or psychological measurement.

The actual incremental capability is an addressed explanation of an evidence
edit under a frozen SOURCE law. Its decision is equally important: old global
T AND this old positive conditional Gaussian law do not automatically read the
native context-dependent evidence action. Stop this scale/curvature/precision/
prior/rank family. Next inventory existing event-level observation/mean-alias
and target-meaning interfaces (HM137/138/173/177/203/204/233) before designing
one discriminator of observation compression versus target-reading mismatch.
Do not merely repeat that means lose information or that context matters.
Choice, state, ownership and condition history stay in the measurement object.
Original18/100/44 remain OPEN; goal ACTIVE/nojob, not whole-theory closure.
