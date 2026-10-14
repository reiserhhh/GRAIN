# Measured profiles do not fix the joint evidence law: HM175

2026-10-11. SOURCE-only construction and SAME-event cold comparison complete.
This changes evidence composition, not prediction accuracy. Original18/100/44
remain OPEN. No new mathematical or psychological theorem is claimed.

## 1. Why a singleton scale link is insufficient

For386 empirical SOURCE authors, P_u(a) is the old eight-cell event law and
Y_u is the old five-bit Big5 profile. These are calibrated empirical
observations; their conditional product interpretation is a working law.
Let C_u=Y_u, pi_c=Pr(C=c), mu_c(a)=E[P_u(a)|C=c].

The new profile law preserves the FULL joint prior of all five labels and
every singleton target function. All32 profiles are observed, counts3..22;
no class count, clustering, smoothing or native answer is fitted.

    sum_c pi_c mu_c(a)c_q = E[P_u(a)Y_uq].

Yet combining at the profile level is a different operation:

    L_c(E)=E[product_{a in E}P_u(a)|C=c]
    L_c,pool(E)=product_{a in E}E[P_u(a)|C=c].

In general expectation and product do not commute. Define a retained joint
remainder, not discarded noise,

    kappa_c(E)=L_c(E)-L_c,pool(E).

kappa_c(a)=0 for every singleton; for two distinct events,
kappa_c(a,b)=Cov(P_u(a),P_u(b)|C=c). The SAME cell can occur twice, but
distinct event IDs are required. Negative kappa entries are valid; the
complete L_c remains nonnegative. Posterior composition is

    M_q(E)=sum_c pi_c c_q [L_c,pool(E)+kappa_c(E)] /
           sum_c pi_c     [L_c,pool(E)+kappa_c(E)].

This is finite mixture/Bayes algebra, not an invented new formula. Its
construction lesson is substantive: preserving every single-event
text-to-trait relation does not preserve an author measurement from multiple
events. A creator must specify what is shared across the events, not merely
choose a scoring axis and average it.

## 2. Exact nested law decomposition, not an information fraction

For one target q/group g, all ordered two-event cells obey

    A=E[1(Y_q=g)P_u(a)P_u(b)]
    B=sum_{c:c_q=g}pi_c mu_c(a)mu_c(b)
    C=Pr(Y_q=g)rho_qg(a)rho_qg(b)
    A-C=(A-B)+(B-C).

A-B is weighted covariance INSIDE a full measured profile. B-C is covariance
between profile means inside one target group. All signed entries are kept.
The group covariance matrices are positive semidefinite, but their
off-diagonal entries need not be positive. For the signed positive-versus-
negative target contrast, even squared-norm cross terms can be negative.

    ||D_total||^2=||D_within||^2+||D_between||^2
                  +2<D_within,D_between>.

Actual weighted entry RMS: within .004073160; between .000411897. This is an
approximately9.89-fold magnitude contrast in this SOURCE working law, NOT
a fraction of personality, an effect-size r or a variance-cause allocation.
The Agreeableness signed cross term is negative(-.000005036); all other four
are positive. No covariance is removed to make the account smaller.

## 3. Nonmonotonic changes in contextual target evidence

All8 context cells x8 new cells x5 targets are retained. The three laws share
the prior and singleton values but give different context log Bayes updates:

| Law | Direction reversals /320 | Increment-change RMS nats |
| --- | ---: | ---: |
| Target-conditioned iid |0|roundoff5.1e-16|
| Full five-target profile |44|.068072|
| Shared empirical author |85|.187804|

The difference is not simply41 new reversals:24 are shared,61 occur only
with the empirical author mixture, and20 occur only after profile pooling.
Refining the common variable can create, remove or reverse a contextual
interpretation, rather than monotonically strengthen a fixed trait vote.
These320 questions are SOURCE-law enumerations, not320 people or p values.

By target(profile/shared/both/profile-only/shared-only):

- Extraversion:5/13/3/2/10.
- Neuroticism:19/24/12/7/12.
- Agreeableness:13/17/4/9/13.
- Conscientiousness:5/20/3/2/17.
- Openness:2/11/2/0/9.

## 4. Consequence on the SAME natural evidence families

No new neural inference or application labels were used. The480 old cold
families comprise24 authors x2 views x5 subsets x2 vector arms. On these
families, profile versus author posterior probabilities differ by RMS.148962,
maximum.868529; binary log odds differ by RMS.821980 nats, maximum7.487833.
Those differences do not say which law is more correct for a human.

The old native additive-increment violation is reused as motivation only.
No new correlations with native answers, no native-model selection and no
psychological prediction promotion were computed. The old iid and author
comparators replay EXACTLY(error0). Joint label prior error0; singleton
target mean discrepancy1.89e-15. All480 families supported all targets.

Natural subset J_sum(binary log odds) RMS: iid~2.47e-15, profile.330471,
author.571628. The magnitudes describe different working laws; neither
similarity to native RMS nor a larger magnitude identifies the LLM circuit.

## 5. What the remainder means and the next scientific decision

Five binary labels are an insufficient conditional-independence specification
for this empirical author-resampling law. This does not mean Big5 lacks all
five continuous dimensions, or that an omitted personality trait is discovered.
Within-profile variation can include continuous label resolution, unmeasured
traits, selection, persistent habits, occasions and finite text estimation.
P_u was estimated from34 old current events per author; its covariance is
not automatically a population author covariance or natural joint frequency.
The matrices A/B/C are constructed product-mixture laws, not freshly observed
human two-event joint data.

The useful next distinction is reusable within-profile response versus
view-specific empirical heterogeneity. Existing HM170/171 cache has two
disjoint17-event author windows. Inventory its source split and same-event
interfaces before a finite follow-up. Use fixed target profiles and preserved
choice/state conditions to ask what carries across views; do not subtract
multinomial variance under an unverified iid assumption or require unchanged
human personality over time. Do not promote another fit/precision grid.

## 6. Prior art, artifacts and checks

Deep Sets2017/2018 Sections2.2-2.3 explains nonlinear set aggregation and
exchangeable mixtures: https://arxiv.org/html/1703.06114 .
Allman/Matias/Rhodes2009 abstract describes identifiability conditional on
independence structure: https://arxiv.org/abs/0809.5032 ; its generic
identification theorem was NOT applied here. Total covariance and finite
mixture algebra are established tools. Existing HM142/HM166 already used
them in different observation families; this is not first Bayes use.
OpenAI/math adc7f124 three-state foundations.tex120-235 was read for its
common-input product rule. Its tree/spin/threshold theorem does not transfer.
Newness here is the controlled SAME-natural-event profile comparator and
retained multi-event remainder interface, not new mathematics.

Canonical results/grain_math_response_v1/target_profile_composition_r1.
19new/43focused tests,5Python lint/format PASS. Alternate direct-product
audit14400 scalars(max8.22e-15) and centered-covariance audit1280 entries
(max1.27e-16), same-agent not blind. Four scientific commands all exit0;
no retry, new neural/API/train/Git/Claude/private/lockbox or old queue.
Cold CLI.612s; analysis.166s. GoalACTIVE; no background or next execution job.
