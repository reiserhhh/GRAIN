# F15 recovery: predictive state, Blackwell order, auxiliary identification

Date: 2026-09-26 JST. Worker: Claude Opus 5.5. Replaces the rejected F15 r1
(random-number method comparison; LP without the B=AK constraint), which is
retained untouched. Protocol: `docs/GRAIN_F15_RECOVERY_PROTOCOL_V1.md` with
amendment v2 (after Fable seminar 01). Canonical run
`results/grain_gptpro_foundation_v3/F15/recovery_r2/` (COMMON eight files per
card); `recovery_r1` kept (F15-01..03 done, F15-04 failed at CSV writing).
Module `grain_research/gptpro_foundation_v3/f15_theory.py`; runner
`scripts/run_grain_gptpro_f15_recovery.py`; 12 tests. No neural training, API or
private data. All results are exact finite computations on declared worlds.

## Findings that change production rules

1. **A short history can masquerade as a complete state.** In a declared hidden
   order-3 world (y_t depends only on y_{t-3}), the system-dynamics block with
   histories <=2 has rank 4, the same as the order-2 C2 XOR world, but histories
   <=3 reveal rank 8. A linear PSR fitted on the short block fails exact closure
   by .600; the <=3 block closes to 1.7e-14. The C2 XOR PSR (rank 4) closes to
   2.7e-15 over 1,778 history/test checks. The discrete Mori-Zwanzig memory term
   of the 2-step state has mean 0 and variance .09 and is a deterministic
   function of the omitted lag: it is memory, not noise.
2. **The mean score can carry zero information about a pattern target.** F09 main
   family (9 hypotheses; even-corner development design, 3 reps): mutual
   information about the family is .558502 nats from full cell counts and
   exactly 0 from the total count, because each family has two +1 and two -1
   even-corner levels and logistic(b)+logistic(-b)=1. Minimal sufficient
   statistics (175 classes) are Blackwell-equivalent to full data (explicit
   kernels, residual 0 / 2.8e-17); full data are not a garbling of the mean
   (Farkas certificate). Beta MI: full .132272 vs mean .008757.
3. **Process names need the observation that breaks their equivalence.** Belief
   process vs its corner-equivalent direct response: corner reports carry exactly
   0 nats about the process; adding intermediate levels (2 reps) gives 2e-05 nats
   (law gaps .00029/.0022/.0068 for beta .5/1/1.5); observing the belief B gives
   .108857 nats. Corner-only data are a garbling of both richer designs (explicit
   kernels) and not conversely (Farkas).
4. **Auxiliary variability is a rank condition, and detection is not recovery.**
   Five declared conditions give natural-parameter rank 4 = 2n (Khemakhem et al.
   2020 Thm 1(iv), p.5 eq.14) and Hyvarinen et al. 2019 variability rank 4 at 20
   points; four conditions give rank 3 and identical conditions 0. A fixed
   quadratic contrastive discriminator (3 seeds, source-only fit) has application
   log loss .0317 [.0265,.0372] nats per pair below chance (Bayes: .0409 below
   chance) but .0067 [.0041,.0096] above than the same model on the known inverse: it detects
   auxiliary dependence without recovering sources. The null world stays at
   chance (+.0010, slight overfit).
5. **First-order action geometry misses null-direction curvature.** Re-executing
   the frozen F10 quadratic design (replay 2.8e-17): the rank-2 action subspace
   has two null directions whose second-order effects have mean square
   .330/.259/.223 (three seeds).

The F04 construction here is a declared binary analog (L/T/Q, a=0, b=1.5),
not the Gaussian F04 targets: world MI full .467565 vs mean .006882.

## Certificates and retained solver diagnostics

HiGHS LP returns residuals 2.4e-10 to 4.8e-9 on structurally feasible garbling
directions (status NUMERICAL_FAILURE at the 1e-10 threshold); these are retained
and the decision uses explicit exact kernels. Infeasible directions are decided
only by verified Farkas vectors (e.g. full-from-mean: rhs.y = -1, min dual
constraint 0).

## Method adoption (F15-05) and rules (F15-06)

A01–A08 each have primary locators (F01 cards; unavailable primaries recorded),
finite evidence, a status and a remaining item: A01 PSR/bisimulation, A02
state-space identifiability, A04 IB/Blackwell/MZ, A07 ADO/ALBA/CAT adopted
standard mathematics with finite certificates; A03 finite certificates with the
learned estimator NOT_EXECUTED; A05 finite certificate (native chart open); A06
primary unavailable, finite certificate present; A08 mixed (RSA extension
contract; counterfactual adopted with difference; CBM not applicable; SAE not
executed). Six representation rules are listed in
`F15-06/representation_rules.json`; `acceptance.json` proposes the cards for
supervisor review with separate implementation / protocol / science statuses.
No parent card is promoted by this report.

## Requirement evidence

R07 (identification counterexamples), R08 (history vs finite-future
equivalence; short-history insufficiency), R30 (sufficient compression, memory),
A01, A03, A04 (finite certificates), A05 (null-direction curvature), R03/R31
(adoption and representation rules). Standard mathematics is used as such; the
contributions are the finite counterexamples and production rules.
