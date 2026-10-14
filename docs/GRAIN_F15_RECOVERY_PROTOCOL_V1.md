# F15 recovery protocol v1: bounded theory certificates (replaces rejected r1)

Frozen 2026-09-26 before the canonical run (smoke diagnostics only before this).
Implements `docs/GRAIN_GPTPRO_ATOMIC_PROTOCOLS_V3.md` section F15 and cards
F15-01..F15-06 (coordination repo). The rejected `results/.../F15/r1` (random
numbers, unconstrained LP) is retained untouched; this is `F15/recovery_r1`.
Module `grain_research/gptpro_foundation_v3/f15_theory.py` (reuses the repaired
`f15.garbling_certificate` and `f15.mutual_information`).

## A. PSR / Hankel (F15-01; R08, R30, A01, A04)

Worlds: C2 XOR (p(next=a XOR b)=.75, order 2, exactly as `f02.c2_next_probability`),
C2 iid, and a declared hidden order-3 world p(y_t=1|y_{t-3})=.8/.2 (lags 1–2
uninformative). Exact stationary laws. System-dynamics blocks D[h,t]=P(t|h) for
all binary histories |h|<=H and tests 1<=|t|<=3, H in {2,3,4}; SVD rank with
tol=max(shape)*eps*smax. Linear PSR from a finite block (greedy core tests,
least-squares projection vectors); closure = iterated-update prediction of
P(t|h) for every |h|<=6, |t|<=3 against the exact law (1,778 checks). Memory
term = P(y|last L history) - P(y|last `short`) weighted by the stationary law.
Decision rule: a PSR is CLOSED if closure error <=1e-10; a short-history block
is INSUFFICIENT if its PSR fails closure while a longer block closes.

## B. Blackwell / garbling (F15-02; R30, A04)

Same-law channels over declared hypotheses:
- F09: 9 hypotheses (family k in {1,2,3} x beta in {.5,1,1.5}), the 4 even
  corners x 3 reps (development design): full cell counts (256 outcomes), mean
  statistic (total count, 13), minimal sufficient statistic (outcomes with
  proportional likelihood columns).
- F04 binary analog (constructed, declared): conditions {-1,-1/3,1/3,1},
  p=logistic(1.5 g_w(z)), worlds L g=z, T g=sign(z), Q g=2z^2-1, 2 reps.
Certificates: explicit deterministic/reverse kernels for feasible directions
(residual and row-stochasticity reported), LP (HiGHS) status retained for all
directions, Farkas certificate for infeasible directions. Exact mutual
information (uniform prior) for hypothesis, family, beta and world targets.

## C. Auxiliary identification (F15-03; R07, A03)

Five equal-frequency conditions (mu1,mu2,v1,v2) = (0,0,1,1),(1,0,1,1),(0,1,1,1),
(0,0,2,1),(0,0,1,2); mixing x=(s1+.2 s2^3, s2). Rank of L=(eta(u_j)-eta(u_0))
(Khemakhem et al. 2020 Thm 1(iv), p.5 eq.14; required 4); the first four
conditions (max rank 3) and identical conditions (rank 0) are counterexamples.
Hyvarinen et al. 2019 Thm 1 assumption 3 (2n+1 values, p.4-5 eqs 8-9) checked
at declared points. Analytic inverse roundtrip reported as a known inverse, not
a learned one.

## D. Fixed contrastive discriminator (F15-04; A03, A05)

Seeds 20260915/16/17; roles source 3000 / validation 1000 / application 2000
samples, u permuted only within role. Positive = true (x,u), negative = permuted
u. Features phi(x)=(x1,x2,x1^2,x2^2,x1x2), onehot(u), phi(x) x onehot(u);
StandardScaler + LogisticRegression(C=1, lbfgs, maxiter=1000) fit on source only.
Arms: quadratic_x (protocol), quadratic_true_s (known-inverse information
reference), null world (all u identical). Metrics on validation/application:
accuracy, log loss (clip [1e-6,1-1e-6], clip count), AUC; exact Bayes log-loss
reference from the known density ratio (unit Jacobian). No neural training;
iVAE / learned nonlinear ICA / SAE are NOT_EXECUTED by protocol. F10 action vs
PCA rank-2 reference is re-read from `F10/geometry_r1` with hashes.

## E. Reviews (F15-05, F15-06)

Method adoption for A01–A08 with primary-source locators verified in
`F01/primary_sources_r1` (or explicit unavailability), finite certificates,
code, status (ADOPTED_STANDARD / FINITE_CERTIFICATE / NOT_APPLICABLE /
NOT_EXECUTED + minimal extension contract). Representation rules only from the
above certificates. No parent promotion by this packet alone.

## Amendment v2 (2026-09-26 01:45 JST, after Fable seminar 01; frozen before recovery_r2)

recovery_r1 retained: F15-01..03 completed, F15-04 failed at CSV writing (mixed
row schemas). recovery_r2 re-runs all cards with these additions; r1 is not
edited. Directory note: `F15/r1` holds the rejected legacy run, so recovery uses
new revision directories (COMMON "new changes get new revisions").

- B2 F09 process sub-world (original protocol F09/F15-B): atoms belief(beta)
  (gamma=1.2) and its corner-equivalent direct no-intercept logistic
  (protocol beta') for beta in {.5,1,1.5}. Experiments with 2 reps per level:
  E1 z1 in {+1,-1}; E2 = E1 + {+1/3,-1/3}; E3 = E1 with the belief B observed
  (direct process: B declared independent of Y with the same Bernoulli law).
  Certificates: explicit marginalization kernels E2->E1, E3->E1; LP/Farkas for
  the reverse. MI about process class and beta.
- Predictive target: in the F09 main family the odd-corner future law is in
  bijection with the 9 hypotheses, so the hypothesis MI is the predictive-state
  MI (declared, not a separate object).
- The F04 construction is labelled `F04_binary_analog_declared` (hypotheses
  L/T/Q uniform, a=0, b=1.5 fixed, 2 reps); it is a new declared world, not
  the Gaussian F04 targets.
- D contrastive: unit = sample (the ICA world has no authors); streams
  SeedSequence([seed,15,3,role,null]); paired bootstrap over application
  sample indices (2000 reps, seed 20260915, seeds stratified) of per-pair log
  loss differences: model - chance and quadratic_x - quadratic_true_s.
  SUPPORTED if the 95% upper bound < 0, else INCONCLUSIVE.
- D2 F10 reference is recomputed by re-executing the frozen F10 quadratic
  design through `f10_geometry` (not re-read) and replayed against
  `F10/geometry_r1/metrics.csv`; null-direction second-order magnitude reported.
- E adds an item-by-item RSA / NLI / concept-bottleneck / SAE table: observed
  object, intervention, non-unique decomposition, whether it solves the F15
  question.
