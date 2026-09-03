# GRAIN Theory Formal Notes v1 — derivations, biases, boundary conditions

Created: 2026-07-07 (no-lock falsification program; FALSIFIER_MATRIX rows
T4/T6/T9). Model: for author u, slice i, venue r(i):

```text
y_ui = f_u + b_{r(i)} + gamma_{u,r(i)} + e_ui ,   r(i) ~ pi_u  (choice)
```

f = flesh (person style base), b = venue style offsets, pi_u = person's venue
distribution. All propositions below are elementary but none were previously
written down — and F1/F2/F3 change how the theory's own C2 row must be read.

## F1 (composite mean) — raw style base is a C1+C2 composite

User mean: ybar_u = f_u + m_u + gammabar_u + ebar_u, with the VENUE-MIX term

```text
m_u := sum_r pihat_ur b_r      (E[m_u] = sum_r pi_ur b_r)
```

m_u is a function of the choice profile alone. Whenever choice is
person-stable (T3) and venues are styled (b != 0), m_u is a person-stable
component of "style base". Therefore the uncentered mean — the C2
operationalization — measures f + m: **flesh plus a choice-mediated term**.
The THEORY table's C2 row ("E[y_u] uncentered" = flesh channel) is
mislabeled as written; C2-as-operationalized is a composite. How large the
m-part is on real data is an empirical question (see
run_suica_c2_purity_decomposition_v1.py).

## F2 (shared-set retest cannot purify flesh)

With the same condition set S_u in both halves and stable within-user
shares, m_u^(early) ~= m_u^(late), so

```text
cov(ybar_e, ybar_l) = Var(f) + Var(m) + 2 Cov(f, m) + [gamma/noise terms]
```

Shared-set (and raw) retest carries the FULL mix variance. Wrong-world
evidence (final frozen run, per-world seeds): in the flesh-null world
(f == 0) the raw retest reads 0.637 and the shared-set retest 0.430 — pure
choice-mediated pseudo-stability; the disjoint estimand reads -0.071
(results/suica_theory_wrongworld_suite_v1, W-B). Round-9 sim verification:
the covariance identity holds to +/-0.05 across 6 seeds.

## F3 (disjoint-set estimand: what it removes and what it keeps)
### [round-9 correction applied: independence of b alone is NOT sufficient]

Sets A_u (early) and B_u (late) disjoint, venue effects {b_r} independent
mean-zero across venues, AND **choice independent of b** (pi ⊥ b):

```text
cov(m^A, m^B) = sum_{r in A} sum_{r' in B} E[pi pi'] cov(b_r, b_{r'}) = 0
=> cov(ybar^A_e, ybar^B_l) = Var(f) + Cov(f, m^A) + Cov(f, m^B)
```

Under flesh-coupled choice (kappa != 0) the weights themselves depend on b,
so m^A and m^B are BOTH driven by f and cov(m^A, m^B) ~ alpha_A alpha_B
Var(f) > 0 — a residual term the display above omits (round-9 audit;
empirically visible in W-B3: rho_cond_disjoint 0.58 exceeds even
pure-flesh-share + cross-terms). Additionally gamma_{u,r} correlated within
class re-opens the same path as class-correlated b. So even the
disjoint-set estimand is an upper bound through THREE channels under
coupling: the two cross terms AND the residual m-m term.
The pure choice-carried term dies under (pi ⊥ b), BUT the flesh-choice
CROSS terms survive:
if stylistically tense people also choose tense venues (Cov(f, m) > 0), the
disjoint-set retest still over-reads flesh. Three consequences:

1. rho_disjoint is an UPPER-bound-flavored flesh estimate under positive
   coupling (exact when Cov(f,m)=0);
2. class-correlated venue effects (b_r correlated within content class)
   re-open a mediation path across "disjoint" condition sets — hence the
   class-disjoint variant;
3. the covariance-accounting estimator must be read component-wise:
   cov(mhat_e, mhat_l) targets Var(m) only; the FULL mediated share
   (everything that vanishes if b == 0) is

```text
mediated_total = [Var(m) + 2 Cov(f, m)] / cov_raw
              = 1 - cov(fhat_e, fhat_l)/cov_raw ,  fhat := ybar - mhat
```

with mhat cross-fitted (venue effects from other-user folds; E1's
self-estimation-leak lesson applies verbatim). Wrong-world licenses (final):
W-B/W-B2 validate cov(mhat,mhat)/cov_raw under independent preferences
(max grid err 0.052); W-B2c licenses the CLASS-disjoint estimand under
class-correlated b (class arm reads Var(f) within 0.115 while the
condition-disjoint arm over-reads by +0.106); W-B3 shows mediated_total is
UPWARD-BIASED under strong flesh-coupled choice (0.541 vs target 0.426,
+0.115 — recorded FAIL vs the point tolerance; same composition-bias
mechanism as F4's second-order term) => under coupling, mediated_total is
an UPPER BOUND on mediation, i.e. 1 - mediated_total is a LOWER bound on
the flesh share. Bracketing rule for real data (corrected):

```text
flesh share in [ 1 - mediated_total  (lower bound under coupling),
                 1 - class/cond-disjoint-based shares (upper-bound-flavored) ]
```

## F4 (centering = mediator adjustment; the P2/E1 theorem)

Choice is a MEDIATOR of person->text: f_u -> pi_u -> venue -> y. Condition
centering subtracts bhat_r; **under pi ⊥ b (no flesh-choice coupling)**
out-of-sample/two-way-FE centering removes m_u from ybar_u exactly in
expectation (round-9 oracle-b verification: with TRUE b the identity is
exact, -1.332 vs predicted -1.335), so

```text
Delta cov(retest) = - [ Var(m) + 2 Cov(f, m) ]  <= 0  whenever mix is
person-stable and non-negatively coupled
```

— centering MUST reduce retest covariance by exactly the mediated share.
P2's pooled delta (-0.091) and E1's cross-fit collapse (-0.05..-0.07) are
the empirical shadow of this identity; "centering destroys person signal"
is not a corpus accident but an algebraic consequence of self-selection
(T2). BOUNDARY CONDITION: under exogenous venue assignment (pi identical
across users — assigned-prompt regimes), Var(m) = Cov(f, m) = 0 and
centering removes only estimation noise: harmless to mildly helpful.
This yields the regime prediction for any future assigned-context corpus,
and the phase-diagram simulation in the wrong-world suite (W-PHASE) must
reproduce the sign boundary.

## F5 (precision formula, continuous form — T9)

At fixed n, resampling venues each half adds within-user variance
Var(mhat_u - m_u) ~ (spread of b over u's support)/n, which grows with the
rind heterogeneity of the user's sampling. Retest attenuation = stable /
(stable + within-noise) therefore DECLINES with rind entropy at fixed
volume. Testable gradient (pre-committed in the matrix): retest r by
entropy tercile within volume strata, low > high.

## F6 (overidentifying constraints — free falsifiers)

1. Spearman-Brown consistency along the volume curve (already exercised in
   P1; violations flag non-exchangeable slices).
2. Transport bound: for any register/occasion pair, cross <= sqrt(w1 * w2)
   (single-factor bound). OP-7a: cross 0.739 <= sqrt(0.782*0.857) = 0.819 —
   satisfied; the round-8 SB-benchmark check used exactly this identity.
3. Under the additive model, mediated shares estimated via (i) shared-vs-
   disjoint contrast and (ii) covariance accounting must agree within
   sampling error — disagreement flags either class-correlated b (F3.2) or
   model misspecification (T1) — a built-in misspecification alarm.

## Status

F1-F6 are derivations; their empirical instantiations live in
results/suica_theory_wrongworld_suite_v1 (licenses) and
results/suica_c2_purity_decomposition_v1 (real-data measurement). Theory
doc consequence: THEORY_BASE section-1 C2 row must be relabeled per the
decomposition outcome (pre-committed rule in FALSIFIER_MATRIX).
