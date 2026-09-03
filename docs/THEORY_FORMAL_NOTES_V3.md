# GRAIN Theory Formal Notes v3 — F11: the motion layer at scale (V6 program)

Created 2026-07-12. Extends F1–F6 (V1) and F7–F10 (V2). Operator directive: mount a
large-scale program on the F10.8 discovery (coherence that exists only in motion), reach
every corner by repeated experiment, identify theorem-level similarities with advanced
mathematical tools, extend them to personality inference, and complete THEORY V6.

Discipline unchanged: predictions and kill conditions are REGISTERED AND COMMITTED BEFORE
each run; wrong leans are recorded, not repaired. All work is Tier-U / label-free; the
Essays corpus is used TEXT-ONLY (the loader reads columns user_id and text and never the
trait columns; no seal is touched; the confirm-half label budget is unaffected).

## F11.0 — The motion layer, consolidated model

Per text (comment/essay), window k of m:

    w_k = p + k·s + g_k        p ~ (μ_u, Π)   person/text level
                               s ~ (σ_u, Σ_flow)  ordered drift along the text
                               g_k ~ (0, Γ)   gust (transient co-fluctuation)

plus the F10.6/F10.7 position structure entering through E[w_k] as inner (boundary-zone)
and outer (relative-t) profiles. Established so far (F10.8): Γ has factor structure
partly invisible in level space (B-gust1: replication .829, level-visibility 0.972,
vocabulary-disjoint loadings); Σ_flow has at least one certified factor (statically
visible); the person-level invisible candidate failed replication at gate ≥3.

## F11.1 — E1 registered predictions: gust robustness sweep (PANDORA)

Arms on the m=3 cohort (618 comments): (a) venue-class replication (classes with >= 60
comments), (b) temporal median-split replication, (c) user-demeaned q contrasts, (d)
window-scale 64 tokens (same comments, first/mid/last 64-token windows).
LEANS: gust1 congruence >= .6 across venue classes; >= .7 across time halves; >= .8 under
user-demeaning (gusts are within-comment objects; person effects should barely matter);
scale-64 congruence positive but attenuated (.4–.6) — gusts may have a characteristic
token scale. KILL: venue congruence at chance (~.23 random baseline) would demote gust1
to a venue-composition artifact.

## F11.2 — E2 registered predictions: Essays transport (the far corner)

Essays = different register (stream-of-consciousness), different population, long
documents (dense interiors that PANDORA lacks). Battery transports via the frozen wcl
transform (built for Essays transport) + v2 scorers. Windows: 128 tokens, cap 12,
endpoint-preserving; leak mask per window, drop essay on any hit.
(a) Position profiles: first_person monotone decline TRANSPORTS (lean YES, flow-type);
    tension closing jump ATTENUATES or vanishes (lean: it is a Reddit discourse act —
    open); directive opening elevation OPEN.
(b) Law-of-the-wall (matched-asymptotics analogy): on m >= 4 essays the within-essay
    position model composite = inner(start-zone) + inner(end-zone) + outer(t) beats both
    single-coordinate models by essay-level CV for the MIXED constructs; overlap check =
    single-coordinate models agree in the interior and disagree at boundaries.
(c) Flow/gust separation on (first, mid, last) triples: gust structure EXISTS on Essays
    (supra-edge, author-half replication) — lean YES.
(d) CROSS-CORPUS HEADLINE BET: Essays gust1 congruent with PANDORA gust1 with |cos| >= .5
    (lean; random baseline ~.23). KILL: failure demotes the gust factor from
    corpus-general to register-specific (still real, smaller claim).
(e) Essay-level flow-only retry (the F10.8 phase-C death, transported): supra-edge +
    author-half replicating + L-invisible motion component on Essays — OPEN, lean YES at
    this power (each essay carries m ~ 4-6 windows).

## F11.3 — E3 registered predictions: gust dynamics

On Essays m >= 4 (and PANDORA m >= 4 descriptively): within-text detrended residuals
projected on the gust axis give a per-window gust series γ_k.
(a) AR(1) of γ_k vs within-essay order-permutation null: LEAN ≈ 0 (white gusts — iid
    assumption holds); significant positive AR would upgrade "gusts" to "weather"
    (mean-reverting OU-like state, a different math object).
(b) |γ_k| vs position: LEAN boundary-elevated (gusts cluster where discourse work is
    done — openings/closings).

## F11.4 — E4 registered predictions: person-level retry (PANDORA)

Power raise on the F10.8 phase-C failure: gates >= 5 and >= 8 long comments; and the
sharp arm residualizes user mean-slope vectors against the user-level static top-4 frame
BEFORE factoring (invisibility by construction; the question becomes existence +
replication). LEAN: at gate >= 5 with residualization, at least one motion component
clears the edge with half-replication >= .5. KILL: failure at both gates bounds the
person-level flow-only factor below current-corpus detectability (native-corpus item).

## F11.5 — Measurement-economics theorem (registered as theory; instrument deferred)

In the F11.0 model with N texts of m windows each: the level estimate μ̂_u has
Var ≈ (Var(p) + Γ_c/m)/N (occasion-limited: dominated by 1/N), while the gust-structure
estimate from within-text contrasts accumulates (m−2) contrasts per text:
Var(Γ̂ entries) ∝ 1/(N·(m−2)) (token-limited). Consequently DIFFUSION-TYPE (motion)
descriptors are cheap in occasions and expensive in tokens; LEVEL-TYPE descriptors are
the reverse. This is the discrete analogue of the classical diffusion-statistics
asymmetry (drift error ~ 1/√T, diffusion error ~ 1/√n within-path). Single-text
sufficiency instrument (cross-text reliability of one-text gust loadings vs one-text
level scores at matched token budgets) is DESIGNED but not executable on current corpora
(PANDORA: too few m >= 4 texts per user; Essays: one text per person) — declared open for
the native corpus (OP-36).

## F12 — MA(1) gusts: exact identities, an identifiability theorem, and the correction
## of our own flow estimator (registered 2026-07-12, before the W2 runs)

Process note (operator directive): analysis/planning by Fable, execution by Sonnet
agents; mathematical reduction to the base model over lexical narration; iterate
experiment -> correction -> theory.

**F12.1 (model (M′) and exact identities; derived, not fitted).** Replace iid gusts in
(M) with MA(1): g_k = η_k − θ η_{k−1} (multivariate: g_k = η_k − Θ η_{k−1}). Write
Γ0 = Cov(g_k), Γ1 = Cov(g_k, g_{k+1}), B = (Γ1 + Γ1ᵀ)/2; E3's anti-persistence means B
has a negative direction. Exact consequences:

(i) Second differences Δ²w_k = w_{k+1} − 2w_k + w_{k−1} kill level and linear drift
    PER TRIPLE (no OLS, no leverage). Univariate lag-1 autocorrelation of Δ² under (M′):

        ρ1(Δ²)(θ) = −(4(1+θ²) + 7θ) / (6(1+θ²) + 8θ),

    equal to −2/3 iff θ = 0, strictly decreasing to −3/4 as θ → 1. This is the
    leverage-free replacement for E3: the iid hypothesis pins ρ1(Δ²) at −2/3 exactly.

(ii) Multivariate moment identities: S0 ≡ Cov(Δ²) = 6Γ0 − 8B and sym S1 ≡
     sym Cov(Δ²_k, Δ²_{k+1}) = −4Γ0 + 7B. Hence the EXACT inversions

        Γ̂0 = (7·S0 + 8·sym S1)/10,       B̂ = (6·Γ̂0 − S0)/8.

(iii) Wide difference d = (w_{m−1} − w_0)/(m−1) has Cov(d) = Σ_flow + 2Γ0/(m−1)²
     with NO Γ1 term (endpoint gap ≥ 2), so a CORRECTED flow estimator exists on m ≥ 5:

        Σ̂_flow(corrected) = Cov(d) − 2Γ̂0/(m−1)²   (per-m stratum, precision-pooled).

**F12.2 (identifiability theorem — our own P9 estimator is contaminated).** On adjacent
3-point support, Cov_ℓ = Σ_flow + Γ0/2 (the ℓ kernel is γ1-free) and Cov_q = Γ0 − (4/3)B,
so the F10.8 estimator satisfies EXACTLY

        Σ̂ = C_ℓ − C_q/2 = Σ_flow + (2/3)B.

Σ_flow and B are therefore NOT separately identifiable on m = 3; the P9 "ordered flow
factor" (λ1 = .486, rep .809) is certified only under B = 0. With B in the E3 direction
the estimator is biased along B's structure. STATUS DECISION (theory-level): the P9 flow
claim is DOWNGRADED to conditional-on-iid-gusts pending an m ≥ 5 correction in that
register (PANDORA m ≥ 5 is too thin: ~41 comments).

**F12.3 (registered leans for the W2 runs; Sonnet executes).**
(a) W2a Δ² dynamics: Essays ρ1(Δ²) below the iid simulation null (matched m's, same
    centering pipeline); gust-axis θ̂ in [.1, .5]. KILL: ρ1(Δ²) consistent with the iid
    null strikes E3's anti-persistence as pure leverage artifact and restores white gusts.
(b) W2b corrected separation (Essays m in 5..12): corrected λ1(Σ̂_flow) RISES vs the
    uncorrected value (the +2/3·B contamination is negative-leaning), but author-half
    replication stays < .5 — flow remains uncertified on Essays (lean).
(c) Bounce factor: λ1(−B̂) exceeds the within-text permutation null; author-half
    replication ≥ .6; top eigenvector congruent with Γ̂0's top ≥ .6 (bounce structure ≈
    gust structure, common-θ reading). Level-visibility: open.
(d) The F12.2 downgrade of P9 stands regardless of W2 outcomes (it is a theorem).

## F12.4 — Exact finite-n correction of the centered moment estimators (derived and
## registered 2026-07-12, before the W3 run)

**Derivation (Fable, closed form; the deployed estimator centers D2 within text, which
biases the naive inversion — quantified exactly here).** With n = m−2 D2 rows per text
and MA(1) gusts, D2 is MA(3) with autocovariances (all linear in γ0, γ1):
c0 = 6γ0 − 8γ1, c1 = −4γ0 + 7γ1, c2 = γ0 − 4γ1, c3 = γ1, c_h≥4 = 0. Within-text
centering subtracts the mean D2, and the pooled centered moments have EXACT expectations

    Ā(n) = (4γ0 − 4γ1)/n²        A_end(n) = (3γ0 − 4γ1)/n   [n ≥ 4]
    E[S0](n)    = (6 − 4/n²)·γ0 + (−8 + 4/n²)·γ1
    E[symS1](n) = (−4 + 2(n−2)/(n²(n−1)))·γ0 + d(n)·γ1,
        d(n) = 7 − 4/n² for n ≥ 4;  d(3) = 56/9  [BOUNDARY CORRECTION]

CORRECTION (caught post-registration by the deployment agent's independent derivation,
confirmed by direct n = 3 enumeration and 1.5M-text Monte Carlo): the closed-form
simplification of A_end dropped the c3 clip at n = 3 — the lag-3 term cannot enter the
boundary sum there, so A_end(3) = (3γ0 − 5γ1)/3 and d(3) = 56/9, not the formula's
59/9 (a, b, c are exact at all n ≥ 3). The registered verification constants (n = 6)
are unaffected; the W3 Essays re-inversion, whose largest stratum is m = 5 (n = 3), was
RERUN with the exact d(3) and the recorded numbers below are the corrected-formula ones.

(mixed corpus: pool the coefficients with row weights n_t for S0 and pair weights
n_t − 1 for symS1). The identities are linear with the SAME scalar coefficients for
every matrix entry, so the corrected multivariate inversion is the 2×2 solve
(Γ0, B) = M(m-composition)⁻¹ (S0, symS1). VERIFICATION VALUES (to be reproduced by the
executing agent): at θ = .4, m = 8 (n = 6): E[S0] = 9.9867, E[symS1] = −7.3440; the
NAIVE inversion on these gives θ̂ = 0.4408 — matching the deployment agent's empirical
0.44–0.45; at θ = 0, m = 8 the naive B̂ bias is −0.0178·γ0.

**Quantitative consequence registered as a prediction:** W2b's Essays B̂ diag mean
(−0.0103, with Γ̂0 ≈ 0.73 and mixed m 5–12) is CONSISTENT WITH PURE CENTERING BIAS AT
θ = 0 (predicted ≈ −0.013·0.73/0.73-scale). W2a is unaffected (its simulation null ran
the same centered pipeline — internally consistent by construction).

**W3 registered leans (BEFORE the run).** (i) Synthetic θ = .4, m = 8: naive θ̂ =
0.4408 ± sampling (validating the closed form); corrected recovers 0.40. (ii) Essays
re-inversion with the corrected map: mean diag B̂ moves to ≈ 0 (|mean| < .004); the
small-θ constructs (tension .070, wcl_23 .103, wcl_54 .057) shrink by more than half;
wcl_60 stays clearly positive (corrected diag θ̂ ≥ .25). KILL: if the corrected Essays
B̂ mean does not move toward 0, the "W2b's tiny negative B̂ is centering artifact"
reading is wrong. (iii) Deployment: motion.py's inversion switches to the corrected
solve (naive kept as an option for comparability); the documented bias note becomes a
resolved item.

## F12.5 — Lag-2 identities: MA-vs-AR discrimination (registered 2026-07-12, before W4)

**The gap.** The signed memory coefficient r = B/Γ0 cannot distinguish a one-step MA
echo (γ_h = 0 for h ≥ 2) from a decaying AR carry-over (γ_h = φ^h γ0). Lag-2 structure
can.

**F12.5.1 (general identity).** The uncentered Δ² autocovariance sequence is the
fourth-difference filter applied to the gust autocovariance sequence:
S_h = Σ_l [1, −4, 6, −4, 1]_l · γ_{h+l−2}. In particular S2 = γ0 − 4γ1 + 6γ2 − 4γ3 + γ4
(this reproduces S0 = 6γ0 − 8γ1 + 2γ2 and S1 = −4γ0 + 7γ1 − 4γ2 + γ3 as the h = 0, 1
cases). Under MA(1), γ_{h≥2} = 0, so S2 is EXACTLY PREDICTED by the identified pair:
S2_pred = Γ0 − 4B (entrywise, symmetrized).

**F12.5.2 (discriminator).** R2 ≡ S2_obs − S2_pred(Γ̂0, B̂), all three moments taken with
their EXACT within-text-centering corrections. Under MA(1), E[R2] = 0; under AR(1) with
small φ, R2 ≈ 6φ²γ0 > 0; R2 < 0 falls outside both one-parameter families (→ MA(2)+).
METHOD NOTE (the d(3) lesson): centering-corrected coefficient maps for S0/S1/S2 are to
be computed by EXACT ENUMERATION (Toeplitz covariance of the D2 rows under the
hypothesized γ-structure, centering projection P = I − J/n, read the pooled moments), not
by hand-simplified closed forms. S2 pairs exist for n ≥ 3 (n − 2 per text).

**F12.5.3 (per-construct scalar overidentification).** For each construct, fit both
one-parameter families to the centered (S0, S1, S2)_cc diagonal triple — MA(1): (γ0, θ);
AR(1): (γ0, φ), the AR map computed by exact enumeration from the AR(1) Toeplitz — and
compare overidentification residuals (2 parameters, 3 moments, 1 dof each).

**W4 registered leans (BEFORE the run).** (a) Corpus-mean diag R2 slightly POSITIVE
(bootstrap CI excluding 0 on the positive side) — the carry-over decays over more than
one window. (b) Content carry-over constructs (wcl_35, wcl_36, wcl_03, first_person,
wcl_13) classify AR-type (R2 > 0; AR fit beats MA). (c) wcl_60 classifies MA-type
(burst-echo is one-step: R2 ≈ 0; MA fit beats AR). KILL semantics: R2 ≈ 0 everywhere →
MA(1) suffices and the AR extension is REJECTED on parsimony; widespread R2 < 0 → both
families fail and the model moves to MA(2)+.

## F12.6 — The gust ladder: lag-structure inversion and the period-2 question
## (registered 2026-07-12, before W5; REFINES the W5 sketch in F12.5 results)

**Design correction (recorded).** F12.5's closing line proposed lag-3 as the period-2
separator (γ3 < 0 under oscillation). That works only for PHASE-LOCKED alternation. The
observed wcl_20/22 shape (lag-1 ≈ 0, lag-2 large positive) fits an AR(2)-even process
(x_k = a2·x_{k−2} + η, a1 ≈ 0 — a damped standing wave), which predicts γ3 ≈ 0 and
γ4 = a2·γ2 > 0; MA(2) predicts γ3 = γ4 = 0. The general instrument is therefore the
LADDER INVERSION, not one more residual.

**F12.6.1 (instrument).** Extend the exact-enumeration centering maps to S3 and S4
(lag-h pairs need n ≥ h+1; lag-4 draws only on m ≥ 7 texts, ≈ 1.1k pairs — thin,
declared). Build the exact centered 5×5 linear map (γ0..γ4) → (S0..S4) from the pooled
n-composition (unit-vector enumeration; report the condition number), invert for the
GUST LADDER γ̂0..γ̂4 per construct (assuming γ_{h≥5} = 0), bootstrap over texts for CIs.
Also report the Nyquist spectral value f(π) = γ0 − 2γ1 + 2γ2 − 2γ3 + 2γ4 (period-2
energy) per construct.

**F12.6.2 (classification by ladder shape).** MA(2) echo: γ3, γ4 ≈ 0 with γ2 > 0.
Damped standing wave (AR(2)-even): γ4/γ2 = a2 ∈ (0, 1), γ3 ≈ 0. Phase-locked
alternation: γ3 < 0. AR(1) carry-over: geometric ladder γ2/γ1 ≈ γ1/γ0. MA(1): ladder
dies after lag 1.

**W5 registered leans (BEFORE the run).** (a) wcl_20/22: γ2 CI-solid positive AND
γ4 > 0 with γ4/γ2 ∈ (0.1, 0.9) — the damped-standing-wave reading (a sustained ~2-window
≈ 256-token rhythm; guess: paragraph-scale alternation), rather than MA(2) (γ4 ≈ 0).
(b) wcl_60: ladder confined to lag 1. (c) wcl_35/36/13: geometric-ish decay consistent
with their AR φ̂ ≈ +0.1. (d) The 5×5 inversion is expected ill-conditioned at high lags
(centering attenuation grows with h) — if the bootstrap CIs on γ3/γ4 are too wide to
classify, the verdict is INDETERMINATE-AT-THIS-N, recorded as such (no forcing).
KILL: γ4 CI-solid NEGATIVE for wcl_20/22 would reject both registered families and
reopen the model class.

## F12.7 — The normalized Nyquist ratio: a one-number dynamics fingerprint
## (registered 2026-07-12, before W6)

**Definition.** ρ_π ≡ f(π)/γ0, the period-2 spectral energy relative to total gust
variance. Exact reference values: white = 1; MA(1) memory θ: ρ_π = (1+θ)²/(1+θ²)
(bounce θ = .4 → 1.69; carry-over θ = −.4 → 0.31); AR(1) φ: ρ_π = (1−φ)/(1+φ)
(φ = +.1 → 0.82); AR(2)-even a2: ρ_π = (1+a2)/(1−a2) (a2 = .4 → 2.33). One number
separates bounce/alternation (> 1), carry-over (< 1), white (= 1).

**Resolution of the W5 units puzzle (prediction, not post-hoc):** W5's RAW f_π ranked
wcl_60 lowest — but wcl_60 is the certified bouncer, which must have HIGH relative
Nyquist energy; its low raw value reflects a small γ0 in pooled-standardized units.
Under normalization the ranking must INVERT for wcl_60. Estimation: f_π and γ0 via the
exact left-functionals of the centered moment map (the W5 left-solve is well-conditioned,
|v| ≈ 4.1, unlike the ladder coordinates); graceful degradation to a lag-2-truncated
functional when lag-3/4 pairs are scarce (flagged mode).

**W6 registered leans (BEFORE the runs).** Essays, normalized ratios with bootstrap CIs:
(a) wcl_60 ρ_π > 1, CI-solid (bounce; ≈ 1.4–1.7 at its θ̂ ≈ .23–.34). (b) The carry-over
set (wcl_35/36/13, first_person, wcl_03) ρ_π < 1 (≈ .8–.95). (c) wcl_20 and tension
ρ_π > 1 (the period-2 flag, now in the right units). (d) Deployment: motion.py gains
period2_energy (normalized ρ_π per construct) with synthetic tests recovering the four
reference values above. KILL: wcl_60 ρ_π ≤ 1 CI-solid would contradict its certified
bounce and reopen the wcl_60 story.

## F12.8 — The spectral shape pair: mid-frequency separator and the hybrid functional
## (registered 2026-07-12, before W7)

**Definitions.** f(π/2) = γ0 − 2γ2 + 2γ4; ρ_{π/2} = f(π/2)/γ0; shape pair =
(ρ_{π/2}, ρ_π); Δ_shape = ρ_{π/2} − ρ_π. Exact references: white (1, 1), Δ = 0.
MA(1) ANY θ: ρ_{π/2} = 1 EXACTLY (γ2 = 0) — a sharp invariant. AR(1) φ:
ρ_{π/2} = (1−φ²)/(1+φ²) ≈ .98 at φ = .1, so Δ_shape ≈ +.16 > 0. AR(2)-even a2:
ρ_{π/2} = (1−a2)/(1+a2) — strongly depressed. Even-lag anomaly (γ2 > 0) hits ρ_{π/2}
directly: Δ_shape is pushed NEGATIVE. So the sign of Δ_shape separates carry-over
(positive) from even-lag anomaly (negative), and ρ_{π/2} = 1 tests pure-MA(1) stories.

**Hybrid functional (deployment fix, from the W6 conditioning findings).** Numerator
spectral functionals (f_π, f_{π/2}) from the 5-moment left-solves (|w| ≈ 4, conditioned);
the DENOMINATOR γ0 from the truncated-3 left-solve (|w| ≈ 5.8) — per-functional
conditioning discipline. Prediction: the hybrid restores the deployed AR(2) m = 8
calibration toward the asymptotic value (the executing agent computes the exact expected
value of the shipped estimator and asserts against IT, per established practice).

**W7 registered leans (BEFORE the runs).** (a) Essays: carry-over set (wcl_35/36/13,
first_person, wcl_03) Δ_shape > 0. (b) wcl_20/22: Δ_shape BELOW the carry-over set's,
lean CI-separable at least against wcl_35; ρ_{π/2} depressed. (c) wcl_60: ρ_{π/2} ≈ 1
(pure-MA(1) invariant); CI-solid deviation from 1 would break its one-step-echo story.
(d) Deployment: hybrid mode ships with the shape pair as "spectral_shape_by_construct";
all reference invariants tested (MA(1) ρ_{π/2} = 1 exactly is the cleanest new test).
KILL: wcl_20 Δ_shape ≥ the carry-over set CI-solid would kill the even-lag-anomaly
reading as well, reopening the model class for W4's R2 excess.

## F12.9 — The signed two-parameter unification: AR(2) texture triangle
## (registered 2026-07-12, before W8)

**Claim.** The four observed gust textures are regions of ONE family: AR(2) gusts
x_k = a1 x_{k-1} + a2 x_{k-2} + eta on the stationarity triangle. Carry-over =
(a1 > 0, a2 ~ 0); bounce = (a1 < 0); interleaved carry-over / period-2 = (a1 ~ 0,
a2 > 0); damped period-4 = (a1 ~ 0, a2 < 0). MA(1) is NOT nested — genuine one-step
echoes must FAIL the AR(2) fit, which retests W4's MA-vs-AR verdicts from a new angle.

**Instrument (W8).** Per construct, fit (a1, a2) to the THREE hybrid estimable
functionals (signed memory r_c, rho_pihalf, rho_pi) by grid least squares over the
stationarity triangle (references exact via Yule-Walker + the enumeration engine's
exact-on-composition expected functionals); 3 moments, 2 parameters, 1 dof —
overidentification residual = the fit gate. Bootstrap CIs on (a1, a2) and residual.

**W8 registered leans (BEFORE the run).** (a) wcl_35/36/13: a1 ~ +0.1, a2 ~ 0, small
residual. (b) wcl_20: a1 ~ 0, a2 ~ +0.15; wcl_07: a1 ~ 0, a2 ~ -0.15..-0.2; both small
residuals (the family's raison d'etre). (c) wcl_60: WORST overid residual of the 19
(genuine MA bounce, not AR-representable) — this is the sharp test. (d) first_person /
wcl_03 (one-step echoes): residual larger than the carry-over set's. KILL: if wcl_60
fits AS WELL as the median construct, W4's MA-vs-AR distinction was noise and the
taxonomy collapses to AR(2)-only.

## F12.10 — Format-aware windowing: lifting or confirming the PANDORA flag
## (registered 2026-07-12, before W9)

**Question.** W2a flagged PANDORA's long-comment dynamics as uninterpretable (rho1(Delta2)
saturating toward -1, theta boundary hits, wcl_11 arm-flip) with QUOTE/LIST FORMAT
PERIODICITY as the suspected mechanism. W9 tests the mechanism directly: strip Reddit
formatting (quote lines starting with ">"/"&gt;", list markers, code fences/inline code,
collapsed whitespace) BEFORE tokenization, re-window, re-run the leverage-free Delta2
dynamics on the same texts; plus a format census (quote/list/code line shares by length
stratum).

**W9 registered leans (BEFORE the run).** (a) The raw arm reproduces W2a's numbers
(anchor; assert direction and magnitude within bootstrap noise). (b) Format census:
formatting is CONCENTRATED in long comments (m >= 4 share >> m = 2-3 share). (c)
Stripping moves the gust-axis rho1(Delta2) TOWARD the iid band by at least half the
excess and reduces theta-saturation counts — the format hypothesis. KILL: if the
stripped dynamics are UNCHANGED (excess persists), the format hypothesis dies and
PANDORA long comments carry genuine extreme register texture (the flag then becomes a
finding). (d) Cohort shrinkage after stripping is expected and reported (quotes removed
-> shorter texts); conclusions stay diagnostic at this n (~50-70 texts), stated.

## F13 — EXPL-3: do motion-layer features add exploratory detection power?
## (registered 2026-07-12, before the run; operator-ordered)

**Question (operator).** How much did detection power improve v4 -> v5 -> V6? Confirmed-
tier power is deliberately unchanged (no new seal). v4 -> v5 exploratory weight-fit was
+51% (.0902 -> .1361, EXPL-2 arm C). V6's contribution is UNMEASURED — EXPL-3 measures it
on the SAME spent opening-#1 labels under the same governance (exploratory banner; gate
replicated verbatim; H2/H6 reproduced to < 1e-9 BEFORE any fitting; ledger row; no new
labels; Essays confirm half untouched).

**Feasibility pre-check (measured label-free, recorded).** tier_l_comments is 1,500-char
capped like Tier-U (max = 1500; 7,047 at exactly 1500) — so m >= 3 windows are
prose-impossible (93 users have any); the V6 feature set REDUCES to the slope/position
layer on m = 2 long texts (open/close pairs): 10,300 texts across 1,050/1,412 Tier-L
users (median 5 per user; 715 users >= 3, 541 >= 5). Spectral texture (r_c, rho_pi,
shape pair) is NOT computable for labeled users under the frozen extraction — the cap
consequence reaches the detection question itself, and is part of the answer.

**Feature set (per user, over their m >= 2 long texts; axes estimated on TIER-U data
only — no label leakage).** (1) mean close-open 19-vector projected on the PANDORA
slope factors D-comp1/D-comp2 (P8-cache recomputation) and on the A-comp3 motion-cluster
axis; (2) per-construct mean diffs for first_person / tension / directive / novelty;
(3) mean OPENING-window first_person level (position-conditional level); (4) log1p
n_long_texts + coverage indicator; mean-imputation for users without coverage.

**Arms.** C = EXPL-2 union anchor (must reproduce .1361 exactly from its artifact);
D = C + motion features; M = motion features alone. Per-trait RidgeCV (EXPL-2 protocol),
plus a single-feature x trait screen on covered users.

**Registered leans.** Delta(C -> D) mean r in [+.005, +.02] (orthogonality bounds the
add); ONE single-feature surprise |r| >= .10 in Agreeableness or Openness from the
motion set; M alone in [.03, .07]. KILL: Delta <= 0 means the position/slope layer adds
NO exploratory signal at this cap — recorded as the honest v4->v5->V6 detection verdict.

## F15 — W10: the anatomy of the invisible factor (registered 2026-07-12, before run;
## operator focus: "what interests me most is the properties of motion-only factors")

**Known properties (consolidated).** Stable (person-disjoint rep .78-.83); theorem-
invisible to averaging instruments (T1); register-local (T6: .20 cross-corpus; and
Essays hosts NO invisible gust factor — all its gust components are level-visible);
carried by the recurring cluster {wcl_02, wcl_07, wcl_11, wcl_20(, tension, wcl_22,
wcl_35)} — the SAME constructs flagged for anomalous spectral texture (wcl_20 even-lag,
wcl_07 period-4). NEW HYPOTHESIS registered: the invisible factor is a SHARED RHYTHM
component — the cross-construct face of the individual spectral anomalies.

**Instrument (W10, on existing caches; uncapped upgrade after N1).**
(a) ACTIVATION ANATOMY: per-window gust1_P activation (within-comment detrended
residual projected on the axis) correlated with LOADING-EXOGENOUS window markers:
question density, quote-character share, digit share, comma density, novelty_play level
and its window-to-window jump (topic-shift proxy), and window position — what is
textually happening when the factor fires.
(b) PERSON SUSCEPTIBILITY: per Tier-U user with >= 2 m=3 comments: split-half stability
of mean |activation| (susceptibility) and of mean signed activation (directionality).
(c) ESSAYS CONTRAST: same activation machinery on Essays windows via gust1_P — expected
near-chance variance ratio (the factor should barely exist there).

**Registered leans.** (a) activation is question/quote-marker enriched (the
conversational-burst reading of the Reddit-local factor) and topic-shift adjacent
(positive |activation| x novelty-jump); position effect weak. (b) susceptibility
split-half r >= .3 (a stable person property — the first PERSONAL trait-like quantity of
a motion-only object); signed directionality weaker but > 0. (c) Essays activation
variance ratio ~ 1 vs its own shuffle (factor absent). KILL: susceptibility r < .1 would
demote motion-only structure to comment-level texture with NO person anchoring — the
"person x register" reading would lose its person half.

## F15 results (W10, run AFTER registration 52421ea; alignment asserts 43,818/43,818
## and 12,039/12,039 rows; axis drift guard passed {wcl_02 −.47, wcl_11 +.45})

**(a) Activation anatomy — registered lean WRONG in an informative direction.** The
factor does NOT fire on question/quote markers; it fires on EMPHATIC/NUMERIC bursts:
allcaps token share r_centered = .161 (perm p < .002), digits/100 chars = .127
(p < .002); question/quote markers weaker. Reading: gust1 = bursts of shouty/numeric
texture (score-talk, emphasis) coordinated across the style clusters. CAVEAT
(recorded): estimated on the capped m >= 3 stratum (687 texts), which borders the W9
format-dense band — the digit enrichment may be partly stratum texture; the N1 uncapped
re-run adjudicates.

**(b) Person susceptibility — historical exploratory headline, downgraded in the
2026-07-15 V7 audit (this passage's earlier "downgraded 2026-07-12" dating was
backdated; corrected 2026-07-17).** Split-half r of mean |activation| = **.441**
(clears the registered >= .3 lean; ICC-share .381; n = 81 users, capped-corpus-limited).
Signed DIRECTION r = −.128 — my "weaker but positive" lean WRONG: people are stable in
HOW STRONGLY they gust, not in which direction. The original text called this an
"amplitude trait". The V7 full-pipeline audit — now properly recorded as
`docs/V7_W10_AUDIT.md` — found overlapping axis discovery/test,
an incomplete null, non-temporal cid parity, and missing opportunity/format/stranger
controls. The licensed wording is now **descriptive susceptibility candidate** pending
SIM-P3/P5/P6; no trait or person-measurement claim is currently licensed.
KILL (r < .1) not fired.

**(c) Essays contrast.** PANDORA activation variance ratio 1.678 (p < .005) vs Essays
1.035 (p = .005) — direction as registered; Essays carries a whisper, not zero (flagged
rather than rounded down).

**Consolidated portrait of the motion-only factor (operator's question).** A register-
local, averaging-invisible coordination of style clusters that fires in emphatic/numeric
bursts; individuals differ stably in its AMPLITUDE (susceptibility .44) but not its
direction; essentially absent in solitary stream-of-consciousness writing; its carrier
constructs are the same ones bearing anomalous spectral texture (shared-rhythm
hypothesis, still open — uncapped data next).

## F14 — The N-program: building the comparison substrate (registered 2026-07-12,
## operator-ordered "execute the native corpus etc.")

**Mandate.** EXPL-3 showed the comparison substrate itself is missing: the 1,500-char
extraction cap blocks the motion layer for labeled users. Three tracks, all frozen-
object-safe (parallel NEW artifacts only):

**N1 (uncapped parallel re-extraction).** The RAW dump (all_comments_since_2015.csv,
5.3 GB) is uncapped (sampled max 29,189 chars; > 1,500 in 1.1% of comments). Re-extract
Tier-L and Tier-U comment sets with the v2 rules MINUS the cap (sanity cap 30,000 chars,
documented) into NEW files tier_{l,u}_comments_uncapped_v1.parquet. Frozen v2 artifacts
untouched. Diagnostics: per-tier m-distribution; texture-capable coverage (m >= 3 / 5 /
7) among gated opening-#1 users. LEANS: >= 60% of gated users gain >= 1 text with
m >= 5; >= 30% gain >= 3 such texts.

**EXPL-4a (Essays dev half; the motion-layer x questionnaire test the cap denied us).**
Single-essay motion features (slope vector projections + q-contrast gust magnitudes +
where m >= 5 crude spectral values) vs Big5 on the DEV half only — the split rule read
from the seal documents; confirm-half labels remain UNTOUCHED (opening budget 1 intact);
exploratory banner. LEANS: any single motion feature |r| >= .08 in <= 3 of 50 cells;
incremental over a level-feature baseline in [+.00, +.02]; honest coin-flip overall.

**EXPL-4b (after N1): uncapped Tier-L motion features x spent opening-#1 labels** under
the EXPL-3 triple-assert governance, now WITH the spectral layer where per-user data
permits. LEAN: Delta over EXPL-3's D in [+.000, +.015]. STANDING KILL: if 4a and 4b both
land <= 0, the conclusion "questionnaire-criterion insensitivity to the motion layer"
becomes the recorded verdict pending BEHAVIORAL criteria (the native corpus's job).

**N4 (native corpus collection instrument, OP-36 execution to the autonomy boundary).**
Deliver the ready-to-run collection pack per SUICA_NATIVE_CORPUS_DESIGN_V1: bilingual
prompt battery (Arm A free / Arm B fixed + AI-interview script per Rulebook G), session
protocol (2 x 2, >= 2 weeks), assistant-echo monitor spec, data schema, anonymization
plan. Participant recruitment and IRB are operator-side by nature and are the ONLY
missing pieces after this deliverable.

## F13 results (EXPL-3, run AFTER registration 36293aa; all governance gates passed
## before fitting: gate n=1,058, H2/H6 to <1e-9, EXPL-2 anchor .1361 exact)

**Numbers.** Coverage 787/1,058 users (74.4%; median 4 long texts). Arms: C = .1361
(anchor), D = C + 10 motion features = **.1370 (Delta +0.0009)**, M = motion alone =
.0300 (Openness-carried, +.127). Single-feature screen 10x5: **zero |r| >= .10 hits**;
closest = open_first_person -> Neuroticism +.096.

**Adjudication.** Lean 1 MISSED LOW (+.0009 vs the [+.005, +.02] band; the Delta <= 0
kill does not fire by letter, but the practical verdict is the kill's spirit: NO
MEANINGFUL ADD). Lean 2 FAILED (no single-feature hit; the near-miss is in the wrong
trait). Lean 3 at the band's lower edge (.0300). Mechanistic reading: the one
label-relevant position feature (open_first_person -> N +.096) is plausibly the H2
signal (first_person -> N +.111) REPACKAGED through the position lens — its content was
already in arm C. This is theory-consistent: components with level shadows were already
captured; the level-orthogonal motion components carry no measured trait signal here.

**The v4 -> v5 -> V6 exploratory detection ledger (spent labels, weight-fit tier):
.0902 -> .1361 (+51%) -> .1370 (+0.7%).** Confirmed tier unchanged throughout (2/7 at
opening #1; no new seal). CRITICAL SCOPE NOTE: the 1,500-char cap makes the V6 SPECTRAL
layer (r_c, rho_pi, shape pair) uncomputable for labeled users — EXPL-3 measured only
the position/slope layer. The honest summary: V6's contribution to DETECTION on PANDORA
is ~nil; V6's contribution is new measurement objects (motion styles, frames, texture)
whose trait relevance is untestable on this corpus and awaits the native corpus and a
new seal.

## F12.10 results (W9, run AFTER registration e2b2240; anchors bit-exact twice —
## cache dynamics = W2a to 0.0e+00, raw rebuild = P8 cache to 0.0e+00)

**Lean (b) FAILED INVERTED; lean (c)'s KILL FIRED ON THE LETTER; and the executing
agent's provenance dig found the true mechanism, which the registered taxonomy missed.**
The registered stripper (quotes/lists/code) was a no-op: 0 of 72 long texts contain even
one such line; stripped dynamics identical (excess −.1199 vs −.1194; saturations 7 = 7).
The registered format hypothesis is dead AS TAXONOMIZED.

**The actual mechanism — CAP-ARTIFACT STRATUM (provenance evidence, all numeric).**
(i) Tier-U bodies are hard-truncated at 1,500 characters at extraction
(extract_suica_tier_u_comments_v2.py:67); 21,733 of 35,181 candidates sit at exactly
1,500. (ii) Normal prose runs ~4.8 chars/token, so m ≥ 4 (512+ tokens) within 1,500
chars is DEFINITIONALLY IMPOSSIBLE FOR PROSE; the cohort's chars/token median is 2.48
(min 1.02) vs 4.79 in the m = 2–3 stratum. (iii) The stratum is populated by
punctuation-dense formats: markdown pipe TABLES (pipe-line share > .5 in 14/72 texts,
mean share .194 — column-periodic, a textbook Δ²-saturation generator) and other
dense-format objects. VERDICT: the PANDORA m ≥ 4 stratum is an extraction-cap artifact,
not a register: **all PANDORA within-text dynamics claims (the W2a PANDORA arm, E3's
PANDORA descriptive) are retroactively marked ARTIFACT-STRATUM — neither psychology nor
"register texture"**. Standing rule: PANDORA within-text dynamics are OUT OF SCOPE under
the current frozen extraction (re-extraction would touch frozen prep — not done); the
native corpus carries the dynamics program.

**Additional fragility caught by the anchor:** gust1_P's PANDORA θ̂ swings 0.02 ↔ 0.48
purely on the pooled-sd convention (per-arm vs full-cache) at n = 69 — dynamics
conclusions on that axis were standardization-dependent, consistent with the
artifact-stratum verdict.

**Scorecard.** (a) anchors ✓✓ bit-exact; (b) inverted (formatting ABSENT from the long
stratum — because the long stratum is not prose at all); (c) kill fired as registered,
and the mechanism FAMILY (format periodicity) is vindicated under the corrected
taxonomy (tables + cap), discovered by the agent's Reviewer-style provenance dig — the
program's clearest case of the executing agent out-reasoning the registration.

## F12.9 results (W8, run AFTER registration e04d5e0; grid 6,564 triangle points,
## five integrity asserts passed incl. white and AR(1) grid rows to 1e-9/1e-6)

**Lean (b) — the family's raison d'etre: CONFIRMED with CI-solid signs.** wcl_07
a2 = −0.16, CI [−0.28, −0.02] (damped period-4, squarely in the registered band; a1 ~ 0
as sketched). wcl_20 a2 = +0.30, CI [+0.04, +0.32] solid positive — but with an
UNANTICIPATED large a1 = +0.68 (optimum at a1+a2 = 0.98, near the stationarity
boundary): near-integrated persistence PLUS even structure, not the "interleaved" pure
a1 ~ 0 texture the lean sketched (sketch corrected, sign conclusion stands).

**Lean (d) — HOLDS emphatically and carries the MA class.** The two one-step-echo
constructs are the WORST AR(2) fits of all 19 (wcl_03 resid 0.830, first_person 0.524
vs carry-over set 0.067–0.154) — exactly what "MA(1) is not nested" predicts, delivered
where the instruments have power.

**Lean (c) FAILED as registered; the KILL fires by letter and is ruled UNDECIDABLE by
mechanism (recorded in full).** wcl_60 ranks 8/19 (ratio to median 1.18). But its
functional sds are 7–13x the field (gamma0 = 0.22 denominator; 29/300 draws dropped) —
the weighted overidentification test has NO POWER for wcl_60 (its resid CI spans
[0.003, 1.76]; under wcl_03's sds its deviations would rank worst by an order of
magnitude). Ruling: the kill predicate presupposed comparable test power, which fails;
the kill's TARGET claim ("the MA-vs-AR distinction was noise") is separately REFUTED by
lean (d). Consequence: wcl_60's PARAMETRIC identity (MA vs AR) is downgraded to
UNDECIDED-LOW-POWER; only its sign facts stand (anti-persistent lag-1, rho_pi > 1
CI-solid). Honest note: this is the program's first letter-fired kill set aside by an
explicit power argument — the reasoning, sd table, and both readings are in the JSON.

**F12.9 verdict — the unification, amended.** Gust texture = AR(2) STATIONARITY
TRIANGLE plus a NON-NESTED MA-ECHO CLASS: carry-over cluster (a1 > 0, a2 ~ 0), even
structure at both signs of a2 (wcl_20 +, wcl_07 −, both CI-solid), echoes
AR(2)-unrepresentable (worst residuals), wcl_60 bounce-region placement power-limited.
Model (M''') = level + flow + AR(2)-textured gusts (+) MA-echo exceptions — four ad-hoc
textures reduced to one two-parameter family plus one discrete class. Promoted to
THEORY V6 as T7.

## F12.8 results (W7a Essays + W7b deployment, run AFTER registration eabc17b)

**Conditioning: the registered hybrid design HELD with no fallback** (|w_pi| = 4.10,
|w_pihalf,5| = 7.05 < 20, |w_gamma0,trunc| = 5.77 on Essays; deployment m = 8: 2.09 /
4.27 / 4.07-vs-39.1 old). Two exact invariants asserted at runtime on the actual
composition (white -> (1,1,0); MA(1) -> rho_pihalf = 1) — both passed to 1e-9.

**Adjudication.** (a) Carry-over set Delta_shape > 0: 4/5 directional (wcl_36 +.094,
wcl_13 +.094, wcl_35 +.065, first_person +.048; wcl_03 the exception at −.030), none
CI-solid. (b) **wcl_20's even-lag reading CONFIRMED in the cleanest number:
rho_pihalf = 0.645 [0.428, 0.914] — CI-solid depressed**; ordering P(Delta[wcl_20] <
Delta[wcl_35]) = .912 (strong lean, short of solid), wcl_22 weaker (.756); tension shows
the same shape (Delta −.159). (c) wcl_60: rho_pihalf = 1.962 [0.898, 5.846] — CI
includes 1, the pure-MA(1) invariant SURVIVES (unstable denominator gamma0 = .22 noted;
point far from 1, undecided in the strong sense). KILL not fired.

**Unregistered finding (third consecutive poke by this family): wcl_07 is the ONLY
CI-solid Delta_shape, POSITIVE +0.540 [+0.092, +1.245]** (rho_pihalf = 1.478) —
elevated quarter-frequency with sub-white Nyquist implies gamma2 < 0 alongside positive
gamma1: a damped ~4-window (~512-token) oscillation. Post-hoc reading (labeled as such):
AR(2)-even with NEGATIVE a2 fits almost exactly ((1−a2)/(1+a2) = 1.5 at a2 = −0.2 vs
observed 1.478). Together with wcl_20 (a2 > 0, period-2) this suggests the UNIFYING
two-parameter family — signed odd-lag memory (theta/phi) x signed even-lag oscillation
(a2) — covering every texture seen so far: carry-over, bounce, period-2, period-4.
QUEUED AS W8 (to be registered before any run): joint signed-(memory, a2) fit per
construct with the shape pair + rho_ladder moments; leans to be written at registration.

**Deployment (W7b).** Hybrid denominator + shape pair shipped
("spectral_shape_by_construct", "delta_shape_by_construct", "period2_denominator",
"shape_mode"); AR(2) m = 8 calibration recovered 1.41 -> 1.75 (asymptotic 2.33; residual
gap = numerator truncation of gamma6, documented); noise roughly halved by the hybrid;
72/72 tests.

## F12.7 results (W6a, run AFTER registration ed971f1)

**Conditioning postscript (premise half-failed, registered fallback absorbed it).** The
"well-conditioned left-solve" premise holds for f_π (|w| = 4.10) but NOT for the
5-moment γ0 functional (|w| = 70.4 — it projects 0.61 onto the centering near-null
direction; wcl_60/wcl_07 got negative point γ0 and 300+ dropped draws). The registered
lag-2-truncated fallback (|w_γ0| = 5.77, cond(M3) = 110, all 19 γ0 positive) is the
HEADLINE mode; 5-moment reported for transparency. Estimable-functional discipline now
reads: check the LEFT-solve norm per functional, not per instrument.

**Adjudication (truncated headline mode, mode-corrected reference values printed).**
(a) wcl_60 ρ_π = 2.513, CI [1.280, 7.354] — **CI-SOLID ABOVE 1; the registered ranking
INVERSION happened exactly** (lowest raw f_π → highest normalized ρ_π); KILL not fired;
point above the predicted 1.4–1.7 band (wide upper CI, 40 small-γ0 draws dropped).
(b) Carry-over set below 1: wcl_35 .752 [.607, .970], wcl_36 .768 [.647, .930], wcl_13
.807 [.689, .959] CI-solid; first_person .841 [.714, 1.015] and wcl_03 .863 [.717,
1.058] directional — 3/5 solid, 2/5 grazing.
(c) **FAILED INFORMATIVELY: wcl_20 ρ_π = 0.770 [.645, .967] — CI-solid BELOW one, the
OPPOSITE side**; tension .904 spans 1. wcl_20 carries the largest absolute f_π AND the
largest γ0: its W4 lag-2 excess coexists with lag-1 carry-over that pulls net Nyquist
energy sub-white. VERDICT: the "period-2 standing wave" reading of wcl_20/22 is DEAD at
the net-energy level; what survives is an EVEN-LAG ANOMALY inside a carry-over-dominated
spectrum (W4's R2 remains a real model-misfit signal). The separator is spectral SHAPE:
f(π/2) = γ0 − 2γ2 + 2γ4 should be DEPRESSED for wcl_20/22 (positive γ2) — queued as W7.
Unregistered: wcl_23 is the only other point > 1 (1.110 [.845, 1.622]), consistent with
its W3 second-bouncer rank. Corpus mass sits at ρ_π ≈ .75–1.0 — the faint carry-over
again, now in spectral units.

## F12.6 results (W5, run AFTER registration 6669ccc; engine cross-checks to 9e-16 on
## the W3/W4 blocks; truncation aliasing quantified pre-run — f_π is the most
## aliasing-robust output, and damped alternation can flip g3's sign, both recorded)

**The registered indeterminacy scenario (lean d) FIRED — and it is structural, not
sampling.** SVD of the exact centered 5×5 map: singular values [14.9, 10.1, 4.05, 0.81,
**0.0087**]; the near-null right vector [0.61, 0.57, 0.45, 0.28, 0.12] is a smooth
"raise the whole ladder" direction that within-text centering renders nearly invisible.
Consequently the individual γ1..γ4 coordinates are unresolvable at this n for every
construct (all point ladders are near-proportional to the null vector; all CIs span 0),
the classification rule fired vacuously (19/19 "MA1" — a rule-design flaw of the
registration, recorded as such), lean (a) is unadjudicable (wcl_20/22 g4/g2 points sit
in the registered band but their CIs span everything), and the KILL did not fire.

**What IS resolved — the spectral functional.** The Nyquist energy f(π) = γ0 − 2γ1 +
2γ2 − 2γ3 + 2γ4 lies away from the null direction and is tightly estimated for all 19
constructs. Its ordering CONFIRMS the W4 period-2 flag at the functional level, with
disjoint CIs at the extremes: wcl_20 f_π = 1.033 [0.833, 1.225] and tension = 1.016
[0.806, 1.260] highest — exactly the two flagged constructs — versus wcl_60 = 0.557
[0.271, 0.965] and first_person = 0.628 [0.515, 0.744] lowest (wcl_20 vs first_person:
non-overlapping CIs).

**F12.6.3 (estimable-functional principle; theorem-grade lesson).** Under within-text
centering, gust dynamics beyond lag ~2 are ill-posed in LAG COORDINATES at feasible n
(the centering null direction); the estimable objects are WELL-CONDITIONED SPECTRAL
FUNCTIONALS of the gust spectrum (f at Nyquist and similarly placed frequencies). The
right parameterization of the motion layer's dynamic texture is frequency-domain — (r_c,
f_π/γ̂0-normalized) — not the autocovariance ladder. This is the deconvolution-style
ill-posedness → estimable-functional move, and it aligns the program with why
realized-covariance theory works with integrated spectral quantities. Deployment
increment (queued W6): expose a normalized period-2 energy in motion.py alongside the
signed memory coefficient.

## F12.5 results (W4, run AFTER registration 5064a0a; enumeration engine reproduced the
## W3 coefficient set incl. d(3)=56/9 to ≤ 9e−16; planted-recovery checks passed at
## MA θ=.4, AR φ∈{.15,.5}; centering attenuates the lag-2 map to (0.708, −3.480) from
## the uncentered (1, −4) — the enumeration mandate vindicated again)

**Scorecard.** (a) Corpus diag-mean R2 = +0.0210, frac>0 = .942, CI [−0.0052, +0.0461]
— direction as leaned, CI NOT solid: the corpus-level decaying-carry-over claim stays
directional. (b) Memory-set classification 3/5: wcl_35/wcl_36/wcl_13 AR-win with φ̂ =
+0.100/+0.095/+0.080 — cross-moment CONSISTENCY with their W3 lag-1 r_c (+0.098/+0.092/
+0.079); first_person and wcl_03 flip to MA (θ̂ = −0.085/−0.070): still positive
one-step memory, but the carry-over does NOT extend to lag 2 — a refined taxonomy:
DECAYING carry-over (wcl_35/36/13) vs ONE-STEP echo (first_person, wcl_03). Per-construct
winner calls are suggestive (all R2 CIs span 0 for these). (c) wcl_60 MA-type CONFIRMED
(second-largest margin; θ̂ = +0.235). Neither registered kill fires.

**Unregistered discovery — the period-2 signature.** The only CI-SOLID R2 constructs are
NOT the memory set: wcl_20 (R2 = +0.175, CI [+0.064, +0.283]) and wcl_22 (+0.119,
[+0.040, +0.197]) — and for both, BOTH one-parameter families fail (RSS 2–3 orders above
the well-fit constructs; fitted θ, φ ≈ 0). Weak lag-1 with large positive lag-2 is
producible by neither MA(1) nor AR(1) (AR needs γ2 = φ²γ0 bounded by the lag-1 level):
an EVEN-LAG / PERIOD-2 gust signature on Essays (tension borderline shows the same
shape, +0.085 CI [−0.013, +0.194]) — structurally reminiscent of the PANDORA
format-periodicity flag, now appearing register-internally for a construct subset.
Registered follow-up: lag-3 identity (the Δ⁴ filter at h = 3) separates period-2
oscillation (γ3 < 0 expected) from MA(2) (γ3 = 0); this is the W5 instrument. The F12.5.2
"outside both families" branch fired in the UNANTICIPATED R2 > 0 form — recorded.

## F12.4 results (W3 + bootstrap, run AFTER registration 0a4a97a; Sonnet-executed with
## independent verification — analytic re-derivation, 4M-text Monte Carlo, and an
## out-of-sample check at θ = .6, m = 10 chosen by the agent)

**The registered point lean was KILLED as written — and the sub-claim survived in a
stronger form.** Corrected inversion on Essays (1,349 texts, coefficients from the
actual n-composition: A = 5.7836, B = −7.7836, C = −3.9348, D = 6.7345 (exact d(3))):

- Corrected B̂ diag mean = **+0.0303, bootstrap CI [+0.0136, +0.0484], 500/500 draws
  > 0** — not ≈ 0 as leaned. The naive mean (−0.0103) has CI [−0.0248, +0.0054]:
  spans zero AND contains the F12.4-predicted pure-centering value (−0.0178·Γ̂0 ≈
  −0.013) — the bias formula is quantitatively confirmed on real data. Verdict: the
  naive negative WAS centering artifact (sub-claim right), but what it was hiding is
  not zero — it is REAL, MILD, POSITIVE gust memory. "White gusts" (v6.1) is revised.

- Construct-level convergence with W2a: positive with CI excluding 0 — wcl_35 +.098,
  wcl_36 +.092, first_person +.083, wcl_13 +.079, wcl_03 +.070 (the same
  content-flavored carry-over set W2a flagged on the less-negative side of its null);
  negative — wcl_60 only (r_c = −.308, CI [−.873, −.035]: SIGN solid, magnitude tail
  beyond MA(1) identifiability |r| ≤ .5, so θ-conversion of the CI is not licensed);
  13 of 19 span zero. **The two leverage-free instruments now agree
  construct-by-construct.** Per-construct registered predictions all held (tension
  .070 → .017, wcl_54 .057 → .004, wcl_23 .103 → .048, wcl_60 stays ≥ .25 at .34).

- Model revision (M″): signed-memory MA(1), θ ∈ (−1, 1] — every F12 identity is
  polynomial in θ and holds unchanged; the operational primary output becomes the
  SIGNED memory coefficient r_c = B̂_cc/Γ̂0_cc ∈ [−.5, .5) (positive = carry-over,
  negative = bounce), with θ reported only where meaningful. MA-vs-AR discrimination
  needs lag-2 moments (S2 identities) — registered as the next identifiability step.

## F12 results (W2a/W2b, run AFTER registration da07462; Sonnet-executed under Fable
## specs; W2b agent validated the inversions on synthetic MA(1) at six θ values)

**W2a — leverage-free Δ² dynamics.**
Essays primary arm (1,349 texts 5 ≤ m ≤ 12, 4,746 pairs; iid simulation null mean
−0.6487, band [−.663, −.634]):
- **gust axes are WHITE: gust1_E ρ1 = −0.6402, p = .871 (θ̂ = 0); gust1_P −0.6581,
  p = .095 (θ̂ = .036). The registered kill fired against E3: its anti-persistence
  (−0.31 vs pairing null −0.19) was the OLS-leverage artifact. E3's bounce claim is
  STRUCK.**
- The one certified bouncer: **wcl_60, ρ1 = −0.6711, p = .0005, θ̂ = .091** (sensitivity
  arm p = .034) — the zero-inflated apostrophe construct. Bounce = burst-recovery of
  sparse events. This UNIFIES wcl_60's three anomalies (F8.3 asymptote violation, the v5
  hurdle regime, θ > 0) under one mechanism: sparse event bursts.
- Systematic MILD POSITIVE persistence (ρ1 significantly LESS negative than null;
  one-sided p ≥ .97) for content-flavored constructs: first_person (.9955), wcl_36
  (.997), wcl_07 (.998), wcl_13/35/15 (~.996-.998), wcl_02/03 (.98), novelty (.971) —
  topic carry-over, |Δρ1| ≈ .015-.024. The gust field is white on average with a
  textured split: sparse constructs bounce, content constructs carry over slightly.
- PANDORA m ≥ 5 (29 texts, 95 pairs; wide null [−.74, −.53]): axes −0.797/−0.805
  (p < .0005) and several constructs SATURATE the MA(1) boundary (wcl_02 −0.846; in the
  m ≥ 4 arm wcl_36 −0.995, wcl_11 −0.976, novelty −0.967) while wcl_11 FLIPS from −0.203
  (m 5-12) to −0.976 (m 4-12) — arm-flipping plus ρ1 ≈ −1 signals near-deterministic
  window ALTERNATION, i.e., FORMAT PERIODICITY (quote/reply and list structure in long
  Reddit comments), not an MA(1) gust field. Register-local texture (consistent with
  T6); psychology NOT claimable; format-aware windowing = new open instrument.

**W2b — corrected separation + bounce factor (Essays, 1,349 texts, 8,793 windows).**
- **B̂ ≈ 0**: diag mean −0.0103 against Γ̂0 diag ≈ 0.73 (implied mean θ ≈ .014);
  per-construct implied θ nonzero for 10/19, led by wcl_60 = .43 (diag method — cruder
  than W2a's .09, same unique flag), wcl_23 = .10, tension = .07.
- **Bounce factor: DOES NOT EXIST** — λ1(−B̂) = .175 with permutation p = .98 (observed
  SMALLER than 98% of order-permutation draws), half-replication .108, congruence with
  Γ̂0 = .026. Registered lean (c) wrong on every gate — coherent with W2a: no common
  bounce structure because the average memory is zero.
- Corrected flow: λ1(Σ̂_flow) = .0307 (comparator without the 2Γ̂0/(m−1)² subtraction:
  .1036 — the drop is removal of gust inflation, not of B contamination, since B̂ ≈ 0),
  half-replication .308, bootstrap point-below-CI bias noted by the executing agent —
  **flow remains UNCERTIFIED on Essays** with the correct machinery.
- P9 status: the F12.2 downgrade STANDS (it is a theorem). B̂ ≈ 0 was measured on Essays
  and by T6 must not be transported to Reddit; PANDORA's own long-text dynamics are
  non-white and format-suspect — so the P9 flow factor stays CONDITIONAL on iid gusts,
  now with the added note that the one measurable register shows B ≈ 0.

**F12.3 scorecard.** (a) WRONG as leaned — axes white; the designed alternative (strike
E3) is what fired; wcl_60 the exception at θ̂ .09 (below the leaned [.1,.5] band edge).
(b) "rises" WRONG (premise of large negative B false); "replication < .5" RIGHT (.308).
(c) WRONG on all gates. (d) stands. Net theory movement: one wrong claim deleted (E3
bounce), three anomalies unified (wcl_60), the gust model cleaned to
white-with-textured-exceptions, and the flow estimator's audit machinery (exact
inversions) now standing equipment.

## F11 results (V6-E1..E4, run AFTER registration commit 8e2b0ba, same day)

**E1 — gust robustness (PANDORA m=3, n=618; gust1 recomputed, leads wcl_02/wcl_11).**
Temporal median split: congruence .956 / .971, cross .864 — TIME-ROBUST. User-demeaning:
.982 — person effects irrelevant (as registered: within-comment object). Scale 64 tokens:
.622 — persists attenuated (upper edge of the registered .4–.6 band). Venue classes: the
two classes with n ≥ 140 replicate at .926 / .896; the two with n ≤ 75 fail (.173 / .147)
— POWER vs true heterogeneity UNRESOLVED at this n (recorded; the registered ≥ .6 lean is
therefore only partly right).

**E2 — Essays transport (2,309 essays m ≥ 2, median m = 5, 12,039 windows; loader reads
user_id/text ONLY).**
(a) Position: first_person lin −1.058 (p < .0005) TRANSPORTS, and gains significant
negative curvature on long essays (−0.447, p < .0005 — fast early drop); tension closing
rise present but attenuated (+0.078, p = .0005) — the registered "attenuates or vanishes"
was half right; directive SIGN-FLIPS (+0.142 close-elevated on Essays vs opening-elevated
in the PANDORA m=3 stratum) — genre-dependent, consistent with the earlier
stratum-dependence flag; wcl_22 curvature also flips sign across corpora.
(b) Law-of-the-wall: composite CV gain ≈ 0 or negative for 12/13 highlight constructs —
registered lean WRONG. Position structure is real (a) but explains a SMALL share of
within-essay window variance; single-coordinate models suffice at current SNR; the
matched-asymptotics reading stays a metaphor, not a fitted necessity.
(c) Gust structure exists on Essays (gust1_E λ = 1.598, replication .933) but is
statically VISIBLE (2.236) and first_person-led — no statically invisible gust factor
found on Essays.
(d) **HEADLINE BET FAILED AS REGISTERED-KILLABLE: gust1_E × gust1_P = .200 (random
baseline ≈ .23) — the PANDORA gust factor is REGISTER-SPECIFIC, not corpus-general.**
(e) Essay-level flow-only (invisible) factor: NOT found (lean WRONG) — instead the
statically VISIBLE motion factors TRANSPORT ACROSS CORPORA: D-comp1_E × D-comp1_P = .940,
D-comp2 × best = .782; static frames rank-1 congruence .923. Flow certification fails on
Essays (λ₁(Σ̂) = .109 below its own bootstrap band, replication .098).

**E3 — gust dynamics (Essays m ≥ 4, 1,805 texts; PANDORA m ≥ 4 descriptive).**
AR(1) of gust projections: −0.314 (gust1_E) / −0.339 (gust1_P) vs within-text pairing
null [−0.21, −0.17], p < .002 — gusts are ANTI-PERSISTENT beyond the detrending baseline:
overshoot-then-compensate at adjacent windows (MA(1)-like bounce; the microstructure
analogy sharpens — negative lag-1 autocorrelation is the bid-ask-bounce signature).
Registered "white" lean WRONG in an informative direction. CAVEAT: OLS-detrending
leverage geometry contributes to adjacency structure and this null does not fully remove
it — flagged; a leverage-free redesign (second-difference prewhitening) is the follow-up.
The position-magnitude arm is INCONCLUSIVE BY DESIGN FLAW (endpoint leverage mechanically
shrinks endpoint residuals; observed edge−interior −0.22 is confounded) — recorded, not
interpreted.

**E4 — person-level retry (PANDORA).** REGISTERED LEAN CONFIRMED: the F10.8 phase-C death
was a power artifact. Gate ≥ 5 residualized (static congruence ≤ .03 by construction):
comp2 λ = 1.641, half-replication .777; comps 3/4/6 replicate .60–.64. Gate ≥ 8
residualized comp1: λ = 2.137, replication .784, loadings {wcl_35 +.45, wcl_22 −.42,
wcl_07 +.41, tension +.40} — the recurring motion cluster, now established as a
PERSON-LEVEL, statically invisible, replicating individual-difference dimension.

**F11 synthesis (the V6 empirical laws).**
1. LAYERED MOTION: level Π / drift Σ_flow / gust Γ each carry factor structure.
2. **PORTABILITY FOLLOWS VISIBILITY** (new empirical law, falsifiable): motion factors
   WITH level shadows transport across corpora (.94/.78); statically INVISIBLE structure
   is register-local (gust1: robust within Reddit — time .96, users .98, scale .62, large
   venues .9 — yet .20 across corpora). Math similarity: integrated covariance
   (fundamental, portable) vs microstructure noise (market-specific) in
   realized-covariance theory. Falsifier: an invisible factor that transports.
3. GUSTS BOUNCE: anti-persistent adjacent-window dynamics (with the leverage caveat).
4. PERSON MOTION-STYLE EXISTS: statically invisible, replicating at .6–.78 — a new
   individual-difference layer, readable only in motion (occasions-cheap, tokens-hungry
   per the F11.5 economics).
5. POSITION IS REAL BUT SMALL at window granularity, and genre-bends (directive/wcl_22
   sign flips) — position-conditional norms must be corpus-anchored.

The V6 document (SUICA_THEORY_V6.md) carries the full synthesis, the mathematical
similarity table, and the personality-inference extension.

## F14 results (N1, uncapped re-extraction; run same day as registration 97ecca2;
## adjudication was never closed out — closing it now, before F16 depends on it)

**Numbers.** Runtime 62s. tier_l: 385,275 rows / 1,412 users (7,039 bodies differ from
the frozen v2 capped parquet, 1.83%; new max length 9,999 vs the 1,500 cap). tier_u:
1,540,040 rows / 5,000 users (21,696 differ, 1.41%; max 16,417). Long-text census
(bodies >= 1,200 chars): tier_l m>=3 4,400 texts / m>=5 1,240 / m>=7 502; at m>=4 the
chars/token median is 4.81 — real prose, clear of the W9 cap-artifact band (2.48).

Coverage among the 1,058 opening-#1 gated users (the EXPL-3/4b/F16 population): m>=3 in
678/1058 (64.1%, median 3 texts); m>=5 in 352/1058 (33.3%, median 2 texts); >=3 texts at
m>=5 in 118/1058 (11.2%).

**Adjudication against the registered leans (>=60% at m>=5; >=30% with >=3 such
texts).** BOTH MISSED LOW: 33.3% vs 60%, 11.2% vs 30% — roughly half the hoped-for
texture-capable coverage materialized. N1 carries no kill condition of its own (it is
infrastructure, not a claim), so this is not a failed hypothesis — but it sets the real
power ceiling for every downstream uncapped-motion instrument (EXPL-4b, F16): about a
third of labeled users get any spectral read at all, about a ninth get a multi-text
within-person read. Downstream instruments must report coverage-conditional numbers
alongside pooled ones, not pooled-only.

## F16 — Visibility taxonomy: static-only / dynamic-only / both, against every existing
## external anchor (registered 2026-07-12, operator-ordered: "静的しか見れない因子と動的
## しか見れない因子，および動静両方見える因子...全部探し出し，既存の外的尺度と相関を
## 調べよう")

**Mandate (retrospectively narrowed by the V7 audit).** F10.8/W10 produced one
motion-only, same-sample person-associated candidate (the hand-built gust1_P
susceptibility composite); it did not establish a stable person parameter. The
operator's directive generalizes the channel screen from one composite to the full
construct space and compares each frozen construct with available external anchors.
This is an exhaustive exploratory sweep of the static/dynamic/hybrid space, not a
personality-signal certification procedure.

**Object.** For each of the 19 frozen v4-battery constructs c (read at run time from the
frozen scorer — project_persona/suica.py / melon.py's ScoreSheet — never hardcoded),
define two channel scores per person u:
- STATIC S_c(u): person mean of c across all scored windows. Bit-identical to the
  existing frozen battery's own construct means (score_slices_v2 output) — no new
  estimator.
- DYNAMIC D_c(u), PRIMARY: per-construct gust amplitude / susceptibility — W10's
  activation measure (within-text detrended residual on a construct's own axis)
  generalized from the one hand-built gust1_P composite to EACH construct's own axis
  individually; person score = mean |activation|, m>=3 windows required (W10's
  estimator, reused, not reinvented). SECONDARY (exploratory-of-exploratory; reported,
  not load-bearing for classification): signed memory (persistence) and rho_pi (Nyquist
  ratio) from suica_core/motion.py, each m-gated per its own estimator requirement.

Before any fitting, the executing agent must verify (not assume) that capped vs.
uncapped extraction does not itself move STATIC scores: if capped/uncapped person means
differ systematically by >|.02| for any construct, flag and report before proceeding —
level is expected to be cap-insensitive (unlike spectral texture, per W9), but this is
an empirical check, not a premise.

**External anchors — three tiers of governance, none conflated.**
(i) **PANDORA official Big5, opening-#1 gate (n=1,058, predictors.parquet) — SPENT
label tier.** EXPLORATORY banner, ledger row, mandatory gate-replication assert
(n==1,058) AND H2 (first_person -> Neuroticism) AND H6 (politics_choice -> Openness)
recomputed to <1e-9 against OPENING_REPORT.json BEFORE any fitting — the EXPL-1/EXPL-3
assert block, copied verbatim, not reimplemented.
(ii) **PANDORA official MBTI axes (EI/SN/TF/JP cont; data_sets/prepared/pandora_
official/mbti_axes/*.csv, ~9,042 users) — FIRST correlational use in the GRAIN
construct-vs-anchor sense.** These labels were used to TRAIN the official MBTI LoRA CV
line (a separate, model-training spend) but have never been fit or correlated against
GRAIN constructs before; there is no prior H2/H6-equivalent to replicate. Governed by
the same EXPLORATORY banner + ledger-row discipline as (i), WITHOUT a replication
assert (none exists yet to reproduce) — and this run ITSELF becomes the reference point
any future reuse must replicate. Record explicitly: this run SPENDS the MBTI axis
labels for future SUICA-correlational (not LoRA-training) work.
(iii) **Essays Big5, DEV HALF ONLY** (essays_regime_dev_half.csv, 1,255 users), via the
verified two-pass parse-time exclusion mechanism (`two_pass_dev_labels`, reused verbatim
from EXPL-4a); CONFIRM half remains untouched (opening budget = 1 remaining, unaffected
by this run).

Dynamic scores use N1's uncapped tier_l/tier_u parquets where a user's population
requires PANDORA text (Big5 gate and MBTI axes), and the existing uncapped Essays cache
(already built for EXPL-4a) for (iii).

**Classification rule (per construct x anchor cell).** BH-FDR correction applied
SEPARATELY within the static family (19 constructs x K anchor traits) and the dynamic
family (same N cells) — q<.05, plus an effect floor |r|>=.08 (the EXPL-3/4a convention)
for a cell to count as signal.
- STATIC-ONLY: static cell passes, dynamic cell (same construct x anchor) does not.
- DYNAMIC-ONLY: dynamic cell passes, static does not.
- BOTH: both pass.
- NEITHER: neither passes.
Report the FULL anchor x construct grid (19 x up to 14 anchor traits x 2 channels) —
no collapsing to a single modal label before the raw grid is shown.

**Leans (directional, registered before run).**
(a) Most constructs land STATIC-ONLY or NEITHER — the v4 battery was built and
validated on static/level correlations (H2 IS a static-level replication), so this is
the null-consistent expectation, not a finding.
(b) 1-4 constructs land DYNAMIC-ONLY on at least one anchor; leaned to be among the W10
gust1_P carrier constructs (its loading-drift-guard set: wcl_02/wcl_11/wcl_07/wcl_20)
rather than a fresh unrelated construct.
(c) BOTH (hybrid) is rare: 0-2 constructs, given EXPL-3 (+0.0009) and EXPL-4a (-0.0096)
already show static+dynamic stacking adds little at the composite level.
(d) **STANDING KILL:** if ZERO (construct, anchor) cells classify DYNAMIC-ONLY anywhere
in the full grid, W10's headline finding demotes from "a general property of the
construct space" to "true only of the one hand-built gust1_P composite" — this must be
recorded explicitly as a demotion, not quietly re-described as consistent with W10.

**Governance checklist (all must hold before results are trusted):** frozen scorer used
unmodified for S_c; suica_core/motion.py used unmodified for D_c; PANDORA gate n=1,058
+ H2 + H6 replicated <1e-9 before any Big5 fitting; MBTI-axis run explicitly marked as
first-use/label-spending; Essays confirm-half bytes never parsed (two-pass mechanism,
tested on synthetic data first, exactly as EXPL-4a); BH correction applied per-family as
specified, never pooled across families; EXPLORATORY banner on every output; new ledger
row; N1's uncapped parquets and the frozen v2/Essays caches are read-only throughout.

## F16 results (run AFTER registration 32befcf; independently adversarially verified,
## overall verdict TRUST — governance_ok/stats_ok both true, zero discrepancies found)

**Governance, reproduced independently.** Gate n=1,058 exact; H2=0.1113549036 /
H6=0.0959130343 both matched OPENING_REPORT.json to <1e-9 (recomputed twice, once by
the build agent, once from scratch by the verifier, diffs 1.4e-17). Essays two-pass
mechanism is the actual imported EXPL-4a function, not a reimplementation; confirm-half
bytes never parsed. MBTI-axis run explicitly banner-marked as first-use/label-spending.
BH-FDR applied as two independent 266-cell families (static, dynamic; not pooled, not
per-anchor-set) — the verifier reimplemented BH-FDR from scratch and reproduced every
q-value and every one of the 266 classifications exactly by hand.

**The grid (19 constructs x 14 anchor traits x 2 channels = 266 cells per family).**
NEITHER 238 / STATIC-ONLY 24 / BOTH 3 / **DYNAMIC-ONLY 1** of 266.

**PRECISION CORRECTION (added 2026-07-12, operator-prompted).** "NEITHER" means "no
BH-surviving correlation with Big5 or MBTI was DETECTED," not "no personality signal
exists." Big5 and MBTI are two taxonomies, not an exhaustive nomological net — a
construct landing NEITHER against both may still relate to a personality framework this
program has not tested (see the Enneagram field newly identified in author_profiles.csv,
Audit log 2026-07-12), or may be a genuinely un-nameable axis outside any existing
taxonomy, exactly the possibility F10.2 raised at the theory's founding. Absence of
evidence against two specific criterion sets is not evidence of absence more broadly;
this correction does not, by itself, predict that a wider net WOULD find something —
it only retracts the overclaim that NEITHER settles the question.

**Adjudication against the registered leans.**
(a) CONFIRMED STRONGLY: 262/266 cells (98.5%) land STATIC-ONLY or NEITHER — the
null-consistent expectation for a battery built and validated on level correlations.
(b) CONFIRMED, at the lean's low end: exactly ONE dynamic-only cell exists —
**wcl_07 (dynamic channel) x MBTI TF_cont, r = -0.093, BH q = 9.0e-05** — and wcl_07 IS
inside the registered W10 gust1_P carrier-set expectation {wcl_02, wcl_07, wcl_11,
wcl_20}, exactly as leaned. Margin note (report honestly, not oversold): wcl_07's STATIC
channel on this same cell also clears BH (q = 1.7e-04) but fails the |r|>=.08 effect
floor (r_static = -0.073) — the cell is dynamic-only by a real but narrow margin, not an
overwhelming one. This is wcl_07's third appearance as an out-of-family construct (after
F12.8's CI-solid Delta_shape and its damped-oscillation texture) — the recurring
motion-anomalous construct now also carries the program's first exploratory
external-anchor association for a motion-only candidate, beyond its internal
split-half and spectral observations. This is not external validity.
(c) SLIGHT MISS: BOTH = 3 (registered band 0-2), one over the upper bound —
first_person_usage_v2 x MBTI TF (static r=.152, dynamic r=.089), wcl_03 x MBTI TF
(static r=.170, dynamic r=.088), wcl_23 x Essays Agreeableness (static r=-.140, dynamic
r=-.112). Recorded as a real, if small, miss rather than rounded into the band.
(d) **STANDING KILL NOT TRIGGERED** (dynamic-only is non-empty). Retrospective V7
adjudication: this count result preserves W10 only as an exploratory candidate; it does
not strengthen W10 through external validity. W10's split-half result was a descriptive
same-sample internal association, and F16 adds one modest MBTI T/F association whose
static channel narrowly missed the pre-set effect floor. Neither result establishes
reliability, external validity, or a psychological construct.

**Unregistered but must-report finding: the cap-sensitivity check FIRED, against the
registration's own prior.** Comparing STATIC person-means from the old capped Tier-U
cache vs. the new N1-uncapped-derived pool (n=3,605 common users): 5 of 19 constructs
move by >|.02| with paired-t p as low as 1.3e-40 — **first_person_usage_v2 alone moves
-0.190 (huge, systematic)**; wcl_36 -0.035, wcl_13 -0.038, wcl_07 +0.045, wcl_03 -0.024,
all p<3e-5. The registration's premise ("level is expected to be cap-insensitive,
unlike spectral texture, per W9") is **WRONG for at least these 5 constructs, most
severely for first_person_usage_v2 — the single construct carrying the program's
central H2 replication (first_person -> Neuroticism)**. Standing caveat, binding on all
future work: STATIC scores computed from the old capped extraction and STATIC scores
computed from N1-uncapped data are NOT directly comparable for these 5 constructs; any
future claim that mixes the two extractions for first_person_usage_v2 specifically must
re-verify rather than assume equivalence. F16's own grid is internally consistent (both
channels for a given cell come from the SAME uncapped pool), so this caveat does not
threaten the grid's own classifications, but it is a live methodological hazard for any
comparison ACROSS this cycle and pre-N1 cycles (EXPL-3, all v4/H2/H6 numbers reported on
capped data).

**Coverage (measured, not assumed).** PANDORA Big5 gate: 794/1,058 (75.0%) static-
covered, 636/1,058 (60.1%) dynamic-covered (m>=3). MBTI axes: only 5,364/9,042 (59.3%)
of labeled users appear in N1's uncapped pool at all (tier_l/tier_u verified disjoint,
0 overlap); of those, 3,933 static-covered, 3,028 dynamic-covered. Essays dev:
1,160/1,255 (92.4%) static, 1,053/1,255 (83.9%) dynamic.

**Secondary descriptors (signed memory, rho_pi/shape pair; exploratory-of-exploratory,
non-load-bearing).** Computed cleanly for all three populations with no estimability
failures; reported in a separate table; per the registration these are POPULATION-level
statistics from suica_core/motion.py (not per-person), so they cannot themselves enter
the per-person classification grid — this is a property of the reused estimator, not a
shortcut taken here.

**Portrait update for the invisible-factor program.** The motion-only signal is not
confined to one hand-tuned composite: at least one of its constituent RAW constructs
(wcl_07) independently clears a real external-anchor bar that its own static channel
misses, on an axis (MBTI Thinking/Feeling) this program had never previously tested
against GRAIN constructs. The effect is modest (|r| ~ .09, one cell out of 266) and
must not be oversold as a validated MBTI-T/F motion marker — it is a single BH-surviving
exploratory cell on spent/first-use labels, one data point toward the class of claim,
not a sealed finding. The larger, more actionable result of this cycle may be the
cap-sensitivity hazard: it retroactively flags that any cross-cycle comparison of
first_person_usage_v2 (the program's single most load-bearing construct) between
capped-era and N1-era work needs an explicit reconciliation step, not an assumption of
equivalence.

## F14 results (EXPL-4b: uncapped Tier-L motion features x spent opening-#1 labels —
## the standing kill's final leg, run AFTER registration 97ecca2; independently
## verified TRUST_WITH_CAVEATS)

**Governance, reproduced independently.** Gate n=1,058 exact; H2/H6 reproduced <1e-9
against OPENING_REPORT.json; EXPL-2 arm-C anchor reproduced EXACTLY (mean r=.1361,
per-trait match). No frozen script modified (EXPL-3's own build_motion_user_features /
finalize_axes / ridge_arm imported and called directly; capped-source artifacts never
read or overwritten). The verifier re-ran the entire script from scratch and got a
bit-for-bit identical output file.

**Numbers.** C (anchor) = .1361 (E .145 / N .146 / A .149 / C .079 / O .162). D (C + the
full motion layer: EXPL-3's original 10 position/slope features + a new signed-memory
and gust-amplitude family) = **.1029 (E .119 / N .089 / A .104 / C .075 / O .127) —
Delta = -0.0332, NEGATIVE ON ALL FIVE TRAITS**, worst on Neuroticism (-.057) and
Agreeableness (-.045), least on Conscientiousness (-.004). M (motion alone) = **-.0129
— motion features alone are actively worse than a null baseline**, a materially worse
motion-alone showing than EXPL-3's own +.0300.

**Coverage — a further, previously-undocumented layer of attrition below N1's own
per-text census.** N1's loose per-text count said 636/1,058 (60.1%) gated users have
>=1 text at m>=3, 316/1,058 (29.9%) at m>=5. But suica_core/motion.py's own POOLED
estimators need many such texts summed together per person (MIN_D2_ROWS=30 pooled
second-difference rows for signed-memory/gust-amplitude; PERIOD2_MIN_LAG_PAIRS=100 for
rho_pi/shape) — at that real per-user bar, only **22/1,058 (2.1%) gated users** are
estimable for the moment family and **1/1,058 (0.09%)** for the shape family. Complete-
case sensitivity (real, non-imputed coverage on every feature family at once) collapses
to n=1 — reported honestly as unfittable rather than smoothed over.

**Adjudication — the STANDING KILL FIRES.** EXPL-4a's Delta (-0.0096, Essays dev-half)
and EXPL-4b's Delta (-0.0332, this run) are BOTH <=0, outside and below the registered
lean band [+.000, +.015]. Per F14's own registered clause, the recorded verdict is:
**questionnaire-criterion insensitivity to the motion layer, pending BEHAVIORAL
criteria (the native corpus's job).** This is not a marginal call — the verifier
confirmed both deltas are clearly negative, not borderline.

**Mechanistic reading, power-annotated (the wcl_60/W8 precedent applies here).** The
kill's two components carry very different evidential weight:
- The POSITION/SLOPE layer alone (n=799, well-powered, EXPL-3's own recipe rerun on
  genuinely uncapped Tier-L text) already flips from EXPL-3's capped +0.0009 to
  **-0.0051 uncapped** — a small but real sign flip on a well-covered sample. Uncapping
  did NOT rescue this layer; if anything it went slightly negative. This sub-conclusion
  is TRUSTWORTHY.
- The SPECTRAL layer's incremental contribution to the further drop (-0.0051 -> -0.0332)
  rests on only 22 (moment family) / 1 (shape family) covered users out of 1,058 — far
  too thin to distinguish a true null from a real-but-undetected effect in a 1,058-user
  pooled ridge fit. This sub-conclusion should be marked **UNDECIDED-LOW-POWER**, not a
  clean falsification of spectral-texture relevance, by the same logic that downgraded
  wcl_60's AR-vs-MA identity in W8. The single-feature screen on the 22-user moment
  family produced 42 nominal |r|>=.10 hits, but at n=22 the two-tailed p<.05 threshold
  needs |r|~.42 — only a handful clear that bar (e.g. amp_proj_dcomp1->E +.408,
  amp_first_person->O -.398, mem_tension->E -.373) and even those are single-cell,
  uncorrected, and should be read as a noisy exploratory table, not confirmed signal.

**Verification caveats (recorded, not hidden).** (1) A real implementation bug: the
generalized imputation helper neuters the position layer's `has_long` missingness
indicator into a constant-zero column (EXPL-3's own code re-sets it correctly; this
script's generalization dropped that step) — one dead column out of ~40 motion
features, verified immaterial to the ridge fit (zero-variance under RidgeCV) but a real,
undisclosed deviation from the stated "mirrors EXPL-3's convention." (2) The build
agent's cited "alternative-standardization Delta=-0.0272" robustness check does NOT
reproduce (a faithful re-implementation gives -0.0332, mathematically expected since the
per-fold StandardScaler inside ridge_arm() is invariant to that class of rescaling) and
has no artifact backing — struck as unverified/incorrect. Its companion claim
("position-layer-alone Delta=-0.0051") DID reproduce exactly and stands.

**Consequence for the three-leg F14 program.** capped PANDORA position-only (EXPL-3,
+0.0009, ~flat) -> Essays uncapped (EXPL-4a, -0.0096, mildly negative) -> PANDORA fully
uncapped position+spectral (EXPL-4b, -0.0332, clearly negative). The cap was NOT masking
a real questionnaire-relevant motion signal for Big5 detection on this cohort — the
well-powered position layer's own sign flip (capped-positive to uncapped-negative) rules
that hope out directly. What remains genuinely open is the spectral layer specifically,
which never got a fair test here: PANDORA (even fully uncapped) does not have enough
per-user long-text VOLUME for the pooled spectral estimators to clear their own
estimability bars for more than ~2% of labeled users. This is the sharpest concrete
argument yet for why the native corpus (N4, already delivered) is necessary rather than
optional — it is designed to guarantee, by construction (>=1,500-2,500 words per
session, 2 sessions/person), the long-text volume PANDORA structurally cannot supply.

## F17 — Enneagram as a third external anchor (registered 2026-07-12, operator-ordered:
## theory-enrichment track, zero new data collection)

**Mandate.** F16 tested Big5 and MBTI only — two taxonomies, not an exhaustive
nomological net (per the NEITHER precision correction above). PANDORA's own
author_profiles.csv already carries an untouched Enneagram field (794 users; type +
wing, e.g. "5w4") — a THIRD, structurally distinct (typological/circumplex, not
dimensional) taxonomy, usable with zero new data collection. F17 repeats F16's exact
classification methodology against this new anchor.

**Object.** Same 19 frozen constructs, same STATIC S_c(u) / DYNAMIC D_c(u) (per-construct
gust amplitude) channels as F16, reusing F16's own already-built and independently-
verified uncapped windows-scored cache (results/suica_f16_visibility_taxonomy/
pandora_uncapped_windows_scored19.parquet) wherever it covers the enneagram-labeled
population — do not rebuild from scratch if that cache already fits.

**Anchor design.** Enneagram type is CATEGORICAL (9 levels), not continuous like Big5/
MBTI-cont — a real methodological difference from F16, to be handled explicitly, not
silently forced into the same shape. PRIMARY: one-vs-rest point-biserial correlation
per type (is_type_1..is_type_9, 9 binary columns) x 19 constructs x 2 channels = 171
cells per family, BH-FDR per family (static, dynamic), same |r|>=.08 floor as F16 —
chosen for direct comparability with F16's own convention, at the cost of the 9
indicators being non-independent by construction (correlated through the categorical
structure) — flag this explicitly rather than treating the 9 as orthogonal contrasts.
SECONDARY, non-load-bearing (mirrors F16's own secondary-descriptor convention): a
one-way ANOVA eta-squared per construct x channel (19 x 2 = 38 cells) as an omnibus
cross-check that does not feed the primary classification.

**Governance.** FIRST correlational use of Enneagram labels in this program (confirmed:
no prior script reads this field) — same treatment as MBTI got in F16: EXPLORATORY
banner, ledger row, no prior replication-assert to reproduce (none exists), and this run
itself becomes the future reference point. Measure actual coverage empirically (how many
of the 794 enneagram-labeled users appear in N1's uncapped pool / the F16 cache) — do not
assume the 793 MBTI-overlap figure transfers to N1 coverage without checking.

**Leans (directional, before run).** (a) Most cells land STATIC-ONLY or NEITHER again,
consistent with F16's 98.5% base rate — no reason to expect Enneagram behaves
differently in kind. (b) Given wcl_07's population overlap is nearly identical to the
793 MBTI-TF-tested users, IF wcl_07's dynamic signal reflects a genuine trait-general
"thinking/feeling-style" property (rather than an MBTI-specific artifact), some echo is
plausible in Enneagram types carrying "head/thinking" lore (types 5, 6) — lean: 0-3
dynamic-only cells total, same order of magnitude as F16, NOT zero but not large; this
lean is explicitly LOW-CONFIDENCE (Enneagram-Big5/MBTI correspondence is itself
contested in the literature, not a settled fact to lean on). (c) BOTH rare, similar
order to F16 (0-3). **KILL condition is deliberately soft here** (unlike F16's standing
kill): zero dynamic-only cells would not by itself falsify the narrow averaging/path
contrast statement. W10 was not externally validated by the exploratory MBTI result;
zero cells would mean wcl_07's observed MBTI-T/F association does not
generalize to a structurally different taxonomy, narrowing (not killing) the claim to
"MBTI-T/F-specific" rather than "trait-general."

**Standing self-check to apply on adjudication (carried from the T8 theorem work below):
re-examine whether any newly-classified DYNAMIC-ONLY cell is a genuine kernel-only
object or a floor-threshold artifact** — report each hit's STATIC r and BH-q alongside
its dynamic r and q, not just the pass/fail label, exactly as F16 should have done more
prominently for wcl_07 (whose static r=-.073 missed the .08 floor by only .007 while
still clearing BH q=1.7e-4).

## Shared-rhythm hypothesis test (registered 2026-07-12, resolving the open item flagged
## in F15/W10's registration: "carriers = the spectral-anomaly constructs (shared-rhythm
## hypothesis open)")

**Mandate.** W10 found gust1_P's carrier constructs (wcl_02, wcl_07, wcl_11, wcl_20) and
left open whether these are the SAME constructs independently flagged as spectrally
anomalous at the population/corpus-pooled level (F12.7-F12.9: wcl_20's depressed
rho_pihalf/CI-solid even-lag a2; wcl_07's CI-solid Delta_shape/damped-oscillation a2;
wcl_60's MA(1)/bounce identity) — i.e., is "a construct co-activates with others in
bursts" (W10, per-person) the SAME phenomenon as "a construct has unusual population-
pooled spectral texture" (F12.7-9, corpus-pooled), or do these merely happen to overlap
in gust1_P's specific loadings by coincidence?

**Object.** For each of the 19 constructs, using N1's uncapped data (reusing F16's own
per-construct dynamic-channel machinery, already built and verified), compute:
(i) SPECTRAL-ANOMALY MAGNITUDE at the population level — the already-published F12.7-9
numbers (|rho_pi - 1|, |Delta_shape|, |a2| from the AR(2) triangle) collected into one
per-construct anomaly score (document the exact combination rule — e.g., max-normalized
mean of the three, or report all three separately if no single combination is
defensible; do not silently pick one without saying so).
(ii) PERSON-LEVEL SUSCEPTIBILITY-STABILITY — W10's split-half |activation| stability
estimator, generalized from the single gust1_P composite to EACH of the 19 raw
constructs individually (reusing W10's person_susceptibility() function).
Then correlate (i) against (ii) ACROSS THE 19 CONSTRUCTS (n=19, one point per
construct) — a small-n, construct-level correlation, explicitly flagged as
LOW-POWERED/SUGGESTIVE-ONLY given n=19, reported with exact p-values and a rank
(Spearman) correlation alongside Pearson given the small n and likely non-normality of
anomaly scores.

**Leans.** Positive association: constructs with LARGER population-level spectral
anomaly magnitude show HIGHER person-level susceptibility-stability — the conjecture
that "population-pooled spectral weirdness" and "individual gustiness-stability" are two
observable faces of the same underlying rhythmic-coordination phenomenon, not two
independent facts that coincidentally overlapped in one composite's loadings. Given
n=19, no specific correlation magnitude is leaned — only the SIGN and nominal
suggestiveness (p<.10 as a "worth a second look" bar, explicitly NOT treated as
confirmatory at this n).

**Kill condition.** If the correlation is null or negative (particularly if wcl_60 —
the construct with the clearest population-level anomaly of all 19, per W6/W7's
CI-solid rho_pi=2.51 — does NOT also rank among the higher-susceptibility constructs),
record "shared-rhythm hypothesis NOT supported at this power — the gust1_P carrier set
and the individually spectrally-anomalous constructs are better treated as two
independent facts that happened to overlap in one composite's specific loadings, not
evidence of one underlying phenomenon." Given n=19, a null result here should be
reported as "underpowered/undecided," not a confident falsification, unless the sign is
actively wrong for the clearest cases (wcl_60, wcl_07, wcl_20).

## F17 results (run AFTER registration 8afb3e3; independently verified TRUST — exhaustive
## re-derivation of all 342 cells + 3 independent BH-FDR implementations, zero
## discrepancies)

**The grid (19 constructs x 9 one-vs-rest Enneagram types x 2 channels = 171 cells per
family).** **COMPLETE NULL: 171/171 NEITHER in both families (0 STATIC-ONLY, 0
DYNAMIC-ONLY, 0 BOTH).** Smallest raw p-values (dynamic wcl_22 x type7, p=.00104; static
first_person_usage_v2 x type5, p=.00129) both fail BH decisively (q=.178, q=.220 — 3.5-
4.4x over threshold, not close calls). Verifier's distributional check: KS-test of all
342 p-values against Uniform(0,1) gives D=.036, p=.76 — genuinely indistinguishable from
a global null, not merely "BH was too strict to catch a real signal."

**Coverage (corrected).** 794 enneagram-labeled users; 372 (46.9%) static-covered, 304
(38.3%) dynamic-eligible (m>=3) in F16's reused cache. MBTI-axis overlap is **790/794
(99.5%), not the 793 figure in the registration** — a small factual correction, caught
and reported plainly by the executing agent rather than silently reconciled (the 4
enneagram-only, non-MBTI users are individually identified in the verifier's output).
Smallest Enneagram type (type 3) has only 18/16 covered users — the thinnest cell in
either family.

**Adjudication against the registered leans.**
(a) CONFIRMED even more strongly than F16's own 98.5% — but for a QUALITATIVELY
DIFFERENT reason worth flagging, not just a bigger number: F16's null was carried mostly
by NEITHER but had a genuine 24-cell STATIC-ONLY bucket; F17 has ZERO hits of ANY kind,
including static. This is a more complete null than lean (a)'s F16-by-analogy framing
anticipated.
(b) Count (0) sits at the bottom of the registered 0-3 band — technically not violated —
but the SPECIFIC low-confidence sub-lean (wcl_07 echoing via "head/thinking" types 5/6)
is NOT supported. wcl_07 shows a CLEAN null here (largest static r=+.136 at type6,
q=.307 — nowhere near BH; dynamic r=+.038/-.068 at types 5/6 — tiny), not a narrow miss
like its exploratory MBTI-T/F association in F16. The two external-anchor results for wcl_07 are
qualitatively different, not just different in size.
(c) CONFIRMED (0, within 0-3).
**Soft-kill TRIGGERED (0 dynamic-only cells) — per the pre-committed framing, this does
not by itself falsify the narrow T8 operator statement that averaging can erase path
contrasts. F10.8, E4, and W10 motivate a residual/path-structure candidate; they do not
independently certify a population kernel rank or a psychological dimension.** It narrows wcl_07's finding from
"trait-general thinking/feeling-style signal" to "specific to MBTI's own T/F axis" —
Enneagram, a structurally different (typological, not dimensional) taxonomy built from
an almost-identical user population, does not replicate it.

**Power vs. absence-of-relationship.** The verifier's explicit read, independently
derived: SE(r) ~ .052-.058 at this n; the single largest observed |r| (.166) is only ~3
SEs from zero — enough for uncorrected p<.05 but far short of what BH across 171
(correlated, one-hot) tests requires. This is best read as UNDERPOWERED, not as
evidence Enneagram is unrelated to these constructs — F16's exploratory associations against
Big5/MBTI were of similar magnitude (|r|~.09-.17) and needed larger, better-balanced
covered populations to clear BH. Consequence for the operator's original objection (that
Big5/MBTI's coverage might be insufficient for a high-dimensional scale): the LOGIC of
that objection is untouched — this result only says the FIRST concrete candidate tested
(Enneagram, at this sample size) did not pan out, not that no other taxonomy would.

## Shared-rhythm results (run AFTER registration 8afb3e3; independently verified
## TRUST_WITH_CAVEATS — computation confirmed exact, but the headline conclusion is
## DOWNGRADED after the verifier executed the same-corpus check this registration only
## asked to attempt "if feasible")

**Registered (cross-corpus) numbers — computationally exact, reproduced from scratch by
the verifier via independent code with zero discrepancies.** Combined anomaly score
(mean of max-normalized |rho_pi-1| [Essays, W6], |Delta_shape| [Essays, W7], |a2|
[Essays, W8]) vs. per-construct susceptibility-stability (W10's split-half estimator,
generalized to all 19 constructs, measured on the PANDORA uncapped pool, n=2,355
split-half-eligible users): **Pearson r=.633, p=.0036; Spearman rho=.426, p=.069**;
wcl_60 ranks 2nd of 19 on susceptibility (wcl_07 3rd, wcl_20 6th); leave-one-out Pearson
stable in [.504, .737] across all 19 drops.

**The problem, found by the verifier going beyond its brief.** The anomaly magnitude was
measured on ESSAYS (where W6/W7/W8 happened to be built) while susceptibility-stability
was measured on PANDORA — a cross-corpus comparison this registration should have
required to be same-corpus from the start (a gap in the registration itself, owned here,
not the executing agent's error). The verifier didn't just flag this in the abstract —
it ACTUALLY RAN the same-corpus (PANDORA-native) version, reusing W6/W8's own validated
closed-form engines (not hand-transcribed) applied to PANDORA's 5<=m<=12 stratum
(3,879 texts/1,656 users — MORE data than Essays' 1,349 texts, and confirmed clear of
the W9 cap-artifact per N1/EXPL-4b's chars/token=4.81 at m>=4).

**Same-corpus result: the PRIMARY (Pearson) statistic COLLAPSES.**

| variant | Pearson r | p | Spearman rho | p |
|---|---|---|---|---|
| Registered (Essays anomaly x PANDORA susceptibility) | .633 | .0036 | .426 | .069 |
| PANDORA-native, all qualifying texts (n=3,879 texts/1,656 users, clustered) | .181 | .458 | .358 | .133 |
| PANDORA-native, one text/user dedup (n=1,656) | .151 | .539 | .447 | .055 |

The stronger, cleaner original statistic (Pearson) is clearly NON-SIGNIFICANT in both
same-corpus variants. Spearman is internally INCONSISTENT across the two robustness
variants (fails clustered, borderline-recovers deduped) — read as noise at n=19, not a
robust secondary confirmation.

**Component-level nuance (not a total wash).** |rho_pi-1| ALONE shows a STRONGER
same-corpus rank correlation (Spearman rho=.49, p=.03, vs. Essays' .38/.11), with wcl_60
staying top-ranked specifically on this one component. |Delta_shape| ALONE FLIPS SIGN
entirely (Essays r=.51/rho=.22 positive -> PANDORA r=-.10/rho=-.15 negative) — this sign
flip, inside a combined score that weights all three components equally, is what drags
the cross-corpus headline number down once the confound is removed.

**Adjudication — DOWNGRADED from the build agent's "SUPPORTED_AT_LOW_POWER."** The
registered combined-score, cross-corpus claim is **NOT SUPPORTED AS REGISTERED**: its
strongest statistic does not survive same-corpus re-derivation. This is best read as a
REGISTER-SPECIFICITY ARTIFACT, consistent with — and a further small piece of evidence
for — T6 ("portability follows visibility"): cross-corpus comparisons of motion-layer
properties are exactly where this program has repeatedly found trouble (E2d's gust1_P
non-transport, .200 vs random .23). The door is NOT fully closed, though: the rho_pi-
specific thread (not the 3-component combined score) shows a same-corpus-ROBUST, even
STRENGTHENED signal with wcl_60 anchoring it — a narrower, live candidate that needs its
own freshly-registered, same-corpus, cluster-robust (PANDORA has ~46% repeated-user
texts in this stratum, unlike Essays) confirmatory test before being treated as
established in either direction. Until that runs, the shared-rhythm hypothesis in its
ORIGINAL combined-score form is recorded as NOT SUPPORTED; the rho_pi-only sub-claim is
UNDECIDED-PENDING-PROPER-TEST.

**Process note.** This is the verification step working exactly as designed: an
initially clean, exciting positive result (r=.63, p=.004) was substantially walked back
by the SAME verification pass that confirmed its arithmetic was flawless — the bug here
was in the registration's own design (an unstated same-corpus assumption), not in the
code. Filed alongside E3's bounce (struck by W2a) and W9's inverted mechanism as another
case where independent scrutiny changed the substantive conclusion, not just the
numbers.

## F18 — V6 identifiability correction and simulation license (2026-07-12)

An independent audit found three blocking overclaims in the first V6 theorem wording:
T1 used the iid `Gamma0/m` result as a general aggregation law; T2 omitted endpoint
autocovariance; and T8 wrote `range(A) orthogonal ker(A)` even though the spaces differ.
The corrected forms are T1-prime, T2-prime, and T8-prime in `THEORY_V6.md`.

The executable SIM-W0..W8 matrix then underwent three adversarial revisions. Hard-coded
guards, mixed statistical fields, an incorrect VAR standard error, an oracle alias
detector, and under/over-covering flow intervals were each rejected and replaced. The
final full run used 500 Monte Carlo repetitions and 1,000 paths per repetition; all 76
row-level gates passed. Exact results and hashes are in
`reports/V6_SIMULATION_PROOF_MATRIX.md`.

**License:** internal algebra/estimator behavior in the named synthetic generators.
**Not licensed:** a human dynamic personality dimension, questionnaire orthogonality,
cross-register portability, clinical meaning, or detection lift. E4 and W10 do not
become confirmatory through this simulation; their selection/null concerns remain open.
