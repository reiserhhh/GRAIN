# GRAIN-C1-02 protocol V1.3 supplement (r4): target-specific dual readouts

Frozen 2026-10-02 ~20:58 JST before any r4 output. Authority: parent
`reports/claude_research_loop_20261002/NEXT_PACKET_C1_02.md`, `SEMINAR_C1_02_DECISION.md`.
V1 applies unchanged: four primary worlds, sample sizes (12 pilot writers, 9 train), materials,
targets, information boundary, F04 kernel, tolerances (anchor |t| >= 4, design R^2 > .5 flag,
tau_T = max(.15, 2 SE), tau_F = .20, nonconstancy) and the 3/3 acceptance rule. r1-r3 verdicts and
roots stand. Code: `grain_research/construction/dual_readout.py` (new; legacy module untouched),
runner `scripts/run_grain_dual_readout.py`, tests `tests/test_grain_dual_readout.py`.
Seeds 20261011, 20261012, 20261013. Output `results/grain_construction_v1/C1/hidden_map_r4/`.

## Algebra (standard; not a new result)

Forward loading v (pilot regression of x on the declared label) and backward readout w differ.
For C = G diag(sb^2, sr^2) G^T, G = [v s] full column rank: w = C^+ v / (v^T C^+ v) =
G (G^T G)^-1 e1, so w^T v = 1 and w^T s = 0 although v^T s != 0. If corr(b, r) = rho,
C equals the independent form with s' = s + (rho sb/sr) v and the minimum-variance readout has
w^T s = -rho sb / sr. In this world (sb = .5, sr = 1, rho = .6) the exact population readout
has style slope -.3 per style SD, i.e. R_F(style) = .6 for the Delta target.

## Arms

| arm | information | construction |
|---|---|---|
| pilot_pcr (fixed comparator) | as V1 | unchanged, LOWO k from V1 grid only |
| lcmv_dual | source x, z, author; pilot train x, label | C = noise-corrected between-author slope covariance (V1 `_slope_moments`); ONE rank rule: keep eigenvalues > max(-lambda_min, 0); v_hat = pilot forward loading; w = C_k^+ v_hat / (v_hat' C_k^+ v_hat) |
| pilot_dual (AUGMENTED) | lcmv_dual + pilot writer IDs + recorded assigned pilot condition `pilot_z` | s_hat = top eigenvector, inside range(C_k), of between-writer covariance of pilot slopes on z (label partialled out), sampling-corrected; LCMV with w'v_hat = 1, w's_hat = 0 |
| oracle_map, oracle_perp_style | privileged | evaluator-only references (true expression; true style loading) |

`pilot_z` is exported by a generator function that replays the pilot design stream up to the
condition assignment (`assigned_pilot_conditions`); fit functions never read EvaluatorTruth.
Unit and offset by V1 classical calibration on pilot train writers.

Frozen rules: rank 0 -> RANK_ZERO_UNSUPPORTED; range fraction ||U_k' v_hat||^2/||v_hat||^2 < .5
-> RANGE_UNSUPPORTED; |t| < 4 -> ANCHOR_UNSUPPORTED_DIRECTION; pilot_dual with k < 2 ->
RANK_TOO_LOW_FOR_TWO_CONSTRAINTS; style eigenvalue <= 0 -> STYLE_UNSUPPORTED. The selected rank,
tau, eigenvalues and range fraction are reported. The rank rule is a heuristic noise-floor witness;
a wrong rank is reported, never replaced by the known latent rank. No rank, regularization,
pilot-size or tolerance search.

Target-specific use: the dual readouts are contrast readouts. Level targets keep their own
estimator (pilot_pcr passed W1 in r1-r3); dual-arm level scores are reported, not forced.
Pilot purity assumption (declared, not proven by the writer-R^2 screen): pilot labels are
independent of writer fingerprint and style, and pilot conditions are assigned independently of
labels.

## Uncertainty and evaluation

Interval variance = posterior variance + between-draw variance over 30 joint bootstrap draws
(source authors and pilot train writers resampled; every estimator step refit, rank rule rerun;
pcr keeps its V1 k). No leakage/bias allowance is added: the residual style loading of a readout is
not observable without independent style-manipulation data; this missing observable is reported.
Evaluation as V1 (cold new authors, target, fingerprint, style, wording, held-topic material,
held condition, coverage, paired differences now vs pilot_pcr). Side coordinate: for pilot_dual,
the per-author slope of s_hat'x is exported as a numeric context-register coordinate (not a named
or pure psychological score).

## Correlated boundary (separate; never used for acceptance)

`correlated_boundary/` uses the W2 latents with corr(b, r) = .6 in source and application
authors (pilot unchanged), same seeds and arms. Prediction from the algebra: lcmv_dual style
R_F near .6 even with high target correlation; pilot_dual should not inherit the correlation
because its style loading comes from the pilot. A failure here is an independence-assumption
boundary, not a failure of all targets.

## Disclosed pre-freeze scratch check

Seed 999 (not an evaluation seed): code runs; the rank rule selected k = 8 (not the latent 2);
both dual arms showed style slope about .03 per SD. The rule was not changed.
