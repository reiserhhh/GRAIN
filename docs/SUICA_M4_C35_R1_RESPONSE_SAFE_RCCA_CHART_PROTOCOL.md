# GRAIN M4-C.3.5-R1 Response-Safe RCCA Chart Protocol

## Status

This protocol was frozen after unit/smoke tests and a two-repetition
development run, but before opening the eight-repetition confirmation endpoint.
The confirmation seed, worlds, estimator, thresholds, and source hashes below
are fixed. Confirmation may be opened once.

The earlier inverse-signal EIV-SVD prototype is a falsified development route.
It inverse-whitened near-zero cross-half eigenvalues and therefore amplified
finite-sample noise. It produced unstable support ranks and axis rotations even
when the whole reliable subspace remained similar. None of its endpoint values
are confirmation evidence.

## Question

Can a chart built from replicated **pre-response condition observations only**
recover a repeatable cross-source condition-relation subspace without using
responses, mechanism endpoints, generator truth, personality labels, or C3.4
downstream headroom?

The target is a chart-equivalence class, not named axes. Near-degenerate
canonical directions are treated as spectral blocks, because their projector
and induced author geometry are identifiable even when individual axes rotate.

## Fixed estimator

For source \(s\), split reference authors into four deterministic blocks and
form block prototypes \(X_{sb}\). From the calibration panel estimate a
Ledoit-Wolf shrunk total covariance \(T_s\) and symmetric cross-block
repeatability covariance \(S_s\):

\[
R_s=T_s^{-1/2}S_sT_s^{-1/2}.
\]

A fixed-whitener condition permutation supplies the max-\(T\) null threshold.
The native support rank \(r_s\) is the number of reliability eigenvalues above
that threshold. Author bootstraps always take the fixed top-\(r_s\) subspace;
they do not select their own rank. If \(P_{sb}\) is bootstrap projector \(b\),

\[
\bar P_s=\frac1B\sum_b P_{sb},
\qquad
\Pi_s=\operatorname{TopEig}_{r_s}(\bar P_s).
\]

The support gate uses the native rank boundary, next-rank boundary, mean
projector stability, consensus eigenvalues/eigengap, and native-consensus
affinity. It cannot be passed by averaging one unstable weak direction into
five stable ones.

Within the frozen source supports, regularized CCA estimates

\[
M_\gamma
=
(\Sigma_{11}+\gamma I)^{-1/2}
\Sigma_{12}
(\Sigma_{22}+\gamma I)^{-1/2}.
\]

\(\gamma\) is selected on the held-out pre-response selection panel by a
one-standard-error rule over the fixed grid
\(\{10^{-4},10^{-3},10^{-2},10^{-1}\}\), subject to condition number at most
100. A frozen-whitener cross-source permutation sets the canonical null.
Bootstrap lower and upper bounds identify the retained shared-rank interval.
Adjacent near-degenerate singular directions are merged into spectral blocks.
The emitted chart weights blocks and never re-whitens individual canonical
axes.

## Confirmation design

- Fresh seed: `577215664`.
- Independent repetitions: `8`.
- Main worlds per repetition: `5`.
- Refusal worlds per repetition: `4`.
- Reference authors: `32`; mechanism authors: `16`.
- Calibration/selection/evaluation occasions: `40/16/5`.
- Support permutations/bootstraps: `199/199`.
- Canonical permutations/bootstraps: `499/199`.
- Null trials per main-world fit: `25`.
- Cluster bootstrap repetitions: `5000`.
- Response-byte mutation, orthogonal gauge, common shift, source shuffle,
  forbidden provenance, support shift, and exact observable alias are fixed
  controls.

The executable specification is
`configs/m4_response_safe_rcca_chart.confirmation.json`.

## Frozen gates

All gates must pass:

1. mean projector affinity at least `.80`, clustered lower bound at least
   `.75`;
2. held-out source Gram CKA at least `.90`, clustered lower bound at least
   `.85`;
3. resolved shared-rank rate at least `.95`;
4. per-world modal rank/block pattern frequency at least `7/8`;
5. support stability at least `.80`, its lower bound at least `.75`;
6. consensus concentration at least `.80`, weakest retained consensus
   eigenvalue at least `.60`, and consensus eigengap lower bound above `0`;
7. retained-rank boundary lower bound above `0`, next-rank boundary upper
   bound below `0`, native-consensus affinity at least `.80`, and no native
   refusal;
8. condition number at most `100`, singular value at most `1+10^-8`, negative
   spectral mass at most `.10`, asymmetric mass at most `.30`, and minimum
   role coverage at least `.80`;
9. aggregate null false-positive Wilson upper bound at most `.05`;
10. source-shuffle zero-rank rate at least `.95`;
11. exact response-hash invariance, gauge error at most `10^-6`, common-shift
    error at most `10^-10`, zero false latent recovery under exact observable
    alias, and complete refusal of forbidden provenance and evaluation support
    shift.

`topology_mismatch` remains a diagnostic refusal world, not a primary gate:
the chart may legitimately transport over some topology changes, while all
support and geometry gates still apply.

## Decision and boundary

- `M4_C35_R1_CONFIRMATION_GO`: the frozen response-safe chart may enter a
  **new, separately registered** C3.4 downstream replacement experiment.
- `M4_C35_R1_CONFIRMATION_NO_GO` or alias failure: stop this chart route.

GO does not establish response prediction, creation-mechanism recovery,
personality meaning, natural-text validity, or M4-D readiness. It establishes
only finite-synthetic, pre-response chart identification under the registered
world family.

## Development evidence

The final code-equivalent two-repetition development run returned
`M4_C35_R1_READY_TO_FREEZE` with all 15 gates passing:

- minimum projector-affinity clustered lower bound: `.998962`;
- minimum held-out CKA clustered lower bound: `.998691`;
- minimum support-stability lower bound: `.869585`;
- weakest retained consensus eigenvalue: `.783813`;
- minimum consensus eigengap lower bound: `.538576`;
- minimum retained-rank boundary lower bound: `.637560`;
- maximum next-rank boundary upper bound: `-.140440`;
- rank pattern frequency: `1.0`;
- aggregate null Wilson upper bound: `.040410` over `250` trials;
- source-shuffle zero-rank rate: `1.0`;
- minimum coverage: `.875`.

These values justified sealing only; they are not confirmation results.

## Frozen source hashes

```text
512aa10f2d4b6ca538e2ebf1e78791d3b661f7b77f5a81adc7ed7b4b6da932fa  configs/m4_response_safe_rcca_chart.confirmation.json
c24150c7c13d9c7f06f60fcbb26381232dfe0beebcf3f2935bf0467a48459ab4  scripts/run_suica_m4_response_safe_rcca_chart.py
0ea35d0748c8ebdde8ba7a767c5f40759f2f967d7379d1e67feb72d5d26f8967  suica_core/m4_response_safe_rcca_chart.py
e76ef8e10d88d6c413ba04877b52dac00de5576885d2baeb0702e4fab797fb43  suica_core/m4_chart_ecology_generator.py
5efc028b2604786838b4d7f4db0c62a1a40bea71999a701450ef3dc19102cc18  suica_core/m4_condition_manifold_contracts.py
84d45bb2ac297241a7ea523bee5b91a6e1a6b4ad64226d4df1b69df558fd1ccf  tests/test_m4_response_safe_rcca_chart.py
```

## Confirmation outputs

- `results/m4_response_safe_rcca_chart_confirmation/metrics.csv`
- `results/m4_response_safe_rcca_chart_confirmation/controls.csv`
- `results/m4_response_safe_rcca_chart_confirmation/alias_audit.csv`
- `results/m4_response_safe_rcca_chart_confirmation/decision.json`
- `reports/SUICA_M4_RESPONSE_SAFE_RCCA_CHART_CONFIRMATION.md`

The opened result may be appended below without modifying the frozen
estimator, config, hashes, or gates.

## Opened confirmation result

The endpoint was opened once. All frozen hashes matched and the output
contained exactly 40 unique main cells (`8 repetitions x 5 worlds`), 96 unique
control cells, and 8 alias cells. The decision was:

```text
M4_C35_R1_CONFIRMATION_GO
```

All 15 gates passed. Principal diagnostics were:

- projector-affinity clustered lower bound: `.998617`;
- held-out source CKA clustered lower bound: `.998678`;
- support-stability lower-bound minimum: `.851876`;
- weakest retained consensus eigenvalue: `.727351`;
- consensus eigengap lower-bound minimum: `.444300`;
- retained-rank boundary lower-bound minimum: `.577633`;
- next-rank boundary upper-bound maximum: `-.136961`;
- rank resolution and rank/block pattern frequency: `1.0 / 1.0`;
- aggregate null false positives: `7/1000`, Wilson upper `.014378`;
- source-shuffle zero-rank rate: `1.0`;
- forbidden-provenance and support-shift refusal: `1.0 / 1.0`;
- response-hash failures and alias false recoveries: `0 / 0`;
- minimum evaluation coverage: `.8125`.

This opens only the separately registered C3.5-R2 chart-replacement test. It
does not alter the C3.4 NO-GO result and does not license M4-D.
