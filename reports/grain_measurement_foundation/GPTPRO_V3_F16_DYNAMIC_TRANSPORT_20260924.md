# F16-R03: dynamic-target pilot transport

Date: 2026-09-24 01:15 JST. **Finite numeric dynamic transport executed and audited.**
The full goal remains ACTIVE. Changed controlled-text reader channels and public
serialization of calibration mixtures remain next; this packet does not close all
F16 or all44 computational questions.

## What this adds to the measurement method

- Calibration is target-specific: the same global reader change damages future
  response scoring much more than current-state scoring in this design.
- High rank correlation can conceal major unit bias. T3 r stays near .902 while
  nominal95 interval coverage falls to .288; independent pilot calibration returns
  scores to their original-unit reference and coverage to .928.
- Conditional changes need conditional calibration. Adding pilot uncertainty to
  the wrong pooled inverse does not remove its systematic error.
- Recovering the original unit is different from having a correctly calibrated
  population model. Even the original-unit reference retains coverage failures.

All correlations below concern known latent/response **simulation targets**, not
human personality labels or a new natural-language accuracy score. No API, neural
training, private data, remote execution, lockbox or publication occurred.

## Design and actual execution

Three seeds, independent pilot100 per seed (first60fit/last40validation), and200
application authors per seed. Two targets share these600 application authors:
T2 current s47; T3 realized mean of y48..51 under (+1,-1,+1,-1). Source parameters
are all200 models in each frozen F16 source bundle. No source-response refit.

Application/pilot a~N(1,1.4), b~N(.2,.7), rho~Uniform(.2,.8), state and observation
innovation variances .25. Prefix48 uses the core four-step condition alternation;
future4 uses the target plan directly. Population change is not standardized away.
Truth files are separate; all scoring completes before a separate evaluation
process opens application truth. Each fit uses200 whole-pilot-author bootstrap
draws, with all source-by-pilot combinations integrated.

Four reader maps: identity;1+1.5*y;z+(1+.25*z)*y;y+.1*y^3. Each has uncalibrated,
global pilot inverse, conditional pilot inverse and known-inverse reference;
nonlinear adds the prescribed cubic inverse.51seed/arm cells,40,800 output rows
(including unsupported), three300-author source norms recomputed. Each full seed
took about170seconds; three ran concurrently. All jobs are terminal.

## Formula and a useful finite compression

Conditional on inertia coordinate eta, the Gaussian prior on (a,b) and the latent
state path can be integrated analytically. Let Y be the48-vector of observations,
C its conditional covariance, m its mean, H the target-observation cross covariance,
and V the target covariance. Both targets use

```text
log p(Y|eta) = -.5*((Y-m)' C^-1 (Y-m) + logdet(C) + 48*log(2*pi))
E[T|Y,eta] = mu_T + H C^-1 (Y-m)
Cov[T|Y,eta] = V - H C^-1 H'
```

For z=+/-1, every global/conditional affine inverse has the exact form

```text
Y_corrected = alpha*Y_new + beta*(z*Y_new) + gamma*1 + delta*z.
```

Thus its likelihood and target moments can be evaluated through the four-column
basis G=[Y_new,z*Y_new,1,z], without refitting an optimizer for every pilot draw.
This is an exact computational identity for the declared calibration family and
schedule, **not** a claim that general text or personality is four-dimensional.
The cubic inverse, if supported, uses [1,Y_new,Y_new^3].

Eta is integrated with64-point Gauss-Hermite; a,b and current/future states are
conditionally analytic. Source components retain equal frozen cut weights, not
application-likelihood-based reweighting. Conditional-state/future, personal
parameter, source and pilot covariance are separated, retaining the T2/T3 cross
covariance and all shared future innovations.

This is a new versioned integration estimator, not bitwise reproduction of the old
Laplace personal draws. All300 source prefixes per seed receive the same new
estimator to freeze new reference mean/SD. Old source parameters, public bundles,
reference norms and archived results are untouched. Three actual old-T2/old-T3
comparison pairs are retained; maximum selected score difference .01403. They are
different approximation methods, not an equality gate or an accuracy comparison.

## Results

Each reported value is the mean across three equal-size seeds, not pooled r.
Intervals are working moment-normal95 intervals. Main table uses the full pilot
mixture, not a plugin point calibration.

| Reader change / inverse | T2 MSE | T2 r | T2 coverage | T3 MSE | T3 r | T3 coverage |
|---|---:|---:|---:|---:|---:|---:|
| Original units / none | .181339 | .743190 | .916667 | .285738 | .901677 | .928333 |
| Global / none | .313406 | .737777 | .861667 | 2.663048 | .901528 | .288333 |
| Global / global pilot | .181339 | .743190 | .916667 | .285738 | .901677 | .928333 |
| Conditional / global pilot | .210877 | .742789 | .900000 | .332392 | .889790 | .901667 |
| Conditional / conditional pilot | .181339 | .743190 | .916667 | .285738 | .901677 | .928333 |
| Nonlinear / none | 1.253234 | .621320 | .728333 | 3.367711 | .846079 | .630000 |
| Nonlinear / global pilot | .291325 | .639616 | .870000 | .446908 | .851926 | .878333 |
| Nonlinear / conditional pilot | .274649 | .637308 | .885000 | .463639 | .849707 | .881667 |
| Nonlinear / cubic pilot | unsupported | NA | NA | unsupported | NA | NA |
| Nonlinear / known inverse | .181339 | .743190 | .916667 | .285738 | .901677 | .928333 |

Paired2,000-draw author bootstrap, stratified by seed and conditional on fitted
artifacts, gives MSE differences:

| Comparison | T2 difference [95%CI] | T3 difference [95%CI] |
|---|---|---|
| Global pilot minus no correction, global world | -.132067[-.161617,-.102881] | -2.377310[-2.572617,-2.185640] |
| Conditional minus global pilot, conditional world | -.029538[-.039783,-.020104] | -.046654[-.066469,-.025687] |
| Global pilot minus no correction, nonlinear world | -.961909[-1.254862,-.690589] | -2.920803[-3.820267,-2.172218] |

All12 single-condition pilot rank controls return missing-anchor. Cubic plugin
fails monotonicity in all three seeds; no resample failures or unsupported rows
are silently removed. Known nonlinear inverse proves recoverability when the
mapping is known; it does not mean the learned inverse succeeded.

Reference coverage by seed: T2 .895/.945/.910; T3 .955/.930/.900. Under the unchanged
Wilson95 criterion, T2 fails seeds915/917 and T3 fails917. Correct affine transport
restores these same values, not universal95 calibration. Mean interval width is
1.4914(T2),1.9193(T3) in original units. Those failures stay attached to the model;
they are neither new reader bias nor a reason to erase the transport identity.

## Why the target matters

T2 asks for the state apart from a person's intercept, whereas T3 includes that
intercept. A global offset can be partly absorbed by inferred a when estimating
state, but directly contaminates the future response level. This explains the
qualitative difference; it is a model-based interpretation, not a discovered
psychological causal mechanism.

The conditional map expands as

```text
z + (1+.25*z)*(a+b*z+s+e)
  = (a+.25*b) + (b+1+.25*a)*z + (1+.25*z)*(s+e), since z^2=1.
```

It mixes personal level/slope and changes state/noise amplitude by condition. A
single global scale/intercept cannot generally remove all three effects. This
provides a concrete production rule: preserve the condition and target definitions
when calibrating, rather than treating all apparent score change as personal change.

## Verification and artifact map

Canonical result: `results/grain_gptpro_foundation_v3/F16/dynamic_transport_full_r1/`.
Audit: `results/grain_gptpro_foundation_v3/F16/dynamic_transport_audit_r2/`.
Protocol: `docs/GRAIN_F16_DYNAMIC_TRANSPORT_PROTOCOL_V1.md`.
Implementation: `grain_research/gptpro_foundation_v3/f16_dynamic_transport.py`.
Runner/audit: `scripts/run_grain_gptpro_f16_dynamic_transport.py` and
`scripts/audit_grain_gptpro_f16_dynamic_transport.py`.

- 321 artifact hashes checked; separate causal convolution replays simulator states,
 observed prefixes and the future target without the generation recursion.
- 5,427 calibration coefficient vectors recomputed by QR rather than the runner's
 least-squares solver; all fit indices remain in the60-person pilot partition.
- All600 source components checked using independent augmented Kalman target
 recursion on fixed original/conditional/nonlinear trajectories and calibration
 draws. Aggregated target moments and all saved pilot-mixture collapses yield
 116,040 checked numbers, maximum discrepancy1.52e-13.
- 64 versus128 eta nodes checked at fixed first/last source and application
 objects across all supported arms/draws; maximum7.37e-10, below frozen.02 diagnostic.
 This is a selected numerical diagnostic, not a full-grid convergence theorem.
- All application per-seed metrics recomputed;42 focused tests and Ruff pass.
 Audit is same-agent alternate mathematics, not independent-human replication.

Audit r1 stopped at an incorrect independent-T3 folder locator. r2 uses the actual
`core_<seed>/bundle.json`; scientific runner/models/results were unchanged. Smoke
r1 code snapshot and complete full r1 are preserved.

Reproduce into new directories (no need to rerun routinely):

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. '../project persona/.venv/bin/python' scripts/run_grain_gptpro_f16_dynamic_transport.py --output results/grain_gptpro_foundation_v3/F16/dynamic_transport_replay_new --workers 3
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. '../project persona/.venv/bin/python' scripts/audit_grain_gptpro_f16_dynamic_transport.py --run results/grain_gptpro_foundation_v3/F16/dynamic_transport_replay_new --output results/grain_gptpro_foundation_v3/F16/dynamic_transport_audit_new
```

## Next decisions

1. Calibrate changed controlled-text content/qualification observation channels on
   independent pilot material. Numeric map inversion does not establish this.
2. Put the calibration law and its complete uncertainty into a public cold bundle;
   preserve versioned estimator/reference identity, not just a corrected point input.
3. Attach these finite results to original F12/F13/F16 requirements, retaining
   failure cells, then continue F01/F15/F10 and the full44+6 reconciliation. Do not
   restart the completed numeric matrix as a coverage-optimization branch.
