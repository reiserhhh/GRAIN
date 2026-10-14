# F16-R03 dynamic target transport

Frozen before application scoring, 2026-09-24. Extend the original F12 four
observation changes to the actual F05 core dynamic law, not static F12 records
renamed as a state. Keep T2=s_47 and T3=mean(y_48:51) under (+1,-1,+1,-1).

## Design and roles

Reuse all200 source fits in each of the three frozen F16 T2 bundles. Source
population/model is unchanged. Independently generate pilot100 and application200
per seed, with a~N(1,1.4), b~N(.2,.7), rho~Uniform(.2,.8), Q=R=.25, stationary
initial state and the core four-step alternating condition schedule. Prefix48,
future4 uses the specified plan directly. Simulator truth is written separately;
calibration sees paired observed values, not latent state or noiseless response.
Use distinct RNG namespaces for roles and source. No real participants are added.

Observation channels remain identity, 1+1.5*y, z+(1+.25*z)*y, y+.1*y^3.
Pilot first60 fits/last40 validates, whole-author bootstrap200. Compare no
calibration, global inverse, conditional inverse, known inverse; nonlinear adds
the original prescribed cubic inverse and its monotonicity/range rejection.
Retain missing-condition rank controls. All source-by-pilot combinations enter
the uncertainty; no successful-draw-only renormalization.

## Target-preserving computation, explicitly different integration

Old source Gaussian priors/noise laws remain frozen. This packet uses the already
tested conditional Gaussian integration principle: integrate (a,b,s) analytically
conditional on eta, and integrate eta by64-point Gauss-Hermite. This is a new
versioned estimator, NOT bitwise reproduction of old personal Laplace draws.
Source reference must be recomputed on the original300 source prefixes with this
same estimator; do not inherit old Laplace norms. No source parameter refit.

Compile each source/eta component into observed-prefix covariance C, target cross
covariance H and target variance V. For y=G*c (the calibrated observation vector):

    log p(y|eta) = -.5*((Gc-m)' C^-1 (Gc-m) + logdet C + n*log(2*pi))
    E[target|y,eta] = mu_target + H C^-1 (Gc-m)
    Cov[target|y,eta] = V - H C^-1 H'

At z=+/-1 every global/conditional affine inverse lies exactly in the four-column
basis [new, z*new, 1, z]. The cubic inverse uses [1,new,new^3]. This accelerates
all pilot resamples without changing observations, lowering source counts,
sampling only a diagonal source/pilot pairing, or refitting to application truth.
The fixed common temporal schedule is required for shared compiled kernels.
This finite sufficient computation is standard Gaussian algebra, not a new
universal four-dimensional text representation.

First integrate eta separately for each source/pilot/author, then average source
components without using the application likelihood to reweight source fits.
Finally average all200 pilot draws. Report conditional-state/future, personal
parameter (conditional a,b and eta), source and pilot covariance components.
T3 future innovations are shared across steps; use their joint covariance.
Keep cross-target covariance. Working normal intervals are not exact posteriors.

## Execution and verification

Freeze code/input hashes before scoring; checkpoint by seed/arm, resumable only
on identical hashes. Full200source/200pilot used, plugin included separately.
Exact unchanged observations may reuse scores only after checking numerical
equality; no blind arm aliasing. Keep all failed cells and partial artifacts.

Checks: direct augmented Kalman formula versus compiled dense Gaussian moments;
all corrected observations versus original fit_transport inverse;64 vs128 eta
nodes on fixed first/last pilot/application objects (truth not used for selection);
source norm freeze, full variance decomposition, complete application metrics and
paired bootstrap MSE differences. Selected old T2/T3 Laplace scores are comparison
estimators, not identities or the new norm. Report numerical disagreement instead
of changing tolerances or hiding it. Arithmetic checks1e-9; GH diagnostic differences
above.02 score or variance trigger an explicit numerical follow-up, not scientific
promotion. Coverage retains original Wilson95 gate, with error/width separately.

Artifacts: simulator/paired observations, separate truth, calibrators and statuses,
pilot validation and missing-anchor certificates, source norms, plugin/mixture
scores and components, support/coverage/metrics, hashes and report. This closes
only numeric dynamic transport if verified. Changed TEXT_SYNTHETIC reader channels,
public calibration-mixture serialization and original44+6 remain separate work.
No API, neural training, MEPS/private raw, remote job, lockbox or publication.
