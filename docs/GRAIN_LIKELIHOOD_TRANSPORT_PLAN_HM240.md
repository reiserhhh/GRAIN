# HM240: source-supported target and joint-configuration updates

2026-10-13 JST. Freeze before new calculations. This is a finite construction
using existing numerical likelihoods, not a precision or predictive-accuracy
campaign. Previous turn was a read-only strategic discussion; no job was live.

## Scientific question and new use

HM209 already provides positive joint-profile posteriors, paired versus
independent source laws and local text-vector gradients. HM233/234 connect
source meaning and vector actions; HM239 separates actual author identity
from between-author target relationships. None licenses calling every joint
likelihood change a change in the five individual target readings.

Construct a numerical update capsule that keeps the complete 32-profile law
and separates its instantaneous five-mean response from its joint-configuration
response under the SAME source posterior. No LLM answer or application label
defines this split. An arbitrary Euclidean character truncation is not used.

Competing mechanisms:
1. Log evidence lies in the constant-plus-five-target statistic space. All
   update geometry is visible to the five local readings; one main-effect tilt
   reconstructs the complete endpoint law.
2. Evidence also changes joint configuration. That component is locally
   invisible to five means but may change their finite response through the
   evolving reference. Preserve it rather than delete it as noise.
3. Joint configuration changes while five means remain effectively unchanged.
   Report the configuration operation, not an extra named trait or accuracy.

## Fixed evidence and boundary

Only authenticated HM209 analysis.npz panel0_log_profile/panel1_log_profile
arrays are decompressed. Shapes 216 x 2 x 32, with fixed 24 authors x 9 frames,
two source banks and paired/marginal laws. Copy opaque author IDs from the
existing cold CSV only to verify ordering. Reuse all 432 paired-versus-product
updates and all 768 full-to-edit updates across the two laws. Full/full
relation updates (48 bank-author accounts) are primary. Other frames are
dependent observations, not 1200 people or new neural runs. Upstream 386-source
calibration and E5 encoding remain inherited; the Gaussian source channel is
a declared instrument, not an identified human generative law.

No raw text, new vectors, source fitting, application questionnaire, native
answer, private data, API, generation, training, Git or lockbox. No new
partition, kernel, covariance regularizer, rank, labels, seed or thresholds.

## Mathematics

For normalized positive endpoint laws p0,p1 let ell=log(p1/p0) and

    p_t(g) = p0(g) exp(t ell(g)) / Z(t),  0 <= t <= 1.

This exponential interpolation is a DECLARED numerical path, not a recovered
text/attention trajectory, chronological time or causal human change. With
five binary target statistics Y, set mu_t=E_t Y, C_t=Cov_t(Y), b_t=Cov_t(Y,ell).
Then d mu_t/dt=b_t. Positive mass on all 32 profiles makes C_t positive definite.

    ell_mean,t = (Y-mu_t)^T C_t^-1 b_t
    ell_joint,t = ell-E_t ell-ell_mean,t

The residual has zero weighted mean and zero covariance with all five Y.
Variance decomposes into b_t^T C_t^-1 b_t plus E_t ell_joint,t^2. These are
Fisher score energies of the DECLARED profile update, not fractions of real
personality, causal information or new measured factors.

Integrate with exactly 16 Gauss-Legendre nodes, no refinement/retry. Verify
integral b equals mu1-mu0 and total energy equals KL(p1||p0)+KL(p0||p1).
At each node retain all 26 higher Boolean-character derivative accounts,
including joint residual contributions. Target covariance and the projection
change with t; do not substitute one unconditional global frame.

Also apply the frozen START main and joint components separately as positive
finite tilts. Keep the two single-tilt responses, the full response and their
interaction. A locally zero response is not automatically a finite zero.
These tilts are numeric counterfactuals, not new valid texts or people.

## Ancestry and distinct contribution

HM82/S12 already use weighted projections and Fisher geometry; HM103/104
already have score-fiber controls and noncommuting constructed response paths.
HM177 and HM209 retain complete profile dependence. Do not rediscover those
principles. This packet adds their source-criterion posterior update capsule
on SAME natural evidence, with target-relative joint change explicitly usable
alongside rather than substituted for the five mean readings.

Nielsen2020, information geometry Sections3.3/3.6 and4.2:
https://franknielsen.github.io/entropy-22-01100-v2.pdf
Finite score/covariance/projection identities are derived directly here; no
new statistical theorem is claimed. OpenAI/math pinned adc7f124 three-state
02-foundations110-210 distinguishes a full posterior and valid observation
composition. It motivates preserving the complete law; its three-state/tree
reconstruction theorem is NOT transplanted to this instrument.

## Finite execution and acceptance

Synthetic tests before SOURCE outcomes: identity; additive tilt; pure pair
configuration with zero individual means; nonuniform-reference coupling;
coordinate recoding invariance; exact normalization/derivatives; invalid
support; ordered endpoint reversal and forward reconstruction.

prepare -> compute -> verify -> check once in an exclusive directory. Preserve
any failure and do not increase quadrature or tolerance. Save full endpoint
laws, path accounts, all 1200 update rows, all five signed finite responses,
all 26 joint accounts and complete group summaries. Literal per-profile sums
independently audit derivative orthogonality and energy without sharing einsum.
Arithmetic completion is not scientific promotion. The next decision is which
parts of a source likelihood update must be kept when constructing an author
measurement; no r/AUC/head-fidelity success criterion is used.
