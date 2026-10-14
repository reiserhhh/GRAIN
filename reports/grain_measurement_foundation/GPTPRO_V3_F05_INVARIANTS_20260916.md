# F05 Observation-Law and State-Coordinate Certificates

2026-09-16 JST. Original F05-01/03/06 mathematical requirements, computed from
the frozen two-state model. This is not full F05 acceptance or human evidence.

## 1. What Is Actually Preserved

For a stationary linear Gaussian model, x[t+1]=A x[t]+w[t], y[t]=C x[t]+e[t],
P=A P A'+Q. Output autocovariance is gamma[k]=C A^k P C' for k>0 and
gamma[0]=C P C'+R. Thus the full finite Gaussian observation law is determined
by mean and these covariance terms, not an arbitrary choice of latent axes.

Under invertible S, transport all objects:

    A_new = S A S^-1
    Q_new = S Q S'
    C_new = C S^-1
    P_new = S P S'

Both prescribed transformations were computed: rotation pi/5 and scaling
diag(2,.5). Full 64-observation covariance and the conditional last16 forecast
given the first48 are invariant. Maximum rotation error is 1.11e-16; scaling
errors are zero at stored precision. This is a state-coordinate equivalence,
not evidence that each arbitrary state axis has a psychological name.

## 2. An Additional Noise Ambiguity

There is more ambiguity than rotation/scaling in the unrestricted model.
Keep the original A and C fixed and choose symmetric D with D C'=0. Set

    P_new = P + D
    Q_new = Q + D - A D A'

Then P_new solves the stationary equation and all output autocovariances are
unchanged. The example D=.02 v v', v=(1,-1)/sqrt(2), yields

    Q_new = [[ .257975, -.00865],
             [-.00865,   .1691]]

instead of diag(.25,.16), while remaining positive definite. The full
observation covariance differs by only 1.39e-17.

This is not merely a coordinate change preserving A,C: the original (C,A) pair
is observable (rank2). If S commutes with A and C S^-1=C, observability forces
S=I. Yet Q_new differs. The same observed law can therefore conceal different
process-noise allocations even after holding these A,C coordinates fixed.

This construction requires a freely variable Q, including off-diagonal terms.
It is not valid evidence against identification under fixed or diagonal-only
noise assumptions. Assumptions must be part of a measurement model's contract.

## 3. Restrictions Must Be Evaluated Separately

Complex-step Jacobians of autocovariances at lags0..16 give these local ranks:

| Free parameters | Parameter count | Local rank |
|---|---:|---:|
| A, symmetric Q, C, R | 10 | 5 |
| A, R; fixed C,Q | 5 | 5 |
| A; fixed C,Q,R | 4 | 4 |

Singular values and the explicit relative rank tolerance 1e-9 are saved.
These results are local at the prescribed model. Full local rank is not global
uniqueness, robustness to measurement error or evidence of a psychological
construct. Conversely, unrestricted ambiguity must not be carried unchanged
into a restricted model where its transformations are no longer allowed.

## 4. Missing Time and Unit Roots

Twenty filter cases cover ten synthetic authors at gaps1 and3. Scalar Kalman
likelihood and terminal posterior agree with independent dense Gaussian
conditioning to 8.53e-14. Three-step prediction uses rho^3 and
Q*(1+rho^2+rho^4), not rho,Q.

At rho=1 with unknown initial state, adding d to intercept a and subtracting d
from initial state leaves every observation unchanged. At rho=.98 the same
change gives d*(1-rho^t), which is not zero: for d=1.7, t=63 it is 1.223907.
Near-unit-root finite-sample difficulty and exact nonidentification are different
claims. Source learning/application will quantify the former separately.

## Evidence and Next Work

Canonical outputs: `results/grain_gptpro_foundation_v3/F05/invariants_r1/`.
They contain full transformation matrices, rank Jacobians/singular values,
the alternative-Q certificate, twenty filter comparisons and source snapshots.

Code: `grain_research/gptpro_foundation_v3/f05_invariants.py`.
Runner: `scripts/run_grain_gptpro_f05_invariant_certificates.py`.
Six invariant tests plus nine existing dynamics tests pass (15 total).

No source models were refitted. The next primary execution remains the frozen
source r2 application comparisons, joint shared-shock experiment, missing-time
comparison and source-fixed likelihood profiles. Original task/DAG packaging
remains open; mathematical certificates do not substitute for those experiments.
