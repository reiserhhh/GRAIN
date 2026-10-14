# Source-Hidden Compensation and Observation Transport

2026-10-09JST. HM40finite experiment and HM40Sresult-after sensitivity are
complete. Original measurement-foundation scope and the active goal remain open.

## 1. What the computation establishes

Directions almost invisible to the SOURCE total derivative can retain large,
opposing link/decoder components and carry total action in APPLICATION.
For negative affect, the source-only compensation family reduces uncovered
action mean-square by54.11%/33.53%, compared with+.0084%/-.3128%for matched
random controls. A source-fixed30dimensional sensitivity retains53.55%/29.88%.
These are actions of the frozen reader, not human personality accuracy.

Thus the object to transport is sometimes a joint component family, not just
its summed source derivative. The result is target-dependent and signed:
low mood is weak and changes sign under resolution sensitivity; trust also
changes under the unresolved tail. No universal noise-removal rule follows.

## 2. Exact mathematical object

For the original score F=m(l), let v_L and v_D be HM39's physical link and
decoder mixed covectors, and v_R the recorded numerical roundoff. Their
reference total is v=v_L+v_D+v_R. The Euclidean metric and ordered token
coordinates are fixed separately for each query; different query lengths
are never concatenated.

With old source-active basis U, P_U=UU^T:

```text
r_L=(I-P_U)v_L, r_D=(I-P_U)v_D
c=(r_L-r_D)/2, e=(r_L+r_D)/2
r_L=c+e, r_D=-c+e
(I-P_U)v = 2e + (I-P_U)v_R.
```

The exact sum map T:E* x E* -> E*, T(l,d)=l+d has kernel {(u,-u)}.
For linear component transports G_L/G_D, a total-only transport exists on
the full product domain exactly when the transport respects this kernel:

```text
T diag(G_L,G_D)(u,-u) = (G_L-G_D)u = 0 for every u.
```

Restricted data may require equality only on the observed kernel subspace.
More generally a nonlinear post-map factors through T exactly when it is
constant on each Tfiber. This is a standard quotient/factor-through fact,
not a new theorem or a fitted universal neural transport. Tests contain
equal/unequal linear transports and a nonlinear same-total counterexample.

## 3. Source construction and cold evaluation

Fixed HM36U has ranks87/88/86. Construct one W from32SOURCE ccolumns/query
(16midpoints x2physical arms), after projection against U. Column normalization,
original100eps Gram rank criterion, stable thinQR/smallSVD retain32columns.
One seeded rank32random complement/query is frozen simultaneously.

Source48records and all three W/random families are disk-sealed before
application arrays are loaded. No application singular vectors, labels,
rank selection, neural calls or regularization fitting enter construction.

For each original total covector v and displacement dP:

```text
A=<v,dP> = A_U + A_W + A_remaining
A_U=<P_Uv,dP>, A_W=<P_Wv,dP>
MSE(missing_U)-MSE(remaining)
  = E[A_W^2] + 2E[A_W A_remaining].
```

Orthogonality of vector spaces does not force the scalar cross term to zero.
Increasing covector energy coverage need not decrease the signed action
remainder. Keep input energy, covector energy and action error separate.

APPLICATION W captures .135%-.272%of total covector energy, versus random
.000254%-.000664%. Old U already captures97.93%-98.86%. The additional
physical directions are small by total energy but can matter to their
paired inputs, especially negative affect. This is not an additive
percentage of psychological information.

## 4. Resolution sensitivity changes the interpretation

The recorded HM39roundoff-complement Frobenius norms are1.412e-8/1.456e-8/
1.530e-8. Their normalized proxies are6.456e-5/1.100e-4/9.232e-5. Two of
32normalized singular directions/query fall below this proxy. The proxy
is not a certified bound on each component's neural error, and1e-8coordinate
comparison is not a whole-vector error guarantee.

HM40Swas planned after seeing HM40, explicitly result-after. It fixes a single
30dimensional prefix from SOURCE labels before application evaluation,
using the existing matched random prefix. It does not refit a basis or pick
a cutoff that maximizes application results. Full32results remain primary.

| Target/arm | Full32missing-square reduction | Source-fixed30reduction | Random30reduction |
|---|---:|---:|---:|
| Negative affect/U | +54.11% | +53.55% | +.030% |
| Negative affect/M | +33.53% | +29.88% | -.318% |
| Low mood/U | +3.01% | -1.83% | +.085% |
| Low mood/M | +2.51% | -1.15% | -.766% |
| AItrust/U | -9.55% | +5.45% | -.411% |
| AItrust/M | +.81% | +14.08% | -.095% |

Negative-affect manifestation survives this sensitivity. Trust's signed
effect and low-mood's small gains depend on the numerical tail. Do not
declare32the best rank, promote30as a deployable predictor, or delete
decoder/link components wholesale. This is a mechanism-resolution issue.

## 5. Literature alignment and reading boundary

MIT18.952notes section1.2, proposition1.2.1 and exercises (PDFpages7-11)
give the standard factor-through criterion and the distinction between
quotient and dual space. These are mature mathematical foundations.
[Primary notes](https://math.mit.edu/classes/18.952/spring2011/chapter1.pdf).

OpenAI/math, fixedcommit adc7f1241b42e322a6451854ab7e4b4c146bf78a,
expansion.tex371lines read in full: rectangular joint tests, vanishing
filtrations and reduction of the whole integral subspace motivate examining
an observation family instead of one cancelling sum. Formal-series,
rank-expander and noncommutative-formula assumptions do not hold here.
The manuscript's theorem has not been applied to neural vectors or
independently verified as a new mathematical proof by this packet.
[Fixed manuscript section](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Polynomial-Hitting-Lists-for-Noncommutative-Rational-Formulas-September-24-2026/build/sections/expansion.tex).

## 6. Verification and next scientific decision

Primary complete:96midpoints/192addressed arm rows/192pair summaries/
36moment groups/12coverage groups,3source families and3random controls.
Direct tallSVD, independently reconstructed projection vectors and
compensated scalar moments replay every table; maximum3.16220e-11.
13output hashes/10411ancestry-code hashes checked,0new model calls.
Primary analysis22.77seconds plus verification10.63seconds.

HM40Sfull-result replay plus alternate prefix projections maximum3.45279e-14,
21.38seconds. Combined54.79/300CPUseconds.16newtests, full68-file math
battery609passed4.99seconds; implementation Ruff/format passes. This is
same-agent alternate computation, not independent researcher replication.

Next HM41must separate state-conditioned numerical link curvature C from
the physical logit Jacobians J_H/J_P, preserving their joint source pairing.
Use a fixed three-block substitution/Mobius decomposition, not another
rank/regularization grid. Actual application derivatives remain diagnostic
references, and fully source-only candidates must be separated from oracle
substitutions. No full-path, new-person or psychological-validity claim.

Results: results/grain_math_response_v1/cancellation_transport_r1;
verification: cancellation_transport_verification_r1;
sensitivity: compensation_resolution_sensitivity_r1.
No API/privateMEPS/Claude/automation/training/commit/push/tag/lockbox/oldqueue.
