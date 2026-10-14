# Endpoint Exchange, Observation Coherence, and Paired Design

2026-10-08. HM20 finite packet executed and alternate-arithmetic verified.
This is a controlled reader-action theory companion, not a claim that a new
human personality dimension or a general language symmetry was discovered.

## 1. Coordinates and the exact involution

Keep the original source origin o and two positive scales d,s. For actual
reference/target observations r,t, set x=(t-r)/d,c=(r-o)/s,kappa=d/s and
m=c+kappa*x/2. Exchanging endpoints is

\[
R(x,c)=(-x,c+\kappa x),\quad R^2=I,\quad m^R=m.
\]

This is not reversing token order, changing the natural recipient, or
changing the prompt. Those operations alter different objects.
Let U=sym(xc^T),V=kappa*xx^T/2,W=U+V,H=skew(xc^T). Direct substitution gives

\[
U^R=-U-2V,\quad V^R=V,\quad W^R=-W,\quad H^R=-H.
\]

The scalar/function-valued projectors E=(I+R)/2,O=(I-R)/2 obey E^2=E,
O^2=O,EO=0,E+O=I. Applied componentwise to U, they give EU=-V,OU=W.
For the balanced orbit probability measure, an even function and an odd
function are orthogonal. This says nothing about their Frobenius inner
product as ambient matrices, nor about an unbalanced observed population.

The abstract lifted pair has rho=[[-1,-2],[0,1]]. It is an involution but
not Euclidean orthogonal. Averaging gives G=(I+rho^T rho)/2=[[1,1],[1,3]],
with rho^T G rho=G. This metric belongs to the lifted pair representation;
it must not be substituted for the original hidden-state metric.
Finite-group averaging is standard representation theory, not a new theorem
claimed here. See [Etingof et al., sections 3.1 and 3.6](https://math.mit.edu/~etingof/replect.pdf).

## 2. Observation span and numerical rank

The invertible transformation (U,V)->(U,-U-2V) proves the exact span identity
span{U,U^R}=span{U,V}. Virtual exchange does not create independent samples.
Recovering UV and VV from the orbit Gram gives maximum errors 2.22e-16 and
4.44e-16 on the frozen source data.

| Source design | U | V | W | Normalized U/V stack | Exchange orbit |
|---|---:|---:|---:|---:|---:|
| All 360 states | 182 | 222 | 264 | 395 | 396 |
| Full-property 64 states | 64 | 64 | 64 | 128 | 128 |

Both source measures give these effective ranks. The 395/396 difference is
not a new direction: nonorthogonal row mixing changes singular values near
the frozen numerical cutoff. Uniform U/V stack's largest discarded value
is 1.31197e-11 against tolerance 1.59872e-11. In HM19 the stack used twice
the total weight; HM20's balanced stack uses probability mass one. Preserve
both numerical results and the exact span identity rather than tuning a
threshold to force agreement. None of these ranks is a psychological factor count.

## 3. The even defect and what reversal cannot identify

For the fixed receiver approximation f=x^T b+x^T A c, write A=S+H with
S symmetric and H skew. Explicitly evaluating both endpoints gives

\[
Ef=-\frac\kappa2 x^T Sx=-\langle S,V\rangle,
\qquad Of=x^T b+x^T A m=f+\langle S,V\rangle.
\]

The separately source-fitted endpoint law x^T b+x^T S m is already odd.
Its coefficients are not the receiver coefficients with a term deleted.
For fixed kappa>0, it is the endpoint difference of
F(c)=b^T c/kappa+c^T S c/(2*kappa).

An odd skew action x^T Hm need not be a scalar potential. A three-node
triangle can have nonzero circulation even though every edge reverses sign.
Thus reversal checks coherence, not conservativity or psychological meaning.

For a deterministic native scalar readout in one fixed complete context
gamma, y=F_gamma(t)-F_gamma(r) reverses sign by definition. Its inferred
reverse -y is not a separately executed reverse intervention. Native even
action is zero under that declared readout; the compiled even defect belongs
to the approximation. This does not justify deleting even human behavior
or claiming human response processes are conservative.

## 4. Actual observations and the remaining gap

Inventory retains all2172 query addresses/724 states/7 arms and original
source1080/validation528/application564. It indexes the complete input IDs,
hook layer, assigned span/positions and exact assigned-span endpoint hashes,
not merely compressed observations. There are 312 exact no-op rows and zero
nontrivial independently addressed inverse edges in this inherited battery.
All2556 consumed native records were hash-checked.

For application full-property, receiver even RMS is 1.618%/0.686%/0.867% of
forward RMS for NA/low/trust. Removing it slightly worsens native-action
RMSE in all three targets. On token-reversal arms the effect differs by
target; preserve these results rather than automatically promoting symmetry.

NA/low native covariance is +8.24505e-7. Receiver compiled covariance moves
from -9.61594e-7 to -9.39939e-7 under odd projection: joint gap decreases
1.2124%, but the sign mismatch remains. Residual covariance stays about
2.28309e-6. Neither virtual balancing nor antisymmetry resolves that joint gap.

Shuffled-context reversal transports the fixed virtual start by kappa*x.
It is not the original shuffled association rerun as a native reverse.
Common real-context diagnostics are separate saved columns.

## 5. Execution and boundaries

Canonical endpoint_exchange_orbit_r1 and verification_r1:20 spectra,
6516 component rows,216 statistics,432 covariance rows,2172 inventory rows.
Alternate explicit dyads/full-ambient covectors verify all rows; Gram error
2.23e-13, action error1.08e-16, native address/score error0. 249 focused
tests across26 files pass. These are same-agent alternate calculations,
not an independent human replication or full-repository test claim.
All code/config/input/result/verifier freezes remain immutable.

No new fits, native forwards, API, private texts, training or publishing.
Original18 protocols/100 cards/44 computational questions, natural authors,
individual response B_u and psychological interpretation remain in scope.
Next question: which part of the native joint gap is already present in
logit displacement, and which comes from the nonlinear categorical readout?
Reuse the existing HM10 softmax integral; do not rediscover or retrain it.
