# GRAIN: a shared origin controls sensitivity to author differences

2026-09-10, NEXT56/pass45. [Report](../reports/grain_vector_mechanisms/SOURCE_OFFSET_MECHANISM_REPORT_V1.md).

## 1. The ingredient is identified

For source coefficient B and explicit unit-feature coefficient b:

    Psi(X)=X+1b^T+V(X),  V(X)=Z(X)B-1b^T.

b is identical across new8 authors/two queries; source4 uses its excluded-source
reference. This is an exact parameter decomposition, not a textual noise class.
Offset-only almost reproduces the full-map gradient (cosine .9998), while
non-intercept-only almost preserves identity's gradient (cosine .9993). Thus the
earlier reader relocation is linked to a specific source ingredient, not merely
an unspecified mean shift or incorrect tangent. Query-contrast r remains .811
without b, vs .051 with b alone. Source4 contrast remains negative.

## 2. A constant can change individual differences

    D(X+b)[D_f]=D_f
    D[S(X+b)][D_f]=<grad S(X+b),D_f>.

For two authors and common b, when S is differentiable on the path:

    S(X_u+b)-S(X_v+b)
      = integral_0^1 <grad S(X_v+b+t(X_u-X_v)), X_u-X_v> dt.

b has zero between-author variance in this fixed reference, yet changes the
operator inside the integral. A shared condition can govern sensitivity to
individual departures. This connects to the user's condition/identity intuition;
it does not identify b as a topic, situation, clinical state or environmental cause.

RMS normalization gives one relevant architectural example. For one token of
width h, rho(x)=sqrt(||x||^2/h+epsilon), fixed gain gamma:

    N(x)=diag(gamma)x/rho(x)
    DN(x)[d]=diag(gamma)[d/rho(x)-x(x^T d)/(h rho(x)^3)].

x+b changes both terms although d is unchanged. RMSNorm does not subtract the
mean and does not generally remove translations.
[Zhang & Sennrich (2019)](https://arxiv.org/abs/1910.07467).
NEXT56 measures the complete suffix, not a unique attribution to normalization;
attention, nonlinear MLPs and residual updates also participate.

## 3. Energy and functional role differ

    E||b+V||^2=||b||^2+E||V||^2+2<b,E V>.

Here offset energy/total is~99.91%, remainder~.046%, cross small and positive.
Yet the small remainder improves new8 query-contrast response r .774->.811 and
error .40394->.35069. Large common energy improves score origin but suppresses
response amplitude; neither component alone gives a correct complete score.
Do not equate common with noise, or high energy with psychological importance.

For output a, either L or tau, exact symmetric ingredient effects are

    A_b=[(a_O-a_I)+(a_F-a_V)]/2
    A_V=[(a_V-a_I)+(a_F-a_O)]/2
    a_F-a_I=A_b+A_V; interaction=a_F-a_O-a_V+a_I.

The state formula is additive; the nonlinear output is not. Signed projections
can be negative or exceed1. They are calculation accounts, not psychological
variance fractions or unique causal mediation.

## 4. Change the approximation object

The state-fit family has answered its mechanism question. V preserves contrast
but not native L. The next mainline move is therefore:

    before: approximate X22, then run the native suffix S
    next: approximate L(X16) and D L(X16)[D_f] with one numerical function.

The latter need not reconstruct every hidden coordinate for every downstream
operation. This is functional substitution rather than only representation
matching, following [Csiszarik et al. (2021)](https://arxiv.org/abs/2110.14633).

GRAIN supplies the perturbation family; source-only original L/tau teach one
explicit function. This applies the value-plus-derivative principle of
[Sobolev training](https://arxiv.org/abs/1706.04859). NEXT53 used that idea on
hidden states; NEXT57 uses it on the complete scalar objective. E5 residuals or
conditioned baseline B are different older targets, not completed whole-L extraction.

H16 remains a neural encoder dependency. Psychometric scoring, old criterion
links and evidence-bound interpretation stay unchanged. Model fidelity is
separate from human Y linkage. [NEXT57](GRAIN_COMPLETE_FUNCTION_EXTRACTION_PLAN_V1.md)
fixes one family/two controls and makes suffix removal the next decision.
