# Token-Space Symmetry and Observation Design

2026-10-08. Result-guided continuation of HM11 and the fixed HM12 design.
Standard finite-group and projection facts below are derived explicitly;
they are not new theorems or assumptions that all text tokens are exchangeable.

## 1. Token Space Is a Separate Axis

For a declared span, let H be a T-by-d matrix. A permutation of the T token
slots acts by H -> P_pi H. Define

\[
J_T=\frac{1}{T}\mathbf1\mathbf1^\top
=\frac{1}{T!}\sum_{\pi\in S_T}P_\pi,\qquad C_T=I-J_T.
\]

Every matrix entry of the average is 1/T because each original slot reaches
each destination in (T-1)! permutations. Thus J_T is the finite-group
averaging projection, with J_T^2=J_T, C_T^2=C_T, J_TC_T=0. Its image consists
of constant token rows; its kernel consists of row-sum-zero matrices.
Both subspaces are preserved by every P_pi.

For donor displacement D, M=J_TD and R=C_TD satisfy

\[
D=M+R,\quad \langle M,R\rangle_F=0,\quad
\|D\|_F^2=\|M\|_F^2+\|R\|_F^2.
\]

This decomposition concerns token positions, not the separate d-dimensional
source-vector projection U used in HM10B3/HM11. More mean directions in d
cannot recover R erased by J_T. Conversely R can be irrelevant for one
specific reader and important for another; geometry alone does not decide.

## 2. Averaging Input and Averaging Responses Differ

The actual downstream reader F_q(H;c) has fixed receiver/context/query
states c. It need not be invariant under token permutations: positions,
causal attention and neighboring states remain in the computation.

In general,

\[
F_q(J_TH;c)\ne\frac1{T!}\sum_\pi F_q(P_\pi H;c).
\]

Even if F is permutation-invariant, it need not factor through the raw mean.
For example F(H)=||C_TH||_F^2 is invariant and nonzero on many equal-mean
inputs. The orbit of token permutations is much smaller than an entire
fiber of the mean map. These are different equivalence relations.

The [Deep Sets](https://arxiv.org/html/1703.06114v3) construction uses
transformations phi before summation. It does not assert sufficiency of
the raw mean (phi=identity) for arbitrary readers. Its fixed-size scalar
moment proof uses enough power sums to reconstruct a multiset; that is not
a theorem that two moments reconstruct a 1536-dimensional token sequence.

[Janossy pooling](https://arxiv.org/html/1811.01900v3) averages a
permutation-sensitive function over orderings and separates higher-order
dependencies from unary pooling. It motivates testing relational summaries
before forcing invariance, not using unordered pooling on every text input.
We do not reproduce its full architecture or infer exchangeability.

## 3. Finite Action Decomposition

At one fixed receiver H,c, HM12 measures four outputs:

\[
F_0=F(H;c),\quad F_M=F(H+M;c),\quad
F_R=F(H+R;c),\quad F_D=F(H+D;c).
\]

Their action identity is exact as a scalar definition:

\[
F_D-F_0=(F_M-F_0)+(F_R-F_0)+I_q,
\quad I_q=F_D-F_M-F_R+F_0.
\]

I_q can cancel or amplify either component; squared vector orthogonality
does not imply additive output. If the ideal reader is C2 on the rectangle,
the fundamental theorem of calculus additionally gives

\[
I_q=\int_0^1\int_0^1
D^2F_q(H+sM+tR;c)[M,R]\,ds\,dt.
\]

HM12's actual endpoints are rounded to float32. The saved scalar identity
holds exactly for those outputs. The smooth integral describes ideal
unrounded corners; we do not report it as a measured derivative or equate
the rounded full corner with the sum of two separately rounded corners.

For T=1, R=0 and the actual centered assignment is unchanged. This is a
live no-op control. It cannot supply evidence that multi-token organization
is unimportant. All token-count opportunities must be retained separately.

## 4. Next Representation Decision

If a nonzero centered action survives near-zero full-dimensional mean shift,
the next observation must retain more than a point: token dispersion,
position-conditioned structure, or typed relations. That finding alone
does not choose among them.

A useful subsequent distinction is between an unordered token configuration
and its placement: row permutations preserve the mean, covariance and all
unordered moments, but not necessarily the downstream response. If such a
permutation changes F, no summary depending only on that multiset can be an
exact observation of this fixed reader on these intervened states. This
does not claim that natural paraphrases or human personality change.

If centered action is small and the mean intervention retains full action,
expand source-supported mean directions instead, while keeping the
upstream event/actor/mode ledger. Do not keep increasing polynomial degree
on an observation that fails its own intervention test.

## 5. OpenAI Math Reading Boundary

The pinned `Expected-amenable-subalgebras-preserving-core-commutants` manuscript
was read at `build/paper.tex` lines1-210 (abstract, introduction, theorem
statements and proof overview). It emphasizes preserving constraints in
the whole ambient object rather than a smaller comparison algebra. This
is a design analogy only. Its von Neumann algebra/modular expectation,
amenability and continuous-core claims are not proved properties of GRAIN;
no bicentralizer theorem or new decoder formula is imported from it.

The executable finite averaging formula above follows from counting
permutations, not from that manuscript's unverified strong theorem.
Reading a new advanced topic is useful only when it changes a concrete
observation/intervention decision; mathematical names are not substitutes
for actual token-state experiments.

## 6. Psychometric Position

This concerns the text-vector observation step before author aggregation.
Observed author eligibility, opportunity, history, condition response and
state must still remain explicit. A native narrow-query action is not a
Big5/MBTI score, psychological factor name or human accuracy result.
It can nevertheless locate an information loss that a later text-based
measurement method must handle.

## 7. HM13 Actual Placement Result

The exact multiset test is now completed:252 actual forwards,36 one-token
noops output difference0. Original and centered-only property matrices
were reversed with all other receiver/query states held at their original
block16 inputs. Full1536-dimensional means were exactly equal and every
row multiset was byte-exact, yet trust output changed by up to.009014.

Therefore an untagged set statistic cannot exactly preserve these actions;
the largest checked two-endpoint approximation lower bound is.004507.
This rules out unordered summaries on the tested fixed-context endpoints,
not all set/graph methods or all personality models. Position-tagged sets
and typed edges can retain order while remaining invariant to *joint*
coordinate relabeling. The next ordered observation/probe formulation is
`GRAIN_POSITION_CONDITIONED_OBSERVATION_V1.md`; bounded execution plan is
`GRAIN_ORDERED_MOMENT_ACTION_PLAN_V1.md`. The latter is planned, not run.
