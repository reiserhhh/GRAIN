# Intervention-Preserving Observation Units

2026-10-08. HM11 result-guided theory update. This is a standard factorization
criterion applied to GRAIN observations, not a new theorem or a complete
psychological measurement model. Retain HM10B3 and HM11 frozen artifacts.

## 1. Three Different Objects

Let H denote the token-state matrix at a declared decoder boundary. Let c
denote everything held fixed in a local intervention: the receiver, all other
token states, and the original question-conditioned states at that boundary.
For query q, the unchanged downstream network gives F_q(H;c).

A compression tau(H) and a fitted reader f_q(tau(H);c) are separate objects.
The carrier displacement d found by a same-word owner swap is not the reader's
covector D_H F_q. A large or interpretable displacement does not establish a
complete inverse from that displacement to the target. HM11 uses no question
endpoint features; it therefore also has no automatic access to all of c.

GRAIN's author evidence/eligibility ledger remains upstream of aggregation.
None of these native query functions turns a colleague's report into the
author's experience or supplies unobserved human personality truth.

## 2. Exact Factorization Criterion

For a fixed q,c and declared domain D, an unrestricted scalar f exists such
that F_q(H;c)=f(tau(H);c) for all H in D if and only if F_q is constant on
every fiber of tau within D:

\[
\tau(H_1)=\tau(H_2)\Longrightarrow F_q(H_1;c)=F_q(H_2;c).
\]

Necessity follows by substitution. For sufficiency, assign each compressed
value the common F_q value of its fiber. This proves a set-function result,
not smoothness, sample generalization or learnability from a finite source.

If a fiber has output diameter d, every single-valued approximation has
maximum endpoint error at least d/2. A midpoint constant attains this bound
for two scalar endpoints. More polynomial degrees cannot undo a genuine
fiber collision; the observation unit must retain the missing distinction.

## 3. Rounded Coordinates and Approximate Reading

Actual decoder interventions use float32. Their compressed inputs can differ
by epsilon rather than being exactly equal. If a candidate f is L-Lipschitz
on the segment joining the two inputs, the triangle inequality gives

\[
\max(|F_1-f_1|,|F_2-f_2|)
\geq \frac{(|F_1-F_2|-L\varepsilon)_+}{2}.
\]

For the fixed linear functions, L is the raw coefficient norm. For the fixed
degree-two functions at unchanged order, the gradient is affine along the
segment, so the maximum of the two endpoint gradient norms bounds its norm
along that segment. The exact trapezoidal gradient integral checks this
finite-polynomial statement independently.

The actual HM11 companion checks this bound for four fitted functions. It
does not rule out arbitrary discontinuous functions distinguishing rounding
bits. Nor did it run the neural reader at unrounded ideal endpoints. Keep
the exact algebraic criterion and actual bounded-L evidence separate.

## 4. Two Compression Stages, Two Kernels

For a property span with T tokens, write its donor displacement as

\[
d_j=\bar d+r_j,\qquad \bar d=T^{-1}\sum_jd_j,\qquad\sum_jr_j=0.
\]

Token averaging loses r. Projecting the mean onto an orthonormal source
frame U also loses directions orthogonal to U. These losses are distinct.
The current twelve-coordinate view combines both and must not be treated
as a lossless summary merely because local action was found in U.

For P=UU^T, the ideal full and projected interventions have the same retained
coordinates; the complement and baseline have the same retained coordinates.
HM11 reconstructs the original float32 assignments before measuring the
small actual coordinate gaps. A response to a discarded direction is not
automatically a psychological nuisance; it is missing observation information.

Under differentiability, exact local factorization requires
ker(D tau) subset ker(D F_q) for every admitted q,c. This is necessary, not
globally sufficient. Sufficiency additionally needs compatible connected
fibers and conditions ensuring F_q is constant along them. Mean owner
directions alone do not check this derivative condition.

## 5. Conditional Residual Is a Function, Not a Deleted Error

For retained and complementary coordinates p,r, the elementary fundamental
theorem of calculus gives, where F_q is continuously differentiable,

\[
F_q(p,r;c)=F_q(p,0;c)+\int_0^1D_rF_q(p,tr;c)[r]dt.
\]

The second term can carry target-dependent and context-dependent action.
Mixed p-r terms explain why orthogonal input components need not yield
additive output scores. HM10B3 measured finite mixed responses; it did not
estimate this whole conditional derivative field or a person's response B_u.

The next question is whether the missing action is in the mean's vector
complement, within-span token organization, or interactions with the fixed
receiver. Answer that before increasing model degree or deleting residuals.

## 6. Evidence and Literature Boundary

HM11: source-only four fixed functions, 224 old controlled documents,
112 source/56 validation/56 application; zero new model calls. Six body spans
projected into the source rank2 frame at original block16. Application NA
level r=.6522 for all-span reading, but many trust owner contrasts reverse.
In 32 application pairs per query, full/projection coordinate gap <=1.78e-7,
trust output gap up to .010884. These are reader-score units, not trait r.

This connects with intervention-based causal abstraction, which distinguishes
input-output fit from agreement under interventions. See
[Geiger et al. 2021](https://papers.nips.cc/paper/2021/file/4f5c422f4d49a5a807eda27434231040-Paper.pdf)
sections 1-2 and
[Causal Abstraction for Mechanistic Interpretability](https://jmlr.org/papers/volume26/23-0058/23-0058.pdf)
sections 2.4-2.5 and 3.1-3.2. These selected sections were read, not the entire
64-page proof battery. GRAIN does not inherit a constructive-abstraction
certificate from those papers or implement their full intervention algebra.

Pinned OpenAI/math's conditional-sweep setup and commutant introduction
inspired keeping the conditioning information and a family of readouts.
Their independent-coin, infinite-dimensional and commuting-operator
assumptions are not properties proved for this decoder. No manuscript's
strong theorem is imported as a personality-identifiability result.

## 7. Place in the Psychometric Chain

Raw text -> typed observation/author eligibility -> token vectors -> source
observation unit and conditional target reader -> event/history account ->
author choice/condition/state profile -> behavioral and external meaning.

HM11 advances the observation-unit step by locating a concrete information
loss. It does not replace the author/history/transport steps, shrink the
original 18/100/44 scope, or turn teacher-score fidelity into personality
measurement validity. Retain partial reading functions without production
promotion, and use the new mean-vs-within-span experiment to change the unit.
