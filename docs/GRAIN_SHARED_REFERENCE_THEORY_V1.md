# Shared SOURCE references and comparison functionals

2026-10-13 JST. HM231 finite packet completed. This is a new project construction
using standard coupling mathematics, not a new probability theorem, personality
accuracy improvement, or completed psychological measurement theory.

## 1. A missing part of the observation contract

For an author's existing SOURCE masses w_ui and joint five-anchor rows y_i,
HM229 declares F_u=sum_i w_ui delta_y_i. It does not identify a joint law for
two authors' SOURCE draws. Independent J_uv(i,j)=w_ui*w_vj is one contract.
Different contracts may preserve each author's full F_u, including all target
CDFs, means and covariance, yet supply different pair comparisons.

HM143 already demonstrates marginal/joint identification issues in anonymous
coordinate bridges. Here the contribution is an independently runnable,
SOURCE-supported comparison family on the SAME natural-author numeric material.
No natural-human causal mechanism is identified by declaring this family.

## 2. Positive, global construction

Use common E_i~Exp(1), independent across SOURCE identities i. Define for ALL
authors at once I_u(E)=argmin_i E_i/w_ui, then Y_u(E)=y_(I_u(E)). Each marginal
identity law is w_u and all five anchors travel together. This supplies an
actual joint law for any collection of authors, not pairwise couplings that may
be incompatible on triples. SOURCE row permutation merely permutes iid E.
No ordering of anchor values, target-by-target pairing, fitting or temperature
search is needed. It is equivalent to the shared Gumbel-max construction.

The SOURCE anchors make this a declared observation of existing scale values.
They do not turn the common exogenous race into a real author's latent psychology.
Nor is Gumbel-max generally the maximal-agreement coupling: optimization of
agreement is deliberately NOT this construction's objective.

## 3. Exact finite integration

For p=w_u,q=w_v,r_i=p_i/q_i, S(t)=sum_k max(p_k,t*q_k):

    J_ii=p_i/S(r_i)
    J_ij=p_i*q_j*integral_(r_j)^(r_i) S(t)^(-2)dt  if r_i>r_j; otherwise0.

Derivation: write the two winning times A=E_i/p_i, B=E_j/q_j. If i wins p and
j wins q, their ratio t=B/A must satisfy r_j<t<r_i. Every other E_k is at least
A*max(p_k,t*q_k). The density and change of variables produce
p_i*q_j*integral A*exp(-A*S(t))dA dt; integrate A to get S(t)^(-2).
For the same winner i, the ordinary exponential-race diagonal gives p_i/S(r_i).
Between sorted breakpoints a,b, S is affine; the integral is (b-a)/(S(a)*S(b)).
This is integration of the known coupling, not Monte Carlo or an inferred
natural posterior. Strictly positive SOURCE masses are required here; invalid
or zero caller masses are rejected, not smoothed or repaired.

Define C_q(i,j)=sign(y_iq-y_jq). The signed comparison is
d_q=sum_ij J_ij*C_q(i,j), and Pi_q=(1+d_q)/2 with half credit for ties.
The same-SOURCE probability is trace(J); conditional comparisons on changed
SOURCE identity remain separate outputs, undefined for identical laws.
Their possible amplification is conditioning, NOT increased psychological evidence.

## 4. What dependence can and cannot change

For D=J-p*q^T, D*1=0 and D^T*1=0. Thus every additive contrast is invariant:

    <D,phi(i)-phi(j)> = 0.

The five expected scale differences are exactly p^T*y-q^T*y under BOTH mechanisms.
Nonlinear comparisons can change only through the non-additive part of their
two-argument observation function. This follows from ordinary marginal algebra;
it is not a discovery of new human dimensions. Both shared-reference dependence
and all signed cross-target covariance are retained, not deleted as noise.

The reference-draw contrast variance is

    Var_J(Y_uq-Y_vq)=Var_Fu(Y_q)+Var_Fv(Y_q)-2Cov_J(Y_uq,Y_vq).

This is variance of the declared joint SOURCE draw. It is NOT a sampling CI,
measurement-error variance, trait/state decomposition or improved reliability.

## 5. Post-outcome local expansion explains the actual near-invariance

Let b_i>0 be fixed, p_i=b_i+epsilon*a_i,q_i=b_i+epsilon*c_i with zero-sum
a,c and sufficiently small positive epsilon. Write alpha_i=(a_i-c_i)/b_i.
Then r_i=1+epsilon*alpha_i+O(epsilon^2), and S(t)=1+O(epsilon) on these intervals.
For i!=j:

    J_shared(i,j)=epsilon*b_i*b_j*(alpha_i-alpha_j)_+ +O(epsilon^2)
    J_shared(i,j)-J_shared(j,i)
       =epsilon*b_i*b_j*(alpha_i-alpha_j)+O(epsilon^2)
       =p_i*q_j-p_j*q_i+O(epsilon^2).

Therefore ANY fixed bounded antisymmetric comparison function, including C_q,
has the SAME first-order signed response under shared and independent races.
Meanwhile at epsilon0 shared draws are identical and contrast variance is0,
whereas independent contrast variance is2Var_b(y_q). Shared contrast variance
is O(epsilon), so strong dependence can change dispersion substantially while
leaving directional comparisons nearly unchanged. This is a local algebraic
explanation, not a uniform empirical bound for every actual author or kernel.
It was derived AFTER primary outcomes and is labeled accordingly; no epsilon
sweep, new data, fit or scientific-code alteration tests it into a promotion.

## 6. Actual results and next scientific decision

All386SOURCE/24application authors/288events/3old material arms/5targets retained.
The independent cold process computed900 joint pair laws in1.510sCPU, no model,
API, private material, neural training, new texts or new people. Native comparison
answers enter only later analysis; application labels never enter this packet.

Full-history32edges: shared identity probability .889442-.972911, average changed
identity probability .064075. Shared/independent edge-vector cosines E/N/A/C/O:
.999991995/.999995166/.999998894/.999997702/.999995764. All160 directions and
all1380 full mathematical target-pair directions remain unchanged. Across all
3arms,2/4140 full-pair directions change (remove_support E/C,one each), none on
the32native edges. All five mathematical graphs remain acyclic in every arm.

The old native A cycle corresponds to shared Pi=.502650/.494119/.503224:
its second edge remains opposite. Full shared/native edge contact E/N/A/C/O:
.494530/.104548/-.080193/.408001/.103529, essentially the earlier independent
contacts. E/N shared edge cosine is+.088627, NOT native comparative+.693144.
The positive construction works, but this particular shared-reference mechanism
does not account for the neural observer's target organization or A cycle.

Post-outcome contrast variances under shared references are .06283-.06499 of
independent values on the old graph. This is a ~94% reference-draw dispersion
reduction WITHOUT corresponding new directional evidence. It is exactly why
variance reduction or a cleaner-looking output cannot be equated with progress
toward a personality mechanism. No prediction r/AUC promotion is performed.

Decision: stop common-reference/coupling-family optimization. The next inventory
concerns pair-conditioned use of target-qualified evidence versus a shared
comparison-task action (HM221/223/224/228): define what target operation reads
the SAME event content, not fit native answers or tune the SOURCE kernel. Keep
choice, state, conditions, joint structure and original18/100/44 OPEN.

## 7. Reproduction and ancestry

Canonical results: results/grain_math_response_v1/shared_reference_r1.
From the implementation repository, use the parent .venv Python:

    python scripts/run_grain_shared_reference_hm231.py check

Never rerun prepare/compute/analyze/verify over the sealed canonical output.
The cold interface is score_grain_shared_reference_cold.py --inputs --output;
input schema is weights/anchors/authors/edges only. Old E5 encoding remains
neural; this does not create a raw-text model-free encoder.

All900 two-marginal and mean accounts verified; alternate interval summation on
5fixedpairs and10 direct quadratures agree. Max audit error8.59e-15;39focused
tests/fourPython lint-format pass. Same-agent alternate arithmetic, NOT independent
blind replication. Earlier failures/science seals preserved; no live process.

- Oberst and Sontag2019: https://arxiv.org/html/1905.05824 ; SCM/stability sections read.
- Lorberbom et al.2021: https://arxiv.org/html/2111.06888 ; coupling sections2/4 and AppendixC read. Gumbel coupling and nonmaximality belong to this prior art.
- OpenAI/math pinned adc7f124: Random-coordinate-frames-and-partial-permutation-laws,
  sections/03-marginals.tex read through tuple-law bound. Explicit joint-vs-marginal
  conditioning inspires design, but its finite-field permutations/contraction
  theorem are NOT imported into text/personality measurement.
