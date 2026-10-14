# Position-Conditioned Token Observation

2026-10-08. Derived after HM13; representation proposal, not a completed
standalone personality instrument. Standard interpolation/group facts
are explicit below, not claimed as new mathematical theorems.

## 1. Exact Empirical Obstruction

For fixed receiver context c and query q, HM13 compared H and R H, where
R reverses only the declared property-span rows. The unordered multiset
of complete float32 vectors was byte-exactly unchanged. The original
receiver, question states, other spans and later-network architecture
were retained. All 36 actual one-token controls had output difference0.

The largest measured difference was .009014125525649702, application
pair125, centered_reverse, trust query. Its comparison was the actual
previous centered-only score, not an unrounded hypothetical endpoint.
Thus this finite reader is not a function of the unordered span alone.

For any set-valued observation S with S(H)=S(RH), and any deterministic
readout g(S;c), the triangle inequality gives

\[
\max\{|g(S(H);c)-F_q(H;c)|,
      |g(S(RH);c)-F_q(RH;c)|\}
\ge \tfrac12|F_q(RH;c)-F_q(H;c)|.
\]

For the recorded endpoints this lower bound is .004507062762824851.
It applies to all unordered moments, not just mean/covariance, and does
not depend on smoothness or a particular regression family. It is a
finite intervention-fidelity bound; it is not a psychological error floor.

## 2. Two Different Permutations

Let E describe typed positions/edges/opportunities. A coordinate relabeling
transports both the nodes and their structure:

\[
(H,E)\mapsto(PH,PE),
\]

where PE denotes the corresponding transport of labels and both indices
of any edge matrix. A correctly defined graph function can be invariant
to this relabeling while remaining sensitive to

\[
(H,E)\mapsto(PH,E).
\]

HM13 tested the second operation: vectors moved, decoder positions and
causal structure did not. It did not test joint coordinate transport.
Therefore the result does not refute invariant graph/set methods; it
refutes treating the *untagged* vector set as a sufficient observation
for this fixed action. Tagged sets and ordered sequences are both viable.

## 3. Ordered Moments as a Concrete Observation

For distinct normalized span positions p_t, define

\[
m_k(H)=\frac1T\sum_{t=1}^T p_t^k h_t,
\quad k=0,\ldots,K.
\]

m_0 is the ordinary mean. With symmetric positions p_{T+1-t}=-p_t,
reversal gives m_k(RH)=(-1)^k m_k(H). Odd moments retain placement
information that even moments and the mean lose. For T=1 choose p_1=0;
the placement-sensitive channels vanish, as required by actual no-ops.

For K=T-1, the Vandermonde matrix V_{kt}=p_t^k is invertible because

\[
\det V=\prod_{i<j}(p_j-p_i)\ne0.
\]

Writing M=VH/T yields H=TV^{-1}M. This is an exact encoding of a fixed
finite sequence, not information compression, a universal noise filter,
or a guarantee of good estimation. High-order monomials may be badly
conditioned; a fixed discrete orthogonal basis is a numerical alternative.

When K<T-1, the moment map generally has a kernel. Its usefulness must
be decided by reader-action tests, not algebraic computability alone.
Similarly m_k(HU) retains only a source-frozen vector subspace U; ordered
moments cannot automatically recover the independent vector complement.

## 4. Source-Learned Target Action

A candidate compiled function should be evaluated on interventions:

\[
\widehat{\Delta F_q}
=g_q(\mathcal O(H');c)-g_q(\mathcal O(H);c),
\]

not only natural level correlation. The next bounded source-only packet
compares mean-only observation with fixed first ordered moments and a
complete small-span ordered frame. Source roles alone estimate coefficients
and scales; validation/application outcomes do not choose rank, degree,
layer, prompt, alpha or feature subsets. Constant/zero action remains an
explicit reference. All signs, failures and conditional family intervals
are retained. A successful ordered probe may still miss contextual coupling.

The author/event/condition ledger remains upstream of aggregation. A
position effect belongs to a text observation, not automatically to an
author or trait. Owner eligibility, history and psychological anchors
are not replaced by a raw token placement score.

## 5. Literature and Reading Scope

[Yun et al. 2020](https://arxiv.org/html/1912.10077v2), sections2-3.1,
distinguishes permutation-equivariant transformer families from families
with positional encodings. This is an architecture expressivity result;
it does not prove this trained decoder uses a particular psychological
mechanism. Its unmasked/no-layer-normalization setup is not silently
identified with Qwen's causal attention and RoPE. No full proof reproduced.

Pinned OpenAI/math `Backward-intertwiners-and-a-transitive-commutant-
September-27-2026/build/sections/intertwiner.tex` lines1-130 was read.
The recurring commuting-square idea suggests checking both operation
orders under the same transported condition. Its infinite-dimensional
shift, drift and measure assumptions are not present here. No operator
algebra theorem is imported as a neural/personality theorem.

## 6. Frozen Evidence

- Primary: `results/grain_math_response_v1/token_permutation_action_r1`.
- Alternate arithmetic: `results/grain_math_response_v1/token_permutation_verification_r1`.
- 252 actual local forwards,75.978sec; zero API/training/private inputs.
- Full1536-dimensional means exactly equal; covariance replay <=9.474e-15.
- All old HM12/HM10B3 artifacts unchanged; HM13 primary files frozen.
- Parent report: `TOKEN_PLACEMENT_AND_SET_OBSTRUCTION_REPORT_20261008_ZH.md`.

Natural-text transport, individual response B_u, external psychological
meaning and original18/100/44 requirements remain separately tracked.
