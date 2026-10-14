# Compiled joint observations and qualified author moments

HM112 companion, 2026-10-10 JST. The finite plan, code and numerical outputs are
sealed separately. This is a construction result, not a new theorem about human
personality or an exact reconstruction of natural full-context inference.

## 1. What was missing and is now constructed

HM111 established a block-triangular intervention: independent event carriers
E_j, common prefix P and ownership/context relay R can supply a recursively
updated current carrier C. HM112 makes this into an executable operation.
The source/library stage receives only mid anchors and single-event changes.
For each layer l it stores pre-rotary context keys/values, not measured joint C
states. The constant current seed and rotary coordinates are included.

Given independent library addresses a=(e1,...,e6,r), define

    C^0 = Embedding(current tokens)
    z^l = RMSNorm_l(C^l)
    Q^l = RoPE(W_Q^l z^l + b_Q^l)
    K_C^l = RoPE(W_K^l z^l + b_K^l)
    V_C^l = W_V^l z^l + b_V^l
    U^l = Softmax(Q^l [K_library^l,K_C^l]^T / sqrt(d) + M) V_all^l
    H^l = C^l + W_O^l U^l
    C^(l+1) = H^l + W_down^l[SiLU(W_gate^l RMSNorm(H^l))
                                  * W_up^l RMSNorm(H^l)].

GQA repeats each KV head over its declared query-head group; masks preserve
token causality and exclude padding. These are the actual local Qwen operations,
not the exact LayerNorm/architecture of the 2017 Transformer. After17 blocks,
layer17 RMSNorm and K/V projections yield the fixed target observation K_q(a).
No output-token generation or full model forward occurs in the cold process.

This is NOT neural-model independence: pretrained projection/norm/MLP weights
and upstream event embeddings remain. Compilation makes dependencies explicit
and reusable; it does not discover their learned semantics from first principles.
The 00 instrument excludes prior event contextualization and relay reading of
event content. It is not the full11 instrument, whose signed routes HM111 retains.

## 2. Why this is not fitting a better teacher surrogate

No pair or triple reference is an input to the constructor. Independent libraries
and the operation construct all420 numerical configurations before comparison.
There are zero fitted coefficients. Only28 full-prefix library calls supply two
banks. New triple references are then captured128 times; the formerly opened
HM111 pairs provide a known-material comparison, not a blind new replication.

The maximum readout error of1.53e-5 certifies arithmetic, not psychology. Once
within the frozen tolerance there is no scientific reason to optimize it further.
The purpose of this construction is to make a reusable observation kernel
available for studying author evidence composition without pair-table training.

## 3. The author object is separate from the common instrument

For qualified event relations under conditions c, the author's observation is

    Theta_uq(c) = integral K_q(e1,...,em,r,c) dP_u(e1,...,em,r | c).

K_q belongs to the shared instrument; P_u records the author's evidence and
choice/relations. Neither is a psychological ground truth by definition. The
ownership gate admits only A-owned evidence into A's author law. B-owned cases
remain computational references, not contributions to A. Selection and state
are not deleted; the present current text is fixed, so this packet does not
estimate changing state or a personal response operator B_u.

On binary finite support, classical character expansion gives

    K_q(x | c) = sum_S b_qS(c) chi_S(x),
    m_uS(c) = E_u[chi_S(x) | c],
    Theta_uq(c) = sum_S b_qS(c) m_uS(c).

This exact finite identity distinguishes instrument response b from author
composition m. It is not a universal low-degree personality formula. The cold
collector constructs K directly; it does not require a fitted table of b for
every pair/triple. A scalar observation can depend on distributed relational
evidence even when no stable named psychological factor has been identified.

## 4. A constructive obstruction for all pair summaries

With chi=x1*x2*x3, use positive author laws

    P_plus(x)=(1+.4 chi)/8,
    P_minus(x)=(1-.4 chi)/8.

Their normalization, first moments and all pair moments are equal. Thus every
method whose only input is those summaries returns the same value for both
authors, even if its downstream map is nonlinear. Their observation difference
is nevertheless exactly

    Theta_plus,q - Theta_minus,q = .8 b_q,{1,2,3}.

In actual new triples the RMS difference is .00179604/.00200088 across the two
material banks. Explicit40-count integrations agree with the character equation
to2.66e-15. Construction reproduces these author differences with max error
3.08e-6. Some individual effects are tiny (minimum2.48e-6); do not claim every
cell is separable beyond floating-point effects. There is no population p-value
or new personality correlation in these designed laws.

The two banks'24-cell effect patterns have cosine -.294091 (result-after).
Both wording and situations change. The same operation constructs both, but
slot labels do not transport a universal third-order coefficient. This is an
operation/material distinction, not a rejection of shared psychological traits.

## 5. Large effects can be created by the instrument

Broadcasting layer16 context into all layers violates the library's native layer
units. This fixed ablation makes author RMS .012611/.092200, versus actual
.001796/.002001. A bigger contrast is therefore not evidence of better
personality extraction. The ablation does not prove every layer is necessary or
that no valid compressed library exists; it isolates this particular mismatch.

Precision work ends at the construction gate. The next substantive issue is
which event relations make up P_u, rather than which arbitrary chunk grouping
or concatenation the instrument happened to receive. The same qualified event
graph should not become a different author merely because reports are repacked.
Genuinely different event relations may change the observation. This requires a
construction/observation-law experiment, not another fidelity or layer grid.

## 6. Provenance and contribution boundary

[Vaswani et al.2017](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)
defines scaled dot-product attention/stacked attention and pointwise nonlinear
operations. [Elhage et al.2021](https://transformer-circuits.pub/2021/framework/index.html)
already analyzes transformer composition. Cold KV reuse exists in this repo.
Binary character orthogonality, graph dependencies and the author-integral
identity are mature mathematics. The contribution here is a tested construction
with qualified author evidence and an explicit information-retention consequence.

OpenAI/math adc7f124's06-matrix.tex1--145 supplies the design idea of fixed
complementary observation families. Its harmonic-frame/elliptic assumptions are
not assumptions of the token graph; none of its PDE theorems is imported.
The repository also warns that unformalized manuscripts may have issues.
No novel general theorem or completed original18/100/44 scope is claimed.
