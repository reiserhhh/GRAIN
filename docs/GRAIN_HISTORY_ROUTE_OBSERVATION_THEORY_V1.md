# Contextual evidence writing and route-dependent observation

2026-10-10. HM101 finite result; original18/100/44 remains OPEN.
This is a construction consequence, not a new transformer theorem or a
human personality-accuracy claim.

## 1. Separate the three objects

Let H denote a historical record, B the other prefix tokens and C the current
action tokens. In HM101, B includes the system header, neutral bridge and
current-record header; it is NOT only the neutral bridge words.

For the frozen encoder and a declared permission graph G, write

    X_C = Phi_G(H, B, C),
    F_tau(H, C; B, G) = R_tau(X_C),
    S_u,tau = sum_(i,j) P_u(i,j | B,G) F_tau(i,j; B,G).

Phi is contextual writing, R is a target-specific frozen readout and P is an
author's evidence composition. A geometric displacement of X, a target
response F and an author contrast S are different quantities. None becomes
a psychological trait simply by being nonzero.

HM101 uses current K/V before block17 attention, fixed source queries and
covectors, and positive synthetic P laws. It does not estimate real authors'
P, psychological scores or a full native-answer function.

## 2. What is actually cut

At each completed attention block0--16, d enables H -> C and r enables H -> B.
B -> C, C -> C and H-internal connections are retained. The base causal mask
and all weights are fixed. d=1 admits direct ENTRY into C, not a single-head
path with no later within-C computation. r=1 admits H -> B -> C through
successive layers; B tokens before H cannot relay it under causality.

The historical span is right-padded to a fixed token width with ignored keys.
C/B IDs and positions are identical across historical variants. This is
deliberately controlled token input, not ordinary unmodified natural prose.
Blocked padding cannot act as a relay because no token can attend to its key.

### No-transmission lemma at the declared boundary

Assume deterministic inference, tokenwise normalization/MLP, fixed positions,
and cross-token mixing only through the declared masked attention graph.
Initially only H depends on the historical variant. If d=r=0, no token outside
H can read H. Induct over blocks: tokenwise functions and residual addition
cannot introduce an H dependence; attention with no H-dependent input cannot
introduce one either. Thus X_C is invariant across H variants.

The finite model controls give current K/V difference0 and no-op hook
difference0. These check the implementation of this familiar dependency-graph
argument, not the validity of a personality interpretation. Finite numerical
masking is also checked directly against the allowed graph at every block.

With relay-only, dependence can first enter B in one layer and C in another.
Permission/support establishes where transmission is possible, not that every
permitted edge carries a nonzero effect.

## 3. Exact finite interaction, not additive route assembly

For any one fixed observable, evaluate all four F_dr. Boolean Mobius inversion
gives the signed coefficients

    A0  = F00,
    Ad  = F10 - F00,
    Ar  = F01 - F00,
    Adr = F11 - F10 - F01 + F00,
    F11 = A0 + Ad + Ar + Adr.

This equality requires no linear encoder. The coefficients DO depend on the
intervention definition, graph, token construction and readout. Cutting an
edge alters softmax normalization and subsequent computation. The result is
not an attribution obtained by removing a stored attention contribution, not
donor-activation path patching, and not unique natural causal mediation.

For paired laws P+ and P-, author contrast is the linear functional

    Delta_tau(F) = sum_(i,j) (P+_ij - P-_ij) F_tau(i,j).

It commutes with the finite account:

    Delta(F11) = Delta(A0)+Delta(Ad)+Delta(Ar)+Delta(Adr).

HM101 uses D=[[0,1,-1],[-1,0,1],[1,-1,0]] and P+/-=(1 +/- .5 D)/9.
Both are positive, have equal row/column margins and equal symmetric traffic.
Delta=<D,F>/9 therefore isolates this designed circulation account. This
three-state design is not a model of every human author or a personality test.

## 4. Measurement consequence

The local-query synthetic author contrast in the new-wording bank has RMS
.176971. The interaction component has RMS .166224, relay .054918 and direct
.143960. These are fixed model-readout units across the full retained table,
not uncertainty-adjusted effects or psychological information percentages.
The substantial mixed component rejects additive assembly for this design.

In raw mean-value displacement, per-frame signed alignments average about
direct .65, relay .01 and interaction .34 in both wording banks. But the full
target-projected author-contrast tables have source/application cosine .086190
for local-query and -.194229 for uniform-value. Similar write-side account
geometry does not guarantee a shared target response law. The banks use
different actions/scenarios; design correspondence is not psychological truth
equivalence. No threshold or sign correction is chosen after observing them.

Consequently an isolated action span is not automatically a sufficient
observation unit. Its context-conditioned K/V may already carry historical
relations, including information propagated through lexically neutral carriers.
The correct response is not to discard every neutral/context token, nor to
declare every token psychological evidence. Preserve contextual provenance
and the write/read interface while defining which evidence the author account
is entitled to accumulate.

## 5. Next scientific decision

Connect the history-writing operator to a target-specific reading law and then
to author evidence composition. The next finite packet must distinguish:

- A shared context carrier with target-specific pairing.
- Context-specific write dynamics that require a different observation law.
- Loss caused by summarizing before the mixed response is formed.

Use frozen current tokens, source-constructed operators and held-out public
relations/conditions. Keep both successful and reversed responses. Do not fit
a global rescaling, head/rank grid or lexical classifier merely to improve the
existing table. A precision repair is justified only when it enables a stated
new mechanism distinction.

## 6. Prior work and OpenAI/math boundary

[Elhage et al.2021](https://transformer-circuits.pub/2021/framework/index.html)
already analyze attention-head composition and virtual connections. The
primary search excerpt was consulted; this turn's full web open exceeded the
tool size limit, so it is not recorded as a new full-text reading.

[Wang et al.2022, Appendix B](https://arxiv.org/html/2211.00593v1)
provide donor-based path patching, distinguishing selected direct paths from
other downstream changes. Their backup-head findings also motivate care when
an ablation changes the mechanism. Our permission-mask experiment differs
from their patching algorithm; the general intervention idea is not new.

[OpenAI/math pinned isotypic section](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Signed-tensor-densities-and-diagram-budgets-for-coordinate-sweeps-September-26-2026/build/sections/03b-isotypic.tex),
commit adc7f1241b42e322a6451854ab7e4b4c146bf78a, lines106--194, motivates
declaring placement, support and composition hypotheses. Its signed-density
and tensor bounds are NOT assumed for a transformer. The route account uses
standard finite Mobius algebra and masked-graph dependence instead.

The prospective GRAIN contribution is a reproducible context-conditioned
evidence observation and its author-measurement consequence. It is not the
invention of attention composition, factorial interactions or a new general
mathematical theorem.

## 7. Evidence and reproducibility

Canonical: results/grain_math_response_v1/history_routing_r1.
Plan: docs/GRAIN_HISTORY_ROUTING_PLAN_HM101.md.
Runner: scripts/run_grain_history_routing_hm101.py.
Independent-formula companion: scripts/verify_grain_history_routing_hm101.py
(same Codex agent, not a third-party rebuild).

216 new token streams, 864 intervention forwards plus two no-op controls,
78.470707 model seconds; no training, API, private data or outcome-selected
route. Focused tests14 (7 new), all four Python lint/format checks pass.
All finite stages ended; original requirements and human validity stay OPEN.
