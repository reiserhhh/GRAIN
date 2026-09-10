# GRAIN: conditional gated operators on a contextual direction

2026-09-10. NEXT43 develops the block25 computation located by NEXT42. The
object is a fixed GRAIN direction's model-local response, not a newly named
personality factor. All source/context comparisons retain the same incoming d.

## 1. Three distinct objects

The current final-position state x summarizes earlier context. The tangent d
describes how that state changes under a fixed BODY GRAIN direction. The local
operator J(x) describes what block25 does to that incoming change. Thus:

    GRAIN BODY direction -> native upstream transport -> d
    current context x -> local differential operator J(x)
    source-fixed query readout tbar -> action tbar^T J(x)d.

Changing J while holding d fixed tests a local computational dependency. It
does not replace upstream contextualization, reconstruct the whole model score,
or test whether the observed author personally uses this mechanism.

## 2. Actual SwiGLU derivative

For norm weight gamma, epsilon>0, hidden width h, and actual frozen matrices:

    rho = sqrt(x^T x/h + epsilon)
    z = gamma * x/rho
    N(x)d = gamma * [d/rho - x*(x^T d)/(h*rho^3)]
    g = W_gate z; u = W_up z
    p = silu(g)
    q = u * sigmoid(g) * [1 + g*(1-sigmoid(g))]
    B(z) = W_down [diag(p)W_up + diag(q)W_gate]
    J(x)d = B(z)N(x)d.

Two signed terms are kept before addition:

    input_term = W_down [p * W_up N(x)d]
    gate_term  = W_down [q * W_gate N(x)d].

The gate term is not a separate psychological gate; it is the second term of
the product rule. The input term also depends on the context through p and N.
Neither term alone is generically a faithful substitute for the full derivative.

Norm is likewise not merely a gain: its rank-one radial correction can rotate
the tangent relative to a readout. With epsilon>0, a radial input is not exactly
annihilated. The capture uses pinned float32 native operations; the independent
explicit calculation promotes the same saved values/weights to float64 and
checks parity against the original float32 derivative at the declared tolerance.

## 3. Source operators versus current operators

Let Bbar=mean_s B_s and Nbar=mean_s N_s on the declared source authors. The four
fixed alternatives are:

    current:   B_x N_x d
    source_B:  Bbar N_x d
    source_N:  B_x Nbar d
    source_BN: mean_s(B_s N_s) d.

Source4 uses the other3 authors; new8 uses all4 sources. Source B/N construction
is same-query. The final source tbar is reported separately as own-query and
pooled-query; pooling t does not change B/N source membership.

This is not a conventional independent two-by-two factorial design: source_BN
keeps each source's B_s and N_s paired. Define the operator covariance

    C_BN = mean_s(B_s N_s) - Bbar Nbar.

Then the complete source approximation error has the exact decomposition

    (B_x N_x - mean_s B_s N_s)d
      = (B_x-Bbar)N_x d + Bbar(N_x-Nbar)d - C_BN d.

Separately averaging and multiplying would silently remove the last term.
The synthetic tests explicitly exhibit a nonzero difference. This formula does
not assert that operator covariance is a measured psychological relationship.

## 4. Geometry relevant to the requested reading

A field approximation error e can be large while tbar^T e is small. Conversely,
a small error aligned with tbar can be material. Report both the full update
vector error and signed scalar action; do not infer either from the other.

After source compilation, a fixed local operator can be expressed as a single
readout wbar=mean_s N_s^T B_s^T tbar, giving action=wbar^T d. That is an exact
algebraic compilation of the source_BN program, not an assertion of its observed
fidelity. If used, source weights are constructed once; current d still depends
on the original upstream model. Current-conditioned versions retain additional
current x dependencies.

## 5. Methodological lineage and next decision

[Shazeer2020](https://arxiv.org/abs/2002.05202) defines the gated feed-forward
architecture. Our decomposition follows its product form with RMS normalization
and the chain rule. [Geva et al.2022](https://aclanthology.org/2022.emnlp-main.3/)
examines feed-forward updates in prediction construction; this motivates looking
inside the located block, not naming its units as personality constructs.

Whether fixed source coefficients suffice, current gate coefficients matter,
or norm geometry dominates is an empirical question of the next packet. A
successful simple source operator should be retained rather than discarded for
being less exotic; a conditional dependency identifies what to extract next.

## 6. Observed update to the mechanism

NEXT43 now answers the local question: paired-source J retains new8 action
r=.996936/.998451 (N/confirmation), with SSE/zero=.011988/.002932. Full-vector
relative errors are .213/.194, so useful query response is more tolerant than
whole-vector equality. Both product-rule terms matter, but their finely varying
current coefficients are not required at this tested location and scale.

The source program was then compiled into wbar; all addressed scalar outputs
reproduce within7.772e-16. This exact post-outcome conversion is not another
prediction experiment. Next distinguish a shared reader from query-conditioned
incoming directions using [NEXT44](GRAIN_COMPILED_QUERY_TRANSFER_PLAN_V1.md).
Current upstream transport remains native; no whole-system independence claim.

Read [the fixed plan](GRAIN_GATED_UPDATE_PLAN_V1.md) and
[results](../reports/grain_vector_mechanisms/GATED_UPDATE_REPORT_V1.md).
