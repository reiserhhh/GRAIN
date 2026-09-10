# NEXT43: context-dependent gated transformation of GRAIN directions

2026-09-10. Planned, not executed. NEXT42 located a large block25 MLP update on
new8 authors, with signed projected shares .371/.508 for N/confirmation.
This follow-up is an exploratory mechanism test selected from that observation,
not a new blind validation or an early-exit tuning sweep.

## Decision

Is the update mostly a fixed linear transformation of the incoming contextual
tangent, or does it materially require current-context gate/normalization
coefficients? Either answer develops the extraction path: a fixed source
operator is simpler; needed conditional coefficients identify what must be read.

Use same24 units, same four BODY directions, two queries, source4 LOO and
new8 fixed-four sources. Freeze all arms/code before targets. No new human
labels, new model family, query selection, encoder training or factor naming.

## Exact local object

At the final position of block25 after attention, let input be x and incoming
tangent d. The real Qwen2 MLP has no projection biases and uses SiLU. Verify
these architectural conditions against the pinned runtime before collection.

    z = RMSNorm(x); g = W_gate z; u = W_up z
    M(x) = W_down [silu(g) * u]
    dM = W_down [silu(g) * (W_up dz)
                    + u * silu'(g) * (W_gate dz)]
    silu'(g) = sigmoid(g) * [1 + g*(1-sigmoid(g))]
    dz = gamma * [d/rho - x*(x^T d)/(width*rho^3)]
    rho = sqrt(mean(x^2) + epsilon).

Elementwise products are marked *. Both terms depend on current context;
their names are architectural, not separate content/personality channels.
RMS radial and scale effects must be retained, including epsilon and actual
dtype. Report both signed gate/input terms and their exact sum.

Define B(z) = W_down[diag(silu(g))W_up + diag(u*silu'(g))W_gate],
N(x) = D RMSNorm[x]. Actual update is B(z_x)N(x)d.

Fixed four alternatives, current d held unchanged:

1. Current B and current N: analytic identity/reference, not a fitted model.
2. Mean source B and current N: freeze activation coefficients only.
3. Current B and mean source N: freeze norm geometry only.
4. Mean paired source B_s N_s: freeze the complete local Jacobian.

Use other3 sources for source4 and all4 for new8. Keep each source pair together
in arm4: product of means is not mean of products. Use per-query sources, and
report existing own/pooled final t separately without choosing winners. No
current targets or gradients construct those source operators.

## Bounded execution and checks

1. Add exact analytic helpers, with tiny real-Qwen2 CPU/MPS tests against JVP.
   Include nonzero norm epsilon, zero directions, signed term closure, source
   exclusion and source-pair versus product-of-means distinction.
2. Reuse full H16 forward-only inputs. At most96 JVPs run17..24 and attention25
   to capture final x/d for24x4 directions. One pass returns primal and tangent;
   preserve full-sequence native shapes. Do not execute block25 MLP or later
   blocks just to collect already available native comparison targets.
3. Export/cache all current x/d before numerical source compilation. No current
   score, backward or human label. Compile only source x from these packets.
4. In a separate target-denied numeric process calculate all four alternatives
   and both source readouts. Use actual pinned weights; these remain dependencies.
   Predictions for the declared MLP increment freeze before opening NEXT42
   `updates[:,17]` or its scalar action. It is the target, not an input feature.
5. Analytic current dM vs saved NEXT42 vector must match relativeL2<=1e-4 OR
   maxabsolute<=1e-5. Do not widen tolerances or silently change model kernels.
   Test finite differences locally on the tiny model; no additional real full
   suffix calls are required for numerical comparison.
6. Report all arms/cohorts/queries/factors: vector relative error, scalar r,
   factor-centered r, SSE/zero, RMS, signs, and signed term projection/cancellation.
   Do not label a large projected term as an information fraction. Report if
   different factors require different coefficients; do not tune per-factor arms.

Cap300s after model loading,96 native JVPs,0 new backwards/scores; forecast first
complete author and pause resumably at boundary. Numeric stage cap120s; no
optimizer grid, more source people or unattended fallback. Saved target parity
and artifact replay suffice; do not repeat the native run to fill a replay table.

## Finish / next branch

If source-frozen B/N retain useful action, develop a compiled local operator
instead of assuming dynamic gating is indispensable. If conditional B is
material, next model its structured activation coefficients from permitted
input, with source-only fitting; if norm alone explains the discrepancy, address
its geometry before adding a complex learner. Keep signed and amplitude results.

These are local replacements with the upstream context tangent still native.
Do not reconstruct a purported independent full prediction using unreported
native downstream summands. No current psychological interpretation, release
scorer, lockbox, MEPS use or clinical rubric is changed.

## Deliverables / primary methodological connection

Pre-collection implementation detail: capture/weights keep the pinned float32
values; the independent NumPy analytic program promotes them to float64 for
the explicit formulas. Compare to the unchanged native float32 target using
the tolerance above, not bit equality or a new kernel. Source-B coefficients
are same-query means, also when the final readout t is pooled-query. The two
source choices are different interfaces and are not silently coupled.

Proposed `gated_update.py`, `run_grain_gated_update.py`, synthetic tests; private
`logic_gated_update_v1/`; aggregate `GATED_UPDATE_REPORT_V1.md`; update theory,
input dependency inventory and whole-system ledger with whichever result occurs.

[Shazeer2020](https://arxiv.org/abs/2002.05202), equations5-6, supplies the SwiGLU
architecture. The two directional terms above follow by the product and chain
rules. [Geva et al.2021](https://aclanthology.org/2021.emnlp-main.446/) and
[Geva et al.2022](https://aclanthology.org/2022.emnlp-main.3/) motivate examining
feed-forward activations and additive prediction updates. They do not establish
that these Qwen2 units correspond to GRAIN factors or human personality.
