# Next P1O: exact cached geometry versus conditional response

2026-09-07. Planned only. Zero additional neural forwards; no new labels or private
data. Current decision: does off-target attenuation happen in the anchor state,
in the subsequent readout, or both? P1N functional selectivity alone cannot tell.

## Complete cache-only unit

Use all P1N states. Select canonical event anchors for each person, and object
anchors for room:4 settings x3 targets x4 worlds=48 vectors, grouped into12
four-world rectangles. Person B sign means the queried person's initiating role;
room B sign is original record-level sign only, never a personality axis.

Use the exact four-cell vector expansion:

    n(b,r) = mu + b*u + r*v + b*r*w,
    mu=mean(n), u=mean(b*n), v=mean(r*n), w=mean(b*r*n).

Independently replay with a4-column linear solve. This is finite Boolean/Fourier
algebra, not new latent factors. Save all vectors, norms and reconstruction errors.
No clustering, naming, outcome-driven dimension/rank selection, or significance
test treating96 coordinates or48 vectors as independent people.

Compare for every setting/target and B/R: RMS anchor displacement, RMS observed
single-anchor logit shift, and their ratio. The ratio is an empirical finite
directional gain, not a derivative, global Lipschitz constant or information unit.
Keep event versus attribution readouts separate although their name states match.
Report angles between u/v, mixed-vector norm relative to main effects, and sign/
geometry variation across the4 settings. Do not average away opposing signs.
Examine whether a large geometric off-target change has small functional effect,
and whether near-additive vector geometry still has nonadditive readout. Verify
P1N's difference n_BR-minus-n_ADD equals4*b*r*w up to saved
float32 arithmetic, without claiming natural plausibility for the synthetic state.

## Resource and continuation rule

CPU-only cache analysis capped60seconds, all12 rectangles and80 observed anchor
effect rows per axis included without selecting favorable cases. No model forward,
hyperparameter fit or downloads. Separate immutable output directory and audit
receipt. Test rotation invariance of norm/angle quantities and exact reconstruction
on synthetic matrices; use no real text in tests.

If weak influence is mainly readout-dependent, represent GRAIN mechanism candidates
as query-conditioned functional geometry, not discarded noisy vector coordinates.
If it is mainly in encoding, investigate target-indexed retrieval before inventing
a special decoding direction. Mixed evidence is a valid outcome. Only after this
analysis define the smallest held-material intervention needed for causal subspace
alignment; a larger model or real-author material is not automatically authorized.
