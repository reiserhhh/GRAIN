# GRAIN P05 native-action packet, frozen protocol V1 (replaces proposal V1 for execution)

Frozen 2026-09-26 JST before any numerical model call. Authority:
`reports/claude_research_loop_20260926/SEMINAR_FABLE_05_DECISION.md` (05:43), including
its four Codex corrections. `docs/GRAIN_P05_NATIVE_J_PACKET_PROPOSAL_V1.md` stays unchanged
history.

## Question and scope

Does source-only target-action weighting of the compile objective improve an ACTUAL
compiled intervention? Tested crossed with 4 vs 24 frozen GRAIN input coordinates, on the
frozen conditional_operator_v1/r2 panel, local observer, site L12.H0 (head 0 at the input
of layer-12 `o_proj`), existing query suffixes.
- An improvement supports this specified estimator on this panel. It does not uniquely
  prove objective mismatch.
- A null result rejects only this estimator, input and site scope.
- No generated text, no neural weight training, no downloads, no API.

## Material (exact IDs in `docs/GRAIN_P05_NATIVE_ACTION_PACKET_IDS_V1.json`)

- 36 source pairs and 72 application pairs of r2, in r2 pair order. The queries are
  extraversion (suffix 59 x 128) and neuroticism (60 x 128).
- y_i: the cached donor head-state change of r2, `action_arrays.npz` dh, split per query.
  Source rows are used for fitting only. Application rows are never used for fitting,
  scaling or selection.
- Inputs:
  - de4 is the r2 four-coordinate difference (`dx`).
  - de24 is the frozen GRAIN runtime (24 coordinates) applied to the high and low texts.
  - Both are standardized by the SOURCE population SD. No coordinate has zero source
    variance (minimum SD .0063); a coordinate with SD < 1e-12 would be dropped in every
    cell.
  - The application |z| reaches 3.45 (4-coord) and 6.8 (24-coord) against source maxima
    of 2.94 and 3.14. This extrapolation is reported, not corrected.

## Fitting (four cells, per query segment, no intercept, alpha = 1.0, no rank truncation)

- Euclidean: minimize sum_i ||y_i - W^T z_i||^2 + alpha ||W||_F^2. Closed form
  W = (Z^T Z + alpha I)^-1 Z^T Y. Rank <= p, with no truncation.
- Action-weighted: minimize sum_i s_q (J_iq . (y_iq - W_q^T z_i))^2 + alpha ||W_q||_F^2.
  - J_iq is the native gradient of query q's conditional score at the LOW source text,
    with respect to the L12.H0 suffix input rows (flattened).
  - Metric scale, source-only and frozen: s_q = D_q / mean_i ||J_iq||^2, with D_q the
    segment dimension. Under this scale the expected metric of an isotropic J
    distribution is the identity, so an isotropic or identity metric reproduces the
    Euclidean objective exactly. The shared alpha then has compatible loss units.
  - Dual solution: u_i = sqrt(s_q) J_iq; K_ij = (z_i . z_j)(u_i . u_j);
    t_i = u_i . y_iq; c = (K + alpha I)^-1 t; W_q = sum_i c_i z_i u_i^T.
  - The compiled change is a(z) = sum_i c_i (z_i . z) u_i. No dense parameter covariance
    is formed.
- The old r2 compiled arm (rank 2, joint fit) is a historical reference only, not a
  matched cell.
- Reported per cell and query:
  - exact objective values;
  - eigenvalues of the aggregated metric sum_i s_q J_iq J_iq^T: rank and nullspace
    dimension D_q minus rank;
  - source fit error;
  - application action norms relative to donor norms.

## Call table (items counted; each backward counted separately; hard cap 960)

| Step | Passes | Order |
|---|---|---|
| Replay: unpatched LOW score on replay pairs x 2 queries vs cached | 8 | 1 |
| Replay: cached donor intervention on replay pairs x 2 queries vs cached donor effect | 8 | 1 |
| Native J: 36 source LOW texts x 2 queries, forward + backward | 72 + 72 | 2 |
| Finite-difference check, central +-h on 2 source pairs, extraversion query | 4 | 3 |
| Compiled arms: 4 cells x 72 application pairs x 2 queries | 576 | 4 |
| Equal-norm random, matched to the J-24 arm (per-row norm, seed key `jw24|pair|query`) | 144 | 5 |
| J-24 source linearization diagnostic (36 x 2) | 72 | 6 |
| **Total planned** | **956** | |

- The cap is enforced before every call: an attempt is written to the accounting file
  before the call.
- Per-item saves; no automatic retry; offline only.
- On any failed call, replay miss or gradient-check miss: stop, checkpoint and report the
  exact blocker. The experiment is not expanded. The R17 budget is separate.

## Tolerances

- Unpatched low replay: |score - cached low score| <= 1e-5.
- Donor replay: |effect - cached donor effect| <= 1e-5.
- Finite difference, with h = 0.1 along the unit gradient direction: the central
  difference (s(+h) - s(-h)) / (2h) must match ||J|| within 10% relative. This is float32
  MPS; the test cannot certify FP64.

## Outcomes

- Per application pair, query and arm:
  - patched score; effect = patched score - cached low score;
  - candidate probability mass; intervention norm; errors.
- **Primary reference:** the cached measured donor action on the same pair and query.
- **Secondary, labelled separately:** the full-text contrast, which includes other
  pathways.
- **Per arm and query:**
  - RMSE against donor and against the zero reference;
  - Pearson r;
  - sign agreement against the donor-sign prevalence reference (.528 E / .514 N);
  - random-arm reference.
- **Core paired contrasts:** J24 vs Euclid24 (metric); J4 vs Euclid4 (metric at small
  input); J24 vs J4 and Euclid24 vs Euclid4 (representation under fixed loss).
  - Reported as delta RMSE and delta r, with 2,000-draw cluster bootstrap by material
    cell (axis|family|situation, seeds kept together; seed 20260926).
  - Strata: new_task, new_wording, new_both. Both queries and all four cells are kept.
- **Diagnostics:** J-24 source linearization error (predicted J.a vs patched effect);
  norm and extrapolation.
- **Interpretation (stated in advance):**
  - J24 better than Euclid24 on paired RMSE, CI excluding 0, and better than zero RMSE:
    supports the action-weighted estimator at this scope. This is a candidate Task A
    operation, not a unique proof.
  - Euclid24 about equal to J24, both better than the 4-coordinate cells: input richness
    rather than loss.
  - No cell better than zero RMSE: linear compilation from these lexical coordinates
    fails at this site and scope. The vector path is then an instrument audit only.

## Checksum bindings (verified at run start)

| file | sha256 |
|---|---|
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/action_arrays.npz` | `778a5acdd88649e31eb8b5bd0013ddab0e50bacf4cc3e0d0bbe2325b8f9c0a9f` |
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/pairs.csv` | `1d648e3b822ac42e668bdd6273f0e961f04f3a4e65097881428603d33e67f43e` |
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/pair_order.json` | `a942fd048c42ac89a241ee27929f29f37d7599e0c9c3fb114d3d935eeabaf17a` |
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/native_effects.jsonl` | `db3d17b41a6d3322100811f666cf227b0b5612241437e1cc66bc0702d8647551` |
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/model_manifest.json` | `0f6d1e7535362e5d0c4e25de6afcde7629ca8da4d6de130775bb1467a69f7bae` |
| `results/grain_vector_mechanisms/persona_generation_trace_v1/expanded/generations.jsonl` | `e493f6b30fe8aa0a0baa0c11c003b69d3ac0d9114d74ebef0e9e92d590eb9727` |
| `results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib` | `60ef4369c6203814bc60fa003ce9b4965ccd88452b101da824c4658dc31320e8` |
| `scripts/run_grain_conditional_operator_native.py` | `c3d344b90c70fd91f32f0253fb1368bfdfab83ef21cf59c44b9faf850981754b` |
| `docs/GRAIN_P05_NATIVE_ACTION_PACKET_IDS_V1.json` | `4f3d38693cf8dd2024fc5a8be5a9ee2c7b4f80479bcf230d04e10a1ebe557b87` |
