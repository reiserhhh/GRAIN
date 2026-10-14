# R01 native application-J diagnostic: protocol V1

Frozen 2026-09-26 JST before any model call.
- **Authority.** `NEXT_PACKET_17_APPLICATION_J_WORDING_AND_PROVENANCE.md` Part B; `SEMINAR_FABLE_15_DECISION.md`.
- **Runner and output.** `scripts/run_grain_r01_application_j.py`, writing to
  `results/grain_gptpro_foundation_v3/P05/application_j_diagnostic_r1/`.
- **Scope.** No generation, training, API, downloads, new fits, calibration, or alpha/rank/amplitude search. Nothing is
  selected from application outcomes, and no definition is changed.

## Question

Why do the frozen GRAIN-24 Euclidean compiles partly rank the donor action but overshoot, while the source
action-weighted compile fails? Measure the missing **application** tangent J_app. Then separate tangent (geometry)
mismatch from finite-step remainders.

## Material (frozen; unchanged)

- **Pairs.** The 72 application pairs of `conditional_operator_v1/r2`, in r2 pair order
  (`docs/GRAIN_P05_NATIVE_ACTION_PACKET_IDS_V1.json` → application_pairs). Queries are extraversion and neuroticism.
- **Model and site.**
  - Local observer, float32 on MPS, offline, `max_tokens` 1024, eager attention.
  - Site L12.H0: the first 128 columns of the input to layer-12 `o_proj`, on the query suffix rows. Suffix shapes are
    59 × 128 for E and 60 × 128 for N.
  - The intervention convention is additive, as in `run_grain_conditional_operator_native.intervene`.
- **Score.** The candidate-alphabet expected value (`direct_traits.candidates`), read from the last position.
- **Cached changes.** Each change d is a suffix × 128 array, flattened.

  | Change | Definition | Measured delta (patched score − cached LOW state score) |
  |---|---|---|
  | donor | r2 `dh` row | r2 native_effects kind=donor `score` − state LOW |
  | E4, J4, E24, J24 | native_action_r1 `private_prepared.npz` rows | native_action_r1 `effects.jsonl` `effect` |
  | RAND_J24 | reconstructed: `equal_norm_random(0, J24, "jw24\|pair\|query")` | native_action_r1 `effect` |
  | r2_compiled | r2 `predicted` row | r2 kind=compiled `score` − state LOW |
  | r2_random | reconstructed: `equal_norm_random(0, r2_compiled, "pair\|query")` | r2 kind=random `score` − state LOW |
  | zero | Δ = 0 | r2 kind=self `score` − state LOW |

- **Identity contract for reconstructed randoms.** The norm must equal the cached `action_norm` within 1e-9 relative.
  Otherwise that control is marked UNAVAILABLE and no model call substitutes for it.
- **Material cell** for the cluster bootstrap: axis|family|situation (the IDs file `cells`). Seeds stay together.

## Call table (items counted; each backward counted separately)

| Step | Items | Passes | Order |
|---|---|---|---|
| Replay: unpatched LOW score, 4 replay pairs × 2 queries | 8 | 8 forwards | 1 |
| Replay: cached donor intervention, same pairs | 8 | 8 forwards | 1 |
| Application gradient: 72 LOW texts × 2 queries | 144 | 144 forwards + 144 backwards | 2 |
| Central finite difference, ±h along the unit gradient, h = 0.1 | 2 pairs × 2 queries × 2 | 8 forwards | 3 |
| **Planned total** | | **312** | |

- **Replay pairs.** The IDs file's replay_pairs, which are application pairs: extraversion|anchored|advice|s11, s23,
  s37, and extraversion|anchored|diary|s11.
- **Finite-difference pairs** (fixed now by ID): extraversion|anchored|advice|s11 and extraversion|anchored|advice|s23,
  for both queries.
- **Hard cap 320 passes**, failed attempts included. The cap is checked and the attempt written to
  `accounting.jsonl` **before** every call. There are no automatic retries, and a prior failure blocks further calls.
- **Execution.** One process. Per-item atomic saves: a gradient goes to a temporary file, then is renamed, and its sha
  is recorded.
- **Time limit.** 30 minutes of wall clock after model loading. If exceeded, checkpoint and stop with partial results.
- **Before any call**, the suffix tokens of all 72 LOW texts are checked against the cached states, using the
  tokenizer only.

## Gates (numerical checks, not scientific criteria; fixed now)

1. **Replay.** |LOW score − cached state score| ≤ 1e-5, and |donor effect − cached donor effect| ≤ 1e-5. Tokens, head
   site and intervention convention are identical.
2. **Gradient-pass forward score.** |score − cached LOW state score| ≤ 1e-5 for every item.
3. **Finite difference.** If ‖J‖ ≥ 1e-6, |central difference − ‖J‖| / ‖J‖ ≤ 0.10. Otherwise the absolute error must be
   ≤ 1e-7. Exact errors are recorded.
- **Stop rule.** Any binding, replay, gradient or finite-difference failure stops neural work. The CPU parts of the
  packet (C, D) are still finished, and the diagnostic is not reported as complete.

## Decomposition (CPU, after the gates)

For each query q, pair i and cached change x:

- L_i(x) = J_app,i · x, the tangent prediction;
- R_i(x) = meas_i(x) − L_i(x), the finite-step remainder.

For each non-donor change c, with donor d:

- E = meas(c) − meas(d);
- T = J_app · (c − d), the tangent mismatch;
- ΔR = R(c) − R(d), the remainder difference;
- **identity:** E = T + ΔR. It is checked numerically: max |E − T − ΔR| ≤ 1e-12.

Reported per query and change:
- mean E², mean T², mean ΔR² and 2·mean(T·ΔR). These sum exactly to mean E². RMS values do not add.
- The shares s_T = mean T² / mean E², s_R = mean ΔR² / mean E² and s_X = 2 mean(TΔR) / mean E².
- MAE and signed means of E, T and ΔR.
- **Linearization per change:** corr(L(x), meas(x)), RMS R(x) / RMS meas(x), and the no-intercept slope of meas(x) on
  L(x).
- **Ranking and overshoot:** corr(meas(c), meas(d)) and corr(L(c), L(d)); RMS L(c) / RMS L(d) (overshoot in the
  tangent) and RMS meas(c) / RMS meas(d) (overshoot as measured).
- **Correlation guard.** A correlation is not reported (NaN) if either SD < 1e-8. For N, all terms are also given
  relative to RMS meas(donor), because N's donor amplitude is tiny.
- **Bootstrap.** Material-cell cluster bootstrap, 2,000 draws, `numpy.random.default_rng(20260926)`, one draw
  sequence per query, for s_T, s_R, s_X and the RMS terms.
- **Labels (frozen).** "Tangent mismatch dominant" if the lower CI bound of s_T exceeds the upper CI bound of s_R.
  "Finite-step remainder dominant" in the reverse case. "Combination" otherwise.

## Geometry (CPU)

- **Source span.** Q_q = the orthonormal basis of the 36 source gradients from `native_action_r1/private_jacobians`
  (tree sha 35e1b6d4…). Singular values below 1e-10 × the maximum are dropped, the same rule as
  `native_geometry_r1`.
- **Application share.** The share of each application gradient's energy in Q_q. The random reference is rank / D,
  computed rather than assumed.
- **Comparison.** Source leave-one-out coverage (.973 E, .981 N). Subspace energy is not action preservation.
- **Split tangent.** L(x) = L_in(x) + L_out(x), with J_in = Q Qᵀ J_app. Report the RMS of L_out(c − d) and L_in(c − d).
- **Alignment control (descriptive, not a p-value).** Within the same query, dimension and role, gradients are paired
  with the next pair's change (cyclic shift by 1 in r2 application order). Report the mismatched corr(L(d), meas(d))
  and RMS R(d).

## Extrapolation (descriptive)

- Per application pair: max |z24| and max |z4| (`private_prepared.npz` z24/z4), against the source maxima in
  `prepare.json` (3.14 and 2.94). Also the count of pairs beyond source support.
- Spearman correlations of max |z24| with |T| and |ΔR| for E24 and J24, per query. These cannot prove overfitting or
  causality.

## Decision rule (stated in advance)

- **Application tangent transport** is weakened as the cause if the mean application share in the source span is
  ≥ .9, near source leave-one-out coverage. It is supported if the share is < .5. Between those values it is
  unresolved.
- **Compiled-direction mismatch** is supported for a cell labelled "tangent mismatch dominant".
- **Finite-step nonlinearity** is supported for a cell labelled "finite-step remainder dominant".
- **Otherwise:** unresolved combination.
- **E24 overshoot.** It is placed in the tangent if RMS L(E24) / RMS L(d) is at least 1.5, and otherwise in the
  remainder.
- Both queries are kept. N is interpreted against its donor amplitude. Nothing here is human personality truth or
  LLM-free inference.

## Checksum bindings (verified at run start)

| file | sha256 |
|---|---|
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/action_arrays.npz` | `778a5acdd88649e31eb8b5bd0013ddab0e50bacf4cc3e0d0bbe2325b8f9c0a9f` |
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/pairs.csv` | `1d648e3b822ac42e668bdd6273f0e961f04f3a4e65097881428603d33e67f43e` |
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/pair_order.json` | `a942fd048c42ac89a241ee27929f29f37d7599e0c9c3fb114d3d935eeabaf17a` |
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/native_effects.jsonl` | `db3d17b41a6d3322100811f666cf227b0b5612241437e1cc66bc0702d8647551` |
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/model_manifest.json` | `0f6d1e7535362e5d0c4e25de6afcde7629ca8da4d6de130775bb1467a69f7bae` |
| `results/grain_vector_mechanisms/persona_generation_trace_v1/expanded/generations.jsonl` | `e493f6b30fe8aa0a0baa0c11c003b69d3ac0d9114d74ebef0e9e92d590eb9727` |
| `docs/GRAIN_P05_NATIVE_ACTION_PACKET_IDS_V1.json` | `4f3d38693cf8dd2024fc5a8be5a9ee2c7b4f80479bcf230d04e10a1ebe557b87` |
| `results/grain_gptpro_foundation_v3/P05/native_action_r1/effects.jsonl` | `c40d5e6aaeca3064b98c83d41222b1204b90d12dfcf224681b95f4dfa4389bc5` |
| `results/grain_gptpro_foundation_v3/P05/native_action_r1/private_prepared.npz` | `b9ce2b1c8de9365adf45a1cced3cd351dc83d24f230b663104d4aead92940338` |
| `results/grain_gptpro_foundation_v3/P05/native_action_r1/prepare.json` | `48122d3300c986ff8319153751783332f794835eb5e40f3e8a5fc59e11c552bf` |
| `results/grain_gptpro_foundation_v3/P05/native_geometry_r1/native_geometry.json` | `c3352abf6141f39a4d37b32f80787bad33ea922ce62e58259cc6a76fee79138b` |
| `results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors` | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| `results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/config.json` | `98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670` |
| `results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/tokenizer.json` | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/tokenizer_config.json` | `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` |
| `scripts/run_grain_p05_native_action.py` | `c0c237ac8bafede636386ab48acf16b9fc89a05cc27a06dd28c3d60aed3a80d3` |
| `scripts/run_grain_conditional_operator_native.py` | `c3d344b90c70fd91f32f0253fb1368bfdfab83ef21cf59c44b9faf850981754b` |
| `scripts/run_grain_persona_generation_trace.py` | `eb00d7da6e32807c4903510d7c88784f76ed2a09627b267005b5dfa51899af50` |
| `scripts/run_grain_persona_reverse_trace.py` | `1b42018095a6256d0d9a0fb567550ced70c53186b0a0d85e7b40e89a660f6922` |
| `grain_research/vector_mechanisms/native_action.py` | `f640448fd35a762a7e83b720dec95dd339a32ef9e50c5426549c2303cb7c300a` |
| `grain_research/vector_mechanisms/reverse_trace.py` | `a88c3fe8f03019ff39d26125ef45dff4e30b215e63287fe4b389805ed28dd87a` |
| `grain_research/vector_mechanisms/direct_traits.py` | `5a3b83b8a9e832d863d23697f257d3199ded11fc08dbc86809af322088c3cbf4` |
| `grain_research/vector_mechanisms/trace.py` | `b2270a95422184d3c6f4698ac50fa83cb7277cf418ef6f2655c9c45e5c810f28` |

- **Source Jacobian tree** (`native_action_r1/private_jacobians`, 72 files): sha256 over sorted "name:sha" lines =
  `35e1b6d4d305a8e0aacf0f47827e42bf99921515d0fcf295c19c2e8ea649d751`.
