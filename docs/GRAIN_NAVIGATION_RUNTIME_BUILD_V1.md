# GRAIN portable navigation program: build and scope

Later source-affine selector: [build and scope](GRAIN_SOURCE_AFFINE_BUILD_V1.md).
It has its own bundle; do not pass source_affine to the older four-program
runtime_export_r1 bundle. The original four-program export below is unchanged.

2026-09-09. Local NAV11 research export. The bundle reproduces **specified**
frozen candidate functions from raw author excerpts, not a finished universal
personality interpreter. Results and limitations are in the
[round4 report](../reports/grain_vector_mechanisms/NAVIGATION_ROUND4_REPORT.md).

## Actual input-to-output computation

1. Input JSONL: one record per anonymous author/view/query:
   {"id":"example","texts":["first excerpt","second excerpt"],"query":"TF_cont"}.
2. A frozen query-specific recipe preserves the original system/user/question
   bytes and excerpt numbering. Rust tokenizers reads the local tokenizer.json.
   Token IDs and question coordinates must match the exported query definition;
   inputs beyond3056 tokens raise an error rather than truncate.
3. The selected program computes one of the following maps.
4. Output contains score, effective projections/components, token count and
   program identity. Score is **unclipped** in the original local-function units,
   not a questionnaire percentile or calibrated MBTI probability. F is the
   positive T/F direction. Author IDs and questionnaire labels are not features.
5. Runtime also writes a dependency report. It cannot refit norms, choose a
   method using human outcomes, generate explanations or silently change queries.

## The explicit formulas

For query q and construction chart c, let R_c be reference token representations,
X the new representations and J_c two pullback derivative rows.

~~~text
A_c(X)_ij = softmax_j[
    8 cosine(R_ci,X_j) - 4 (relative_position_i-relative_position_j)^2
]
z_c(X) = z_c(origin) + <J_c, A_c(X) X> - <J_c, A_c(R_c) R_c>
f_hat_q(X) = b_q + sum(mean_c z_c(X))
~~~

The subtraction rebases the correspondence map itself, guaranteeing identity at
a chart origin. In token mode, X is the pretrained token embedding table lookup.
No contextual Transformer runs. Query P tokens and the final input slot are
excluded from the evidence-pooling region; their fixed effects are in the chart.
This is a specific local approximation, not an exact token-space theorem.

In context4 mode, X is the actual output of four extracted decoder blocks.
Evidence uses the same alignment; the query rows and final slot add exact
slot-matched first-order differences:

~~~text
z_c = z_origin + aligned_evidence_delta
      + sum_query_slots J_cj (h_j - h_ref,j)
      + J_c,last (h_last - h_ref,last)
~~~

The mean of the two frozen construction charts is used. The earlier experiment's
affinity gate is not selected using human outcomes. Contextual inference retains
attention, MLP, normalization and positional operations in those four blocks.

For the generated-state program, x stacks early query slots and final slot.
Construction-only paired snapshots fitted:

~~~text
y_hat(x) = x + mean(R) + centered_k(x,U) W
R = Y-X
W = (centered_K + I)^(-1) centered_R
k1(u,v)=u'v ; k2(u,v)=u'v + (u'v)^2/2
~~~

If the fixed reader is f(y)=b+yL+offset, compile it **before runtime**:

~~~text
f_hat(x) = b + xL + mean(R)L + offset + centered_k(x,U) (WL)
~~~

Runtime thus does not materialize the full later-layer P/Q estimate. This exact
compilation of the state estimator is different from claiming the estimator
equals the original Transformer. state1/state2 are retained negative-transfer
candidates, not the recommended interpretation of every real text.

## Build from the actual research artifacts

Run in the independent GRAIN repository, with the existing project-persona venv.
Construction provenance is traced through these existing scripts and outputs:

| Step | Script | Frozen artifacts |
|---|---|---|
| Controlled scenes/references | run_grain_navigation_baseline.py | NAV-01/controlled_r1 |
| Token derivative charts | run_grain_projection_program.py | NAV-03/input_pullback_r1 |
| Contextual derivative charts | run_grain_context_program.py | NAV-03/context_pullback_r2 |
| State snapshot operator | run_grain_state_transport.py | NAV-02/state_transport_r1 |
| Compile/package/replay | export_grain_navigation_runtime.py | NAV-11/runtime_export_r1 |

The exporter verifies source hashes, copies only runtime assets, compiles state
readers, and confirms raw rendering on1/2/12 excerpts including Unicode.
No LLM calls or human-label fitting are needed to re-export existing artifacts.

~~~bash
PY="../project persona/.venv/bin/python"
"$PY" scripts/export_grain_navigation_runtime.py prepare \
  --out results/grain_vector_mechanisms/navigation_v1/NAV-11/my_export
"$PY" scripts/export_grain_navigation_runtime.py replay \
  --out results/grain_vector_mechanisms/navigation_v1/NAV-11/my_export --program token
~~~

Use a fresh output directory. The observed completed export is runtime_export_r1;
do not overwrite it or rerun preparation into that directory.

## Run independently

The exported bundle includes run.py, grain_runtime/, tokenizer.json,
queries.json, manifest.json, parameter files and this build guide. It does not
need a checkout of GRAIN or the full pretrained observer.

~~~bash
python /absolute/path/to/bundle/run.py \
  --bundle /absolute/path/to/bundle \
  --input /absolute/path/to/new_excerpts.jsonl \
  --output /absolute/path/to/new_scores \
  --program token
~~~

Choose --program context4 --device mps for the four-block program on Apple
Silicon, or --device cpu on an appropriate CPU host. state1 and state2 reproduce
the compiled state estimates. CUDA/Vulkan/Intel GPU backends are not supported
by this small export CLI; do not infer support from a machine's VRAM size.

Token mode requires NumPy, SciPy, tokenizers and threadpoolctl, not Torch or
Transformers. Context/state modes additionally need Torch, Transformers and
safetensors. Exact observed dependency versions and file hashes are in
manifest.json; no automatic download or installation occurs.

The fixed12-author norms from NAV08 belong to the original native/functional
readers. Do **not** reuse those means/scales for a different exported candidate.
New candidate norms require an explicitly declared independent reference cohort.

## Evidence and transfer scope

Observed replay:15 raw inputs per mode,60 comparisons total, maximum2.95e-15
score error, zero attempted source-data/model reads. Static mode imported neither
Torch nor Transformers; other modes executed four blocks without a LM head.

The bundle contains only three initial queries: E, T/F and adapted low state.
Prior external results belong to particular candidates/materials:
token-only semantic/order T/F AUC=.7047(n64); context4 mean low-state r=.3506(n46).
These are exposed exploratory endpoints, not fresh held-out validation.
The exporter does not claim either result for all queries or new domains.

Training-time construction still used a pretrained LLM's derivatives and
synthetic scene snapshots. Runtime independence is not model-origin independence.
All weights/charts remain local ignored artifacts. Check the source checkpoint
license before any redistribution; no pretrained weights or private datasets
are committed or pushed as part of this delivery.
