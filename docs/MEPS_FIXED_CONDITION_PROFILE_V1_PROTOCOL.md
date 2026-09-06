# MEPS Fixed-Condition Vector Profile v1: Frozen Protocol

## Implementation erratum, not a retrospective protocol change

2026-09-06: the original shared loader implicitly parsed `mood.csv` even though
the fixed-profile runner did not use its returned values. The corrected runner
uses `include_mood=False`; the loader defaults to that setting, with a synthetic
file-open guard test. Historical outcome values and this protocol's original
decisions are unchanged. Reading the mixed Markdown container is necessary to
recover answers and must be disclosed separately from extracting scale values.
New role-separated material/window work is specified in
[VM06 material protocol](GRAIN_VM06_MATERIAL_AND_ROLE_PROTOCOL_V1.md).

## Question

With one frozen multilingual embedding, do Japanese responses from the same
participant to different shared MEPS prompts show more vector correspondence
than responses from different participants to those prompts?

This protocol evaluates a **within-Japanese, same-session,
protocol-conditioned vector object**. It is not a personality test, a clinical
assessment, a cross-language comparison, or evidence that vectors make
languages automatically equivalent.

## Frozen representation

- Encoder: `intfloat/multilingual-e5-large`
- Revision: `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`
- Input convention for every text: `query: <participant text>`
- Normalization: L2 normalization before dot-product/cosine similarity
- Maximum sequence length: 512
- Runtime: model weights are prefetched separately; the analysis sets
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before raw text is read.

The original E5 usage guidance specifies a `query:` prefix for symmetric
semantic-similarity and feature tasks. The protocol applies that convention to
all MEPS views so prefix role cannot explain a same-versus-stranger result.

## Permitted text route

1. Input root: complete `MEPS+AI_conv_experiment` bundle.
2. Loader auto-resolves `decrypted_output/`.
3. Included: only aggregate `meps_answer` participant answers to Q1, Q2, Q3.
4. Excluded: assistant language, task-assistance chat, free-chat text from the
   primary test, all raw questionnaires, coding files, and `research1_anal`.
5. Free AI chat is permitted only as one **descriptive** free-versus-fixed
   contrast after primary results are complete.

No raw text, source participant identifier, token sequence, or embedding is
written to an output file.

## Primary estimand

For each fixed prompt pair `(a, b) in {(Q1,Q2), (Q1,Q3), (Q2,Q3)}`, let
`e_a(u)` and `e_b(u)` be normalized embeddings for participant `u`:

\[
S_{uv}^{(a,b)} = e_a(u)^\top e_b(v).
\]

The primary discrimination quantity is:

\[
\operatorname{AUC}_{a,b} =
P\left(S_{uu}^{(a,b)} > S_{uv}^{(a,b)}, u \ne v\right)
+ \tfrac12 P\left(S_{uu}^{(a,b)} = S_{uv}^{(a,b)}\right).
\]

It is estimated from all matched diagonal links and off-diagonal stranger
links. A participant-cluster bootstrap resamples rows, not all pair cells, to
give a 95% CI. A row-linkage permutation uses derangements of the right-side
participants and tests the mean matched cosine.

## Opportunity sensitivity

For each left response `u`, stranger responses `v` are restricted to those
whose right-side length satisfies:

\[
|\log(1+L_b(v)) - \log(1+L_b(u))| \le 0.25.
\]

This length-matched AUC is a sensitivity diagnostic. It does not claim to
remove all expression opportunity, context, wording, or prompt effects.

## Promotion rule

The protocol is promoted only if all three fixed-prompt comparisons satisfy:

1. `AUC >= 0.60`;
2. participant-bootstrap 95% CI lower bound `> 0.55`;
3. Holm-corrected row-linkage permutation support at familywise alpha `.05`.

A pass licenses **same-session, protocol-conditioned Japanese vector
correspondence only**. A failure means this small single-session protocol did
not demonstrate the defined correspondence; it does not invalidate Japanese
embeddings or GRAIN generally.

## Execution

Prefetch model weights, before any private text is read:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-large', revision='3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3')"
```

Run locally, after the model is present:

```bash
python \
  scripts/run_meps_v6_fixed_condition_profile.py \
  --input-root 'data_sets/MEPS+AI_conv_experiment'
```
