# HM150: actual joint persona/style/task realization

Frozen before any generation. Original18/100/44 scope remains OPEN.
No accuracy/head/rank/prompt/metric selection, API, private data, training,
Git, lockbox, Claude, automation or old-generation restart.

## Inventory and question

HM147-149 used270 historical outputs: persona and nonpersona style instructions
were separate. Their empirical secant overlap cannot identify moderation.
HM10A/10B manipulate report/owner/order; HM105/106 and119 manipulate behavioral
content/owner/focus and explicit reading products. None supplies jointly
GENERATED persona x style articles. Their algebra is reused, not rediscovered.

Question: on actual joint outputs, is persona expression additive with style,
or is there a repeatable target-by-style response conditioned on task/wording?
The assigned persona is a generator intervention, NOT a human trait label.
Nonpersona styles are NOT assumed psychologically irrelevant.

## Complete materials and finite runtime

Use the original pinned Qwen2.5-1.5B-Instruct weights/tokenizer, float32 MPS,
offline, evaluation mode. Reuse the historical manifest's exactly aligned
high/low statements and all6 task prompts, not old article outputs.
2targets x2poles x3families x6tasks x3styles x3seeds =648 actual articles;
54 neutral-persona articles complete the style controls: TOTAL702.
Style is neutral, newsletter, spelling. Every system prompt contains the same
shared first-person/no-trait-name suffix; style follows persona. The neutral
style adds no style statement. All combinations, failures and refusals remain.

54 immutable batches of13 texts: one task/seed/style, all12 persona conditions
plus one neutral persona. Fixed left padding; seed11/23/37 reset per batch.
Style batches have matching row order, but identical seed is NOT a guarantee
of identical lexical noise or position-aligned articles. Seeds are generation
replicates, NOT authors. Cross-task/cell comparisons are not independent people.
Sampling temperature.8/top_p.95, max192 newtokens, as historical generator.
No length rescue, stripping, regeneration or outcome-dependent filtering.
Whole collection has2400 model/wall seconds cap after load; atomic batches
may finish past cap, then no next batch. Resumable completed batches never rerun.
Observe stage600s cap; incomplete material support is reported, not imputed.

## Observations, kept separate

1. Text-only Qwen input-embedding mean and block16-output token mean. Actual
   generated article tokens only, NO persona/style/task prompt or trait query.
   These are1536-dimensional contextual/vector observations, not answers.
2. Historical frozen4-slice GRAIN coordinates. Its representation runtime is
   TF-IDF -> SVD, NOT E5 or Qwen activations. Preserve original text characters
   and require48 lexical tokens/four supported slices; insufficient text is
   unobserved, not0. No author/reference fitting or application re-centering.
3. Existing HM147 three score arms applied in their frozen source units. The
   current packet does NOT refit them against the new generator outputs.

Record every raw output, token IDs, length, stop reason, disclaimer/trait-word
audit. Each model/representation/bundle and the old program is hashed. New
observations are sealed before group comparisons. No tensor-only artificial
corner may be called an observed article.

## Discriminating mathematics

For a fixed task/family/seed, actual corners are h_s,h_0,l_s,l_0. For each
declared observation map Psi:

    Gamma_Psi = Psi(h_s)-Psi(h_0)-Psi(l_s)+Psi(l_0).

Keep persona contrasts in every style and style contrasts at both poles and
neutral persona. Gamma is a finite randomized-instruction interaction on this
generator, not automatically a human causal/personality interaction.

Raw-vector Gamma and tensor-feature Gamma are distinct. For GRAIN's frozen
source z, define NUMERIC additive completion h_s*=h_0+l_s-l_0. Then

    Gamma_Phi = [Phi(h_s*)-Phi(h_0)-Phi(l_s)+Phi(l_0)]
                +[Phi(h_s)-Phi(h_s*)].

Also report the exact opposite-anchor completion l_s*=h_s-h_0+l_0. These
diagnostics separate nonlinear observation curvature from nonadditive input;
they do not generate valid articles or allocate unique causal percentages.
All3 old arms and both targets are kept, including off-target and signed terms.

For3 seed interactions I_s in the SAME fixed observation units, report

    U = sum_(s!=t) <I_s,I_t> /6,
    ||mean I||^2 = U + sum_s ||I_s-mean I||^2 /6.

Negative U remains. This distinguishes mean-direction support from within-run
energy; it is not a human significance test. Shared deterministic units and
finite generator seeds do not justify universal information/noise fractions.
Qwen observations stay in raw hidden units; RMS divides by sqrt(d), not a
new fitted whitened metric. Compare within, not across incompatible units.

## Decision and next construction

If Gamma repeats in the text-only internal vectors and GRAIN, retain a
conditioned realization operator rather than a universal noise projection.
If only Phi/score Gamma repeats, revise the observation law before claiming
modulation in the base vectors. If seed means do not correspond, collect no
precision rescue: retain stochastic realization/unsupported corners and
identify the next missing semantic observation. All tasks/families remain.
Psychological grounding and real-author transport stay separate full-scope
requirements. A higher classification accuracy is not an acceptance criterion.

## Ancestry

Chen etal2025 Persona Vectors already extracts/steers model character directions:
https://arxiv.org/abs/2507.21509 . This is not the first persona-direction study.
Sundararajan etal2020 distinguishes interaction attribution from main effects:
https://proceedings.mlr.press/v119/sundararajan20a.html . We use elementary
finite differences, not their full Shapley-Taylor method.
Gretton etal2012 provides kernel mean witness mathematics:
https://www.jmlr.org/papers/v13/gretton12a.html .
OpenAI/math pinned adc7f124 Universal-Tensor-Squares preliminaries21-100 was
reread: component-sensitive dual tests motivate keeping multiple observation
maps. No Specht/group/PDE theorem is imported into psychology or claimed new.

## Deliverables

New joint_persona_style_r1 with sealed design,54 generation batches,
raw-text-only observations, complete coverage/contrast/repeatability tables,
curvature/nonadditive score accounts, alternate arithmetic verifier and report.
Tests cover complete702 design, null/additive/modulated worlds, exact tensor
link identity, negative cross-seed U, missing corners and score reconstruction.
Never overwrite frozen HM147-149 or the original generator experiment.
