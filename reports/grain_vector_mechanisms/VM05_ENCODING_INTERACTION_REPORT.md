# Coding changes are not only a constant answer-prior shift

2026-09-07, round35. Bounded POST-HOC diagnostic of the completed invented
semantic panel.0 new forwards,0 source labels,0 fitted calibration parameters.

## Finding

For this pinned observer and these examples, a content-independent vector bias
does not explain the entire primary/counter difference. The semantic-aligned
candidate-logit contrast between the high/low texts also changes. A fixed bias
subtraction cannot make the full candidate-response vectors exactly invariant.
This does not rule out improved scalar ranking or accuracy from calibration.

Let u be logits sorted by DECLARED trait value, with their common mean removed.
Big5 counter digits must therefore be reversed before comparison. This centering
only fixes the softmax gauge, not author/topic variation. Define

```
d(x) = u_counter(x) - u_primary(x)
r(x) = d(x) - d(no_evidence)
J_j = d(high_j) - d(low_j)
```

Under `u_e(x)=b_e+g(x)`, r and J are zero. J is independent of the chosen
no-evidence anchor and unchanged by subtracting any fixed bias vector from either
encoding. Binary J_logodds is the mixed positive-minus-negative logit contrast.
It avoids sigmoid-curvature artifacts.

## All tasks

RMS values are over semantic candidate coordinates and four cases or two pairs,
NOT correlations, personality effects, p-values or accuracy. Do not pool the2-
and10-dimensional geometries into one effectiveness score.

| Task | Candidate dimension | No-evidence bias RMS | Anchored residual RMS | Mixed-pair RMS |
|---|---:|---:|---:|---:|
| Extraversion |10|2.0975|2.9496|5.5612|
| Neuroticism |10|1.2976|1.5222|1.1996|
| Agreeableness |10|1.4994|3.4224|6.3410|
| Conscientiousness |10|2.0980|3.4232|6.2482|
| Openness |10|1.2253|3.9032|7.6763|
| EI |2|2.8750|.7967|.7526|
| SN |2|2.0000|.6796|1.3118|
| TF |2|2.9375|.5087|.7194|
| JP |2|4.1250|1.4042|1.9330|

The two mixed log-odds contrasts are EI +2.125/-.125; SN -3.125/-2.000;
TF +1.375/-1.500; JP +5.375/+1.000. All18 multivariate pair contrasts are
nonzero at the recorded precision. The weak EI -.125 value is near discrete
logit resolution, not established precision-robust evidence. This packet does
not bound accumulated BF16 error against an FP32 counterfactual. Larger deviations
establish failure of the exact constant-bias model for the recorded instrument;
no population statistical test was performed.

Large no-evidence biases coexist with nonconstant effects. Their relative
magnitudes are not a variance decomposition, mediation fraction or percentage of
personality information. Only two authored pairs per task exist; wording, length
and position influences have not been independently isolated.

## Interpretation and prior work

Contextual calibration estimates answer bias using content-free prompts. Our
control makes a related diagnostic possible, but nonconstant residuals neither
reproduce nor invalidate the original few-shot results. No predictions were
corrected here. [Zhao et al., ICML2021](https://proceedings.mlr.press/v139/zhao21c.html)

Surface-form competition distinguishes string likelihood from correctness. Our
high candidate mass limits an outside-alphabet explanation here, but synonym
competition has not been isolated experimentally.
[Holtzman et al., EMNLP2021](https://aclanthology.org/2021.emnlp-main.564/)

The conclusion is narrowly empirical: these trait prompts exhibit input-by-output-
convention interactions, not only fixed offsets. This does not localize attention
circuits, identify hidden personality dimensions, or show that trained vector
heads reproduce native reasoning.

Cross-convention instability is NOT fixed-convention irreproducibility: original
repeated forwards were exactly equal. A fixed protocol might still support a
useful calibrated instrument, retaining model, prompt, coding and reference
identity. This reinforces GRAIN's operator-relative measurement spine, not an
impossibility claim about stable LLM-assisted assessment.

## Reproduction and next action

Implementation881a615:
`scripts/analyze_grain_direct_encoding_effects.py`,
`grain_research/vector_mechanisms/direct_encoding_effects.py`.
Protocol GRAIN_VM05_ENCODING_INTERACTION_DIAGNOSTIC_V1.md is explicitly post-hoc.
Artifact vm05_encoding_interaction_r1, identity
533bc4e9437caf67b243ac8186dc9d8b6f3a9b4207b1606bbf4d94fa6297dd49.

Independent reconstruction uses logs of saved conditional probabilities rather
than centered raw logits and separate norms/cell contrasts: all9 tasks/36 cases/
18 pairs, max difference6.22e-15. Artifact
vm05_encoding_interaction_independent_r1, identity
1f7a0f9e7cbdce11ded61b3eef751e8484db2a84998496c93d1b7fd8a6746091.
This is arithmetic replay, not another neural or precision-sensitivity run.

Four new synthetic tests plus five existing direct tests pass9/9: additive-bias
null with misleading probability curvature, true mixed effect, fixed-shift/anchor
invariance, common-logit and answer-permutation invariance, invalid panel/coding.
Original scope remains: task/history/direct source comparisons, stronger nulls
and source-frozen bridge. No post-hoc rescue or claimed natural-accuracy gain.
