# Encoding bias versus evidence-by-encoding response

2026-09-07, round35. One bounded post-hoc analysis of the COMPLETE invented
direct semantic panel. Motivated by observed large coding shifts, not preregistered
as a primary hypothesis. No new prompts, forwards, source targets or calibration.

## Question

Does switching the output convention merely add a content-independent response
bias, or also alter the contrast between different behavioral descriptions?
Different probability changes alone are insufficient: sigmoid curvature can
make the same log-odds offset produce different probability differences.

For each task, order answer logits by their DECLARED trait values. Counter Big5
digits must therefore be reversed before comparing semantic coordinates. MBTI
values stay0/1. Remove only the arbitrary common-logit offset:

```
u_e(x) = z_e_sorted(x) - mean(z_e_sorted(x))
d(x) = u_counter(x) - u_primary(x)
r(x) = d(x) - d(no_evidence)
J(high,low) = d(high) - d(low)
```

The content-independent candidate-bias model `u_e(x)=b_e+g(x)` implies r=0 and
J=0. Subtracting any fixed per-encoding bias leaves J unchanged. These are exact
algebraic necessary conditions, not p-values or novel psychometric theorems.
For binary tasks also report log odds `l_e=z_positive-z_negative` and its mixed
contrast `(l_counter(high)-l_counter(low))-(l_primary(high)-l_primary(low))`.

Report all nine tasks, four behavioral cases and two pairs each. Preserve both
the no-evidence-relative residual and reference-independent pair contrast, since
the single authored no-evidence string is not a neutral person or a population
prior. Different answer dimensions have different geometry; do not pool Big5
and MBTI RMS magnitudes into a single performance score. All units are logit/log
probability contrast units (natural logs), not personality effect sizes.

No outcome threshold, fitted coefficient, corrected prediction, prompt winner,
new native personality direction or layer search is produced. Nonzero J means
that exact semantic candidate-response invariance cannot be recovered by a
constant vector bias alone on these cases. It does NOT mean all calibration is
useless, that scalar rankings cannot improve, or that the hidden state lacks
relevant information. The contrast cannot locate the computation responsible.

## Literature alignment

Zhao et al. estimate answer bias using content-free inputs and improve few-shot
classification with contextual calibration. This motivates examining our existing
no-evidence controls, not assuming that their intervention or gains reproduce in
our zero-shot trait wording and1.5B observer.
[ICML2021](https://proceedings.mlr.press/v139/zhao21c.html)

Holtzman et al. distinguish surface-string probability from answer correctness
and use task-conditioned prior reweighting. Our high candidate mass already
limits an outside-alphabet explanation in this panel; we have not manipulated
synonyms or reproduced their surface-form competition experiment.
[EMNLP2021](https://aclanthology.org/2021.emnlp-main.564/)

Both are methodological precedents, not evidence of human-personality validity.
The actual direct source protocol remains unchanged. Complete the original
history/task/direct source families and frozen-bridge comparison as planned.
