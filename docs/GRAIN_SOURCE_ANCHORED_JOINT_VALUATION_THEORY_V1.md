# Source-anchored joint GRAIN valuation

HM147, 2026-10-11. Finite construction and descriptive mechanism evidence.
This document does not establish a human personality instrument. The original
18 protocols / 100 cards / 44 computational questions remain open.

## 1. Scientific change, not an accuracy upgrade

HM146 found person-sensitive native operations at a fixed complete token bag.
Its K/V interventions transported those operations with high fidelity, while
the supplied joint diagnostic meaning did not follow. Carrier fidelity and
diagnostic meaning are therefore different interfaces in this implementation.

HM147 constructs a different, executable interface: known source prompt
assignments orient a function of joint GRAIN slice observations; a separate
numeric scorer applies that function to texts from other tasks. No native LLM
answer supplies its target or scoring rule. This is not a rescue of HM146's
reader, another role classifier, or a new persona-detection AUC comparison.

Keep three maps separate:

1. Qualification: which observed event belongs to which author, in what role,
   modality, condition and occasion. Previous event-ledger results cover parts
   of this interface; HM147 does not newly solve natural-text qualification.
2. Joint representation: how qualified observations are retained together,
   including conditional selection, level, order and dependence.
3. Target valuation: which source-supported meaning acts on that joint object.

This packet supplies a finite version of map 3 on existing generated-text
vectors, retaining map 2. It does not identify a human trait observation law.

## 2. Actual source support

The historical expanded persona trace contains 270 generated texts, each with
four cached four-coordinate GRAIN slices. Its original task split is preserved:

| Role | Tasks | Texts |
| --- | --- | ---: |
| Source/discovery | diary, email, opinion | 135 |
| Development, not used for selection | advice | 45 |
| Evaluation | review, neighbor | 90 |

Source high and low groups each contain 27 texts per target: extraversion and
neuroticism generator assignments. Task, phrasing family and generation seed
are balanced. Controls and AI-disclaimer cases remain. Seeds 11, 23 and 37 are
generation replicates, not people. These historical data have been examined
before; cold scoring means separation of scoring inputs, not a blind new cohort.

Only cached numeric slices and design metadata are read. The new program does
not read article bodies, system prompts, native/DeepSeek answers, questionnaires
or private records. Upstream vectors remain products of the frozen GRAIN text
representation and factor bundle; mathematical downstream scoring is not an
end-to-end removal of the text representation.

## 3. Fixed source units and finite tensor map

For source observations \(g_{iv}\in\mathbb R^4\), slice \(v=1,\ldots,4\), define

\[
\mu_v=\frac1n\sum_i g_{iv},\qquad
s_v^2=\frac1n\sum_i\|g_{iv}-\mu_v\|^2,\qquad
z_{iv}=\frac{g_{iv}-\mu_v}{s_v}.
\]

One scalar RMS unit per slice is frozen before application. There is no
application re-centering, per-task residualization, or deletion of conditions.
This changes the reference origin and units; condition variation stays in z.

Use the finite affine tensor feature map

\[
\Phi(z)=\bigotimes_{v=1}^{4}(1,z_v)\in\mathbb R^{625},\qquad
K(z,z')=\prod_{v=1}^{4}(1+\langle z_v,z'_v\rangle).
\]

The kernel equals the explicit tensor inner product. It is not characteristic
over arbitrary probability distributions. For a target k, source assignments
alone define

\[
a_{ik}=\begin{cases}
1/n_{k+}&i\text{ assigned high k},\\
-1/n_{k-}&i\text{ assigned low k},\\
0&\text{otherwise},
\end{cases}
\qquad
w_k=\sum_i a_{ik}\Phi(z_i),\qquad
f_k(z)=\frac{\langle w_k,\Phi(z)\rangle}{\|w_k\|}.
\]

This is a normalized empirical kernel-mean witness. Its output is an unbounded
function value, not Pearson r, a probability, a Bayesian posterior, or a
psychological scale score. Its mean source diagonal high-minus-low contrast
equals \(\|w_k\|>0\) by construction, so that source contrast is not validation.

## 4. Exact component bookkeeping and pairing experiment

For each subset \(S\subseteq\{1,2,3,4\}\), retain the tensor coordinates in
which precisely slices in S have a nonconstant coordinate. The 16 disjoint
sectors are orthogonal under the specified feature inner product:

\[
w_k=\sum_S w_{k,S},\qquad
f_k(z)=\sum_S\frac{\langle w_{k,S},\Phi_S(z)\rangle}{\|w_k\|}.
\]

Three fixed arms use the SAME original full-witness denominator:

- Full: all sectors.
- Single: constant and order-one sectors of that same witness, no refit.
- Source-relinked: cyclically relink source seed assignments across slices by
  shifts [0,1,2,1] within each task/phrasing/condition cell, then form the witness.

Relinking preserves every conditional per-slice empirical multiset and class
balance exactly. It changes joint seed pairing only. It is a numeric
recombination, not an intervention on naturally produced language. The single
coefficients agree to 6.25e-17; higher-order differences isolate dependence on
this pairing at the specified units.

Higher-order here means a tensor sector, not automatically irreducible synergy,
path motion, personality, or a psychological interaction. Products of means,
content and task structure can occupy those sectors. No sector is selected
according to application performance.

Applying independent orthogonal transformations to each slice in BOTH source
and application preserves the scalar units, sector inner products and scores.
This is ordinary Euclidean/tensor covariance. It does not imply invariance to
arbitrary nonlinear embedding changes or language changes.

## 5. What the cold outputs change

The standalone scorer receives only `program.npz` and arrays named `views`.
Application labels, IDs, tasks and text are absent; they are joined afterward
only to summarize high-minus-low generation pairs.

| Evaluation target | Full | Single | Source-relinked |
| --- | ---: | ---: | ---: |
| Extraversion, 18 pairs | -0.017085 | +0.039086 | +0.085930 |
| Neuroticism, 18 pairs | -0.113990 | +0.008743 | -0.115182 |

These are witness units, not correlations. Joint terms can reverse the source
meaning rather than merely amplify a single-slice score. Task-local positive
effects also remain: review/anchored extraversion +0.595845 and review/named
neuroticism +0.555010. Each cell has only three paired generation seeds. Neither
pooled negatives nor selected positives justify a universal conclusion.

The data support a specific next question: does the source semantic function
depend on task affordance or a task-by-target observation law, rather than only
on a common direction whose numerical estimation needs higher precision?
Nonpersona style controls also move the function. A shared scalar interpretation
across tasks cannot be assumed from an executable vector score alone.

## 6. Finite seed-support identity

In identical source units, construct each balanced source seed witness \(w_s\).
For S seeds (here S=3), define

\[
\bar w=S^{-1}\sum_s w_s,\quad
U=\frac{\sum_{s\ne t}\langle w_s,w_t\rangle}{S(S-1)},\quad
V=\frac{\sum_s\|w_s-\bar w\|^2}{S-1}.
\]

Then, in every sector and order,

\[
\|\bar w\|^2=U+V/S.
\]

This exact finite identity separates signed cross-seed agreement from observed
dispersion. Source units were estimated jointly, and seeds are not independent
human samples; U is not an unbiased population signal estimator or a human
personality information fraction. Negative U values are retained.

| Order | Extraversion U | Neuroticism U |
| --- | ---: | ---: |
| 1 | +0.051267 | -0.021954 |
| 2 | +0.315138 | +0.125172 |
| 3 | +0.056939 | -0.011455 |
| 4 | +0.025645 | -0.010523 |

The neuroticism pair-order direction has positive source cross-agreement even
while the pooled new-task direction reverses. Source repeatability and task
transport are thus separate empirical questions; neither is solved by reporting
the higher-order source squared norm alone.

## 7. Mathematical ancestry and possible contribution

Kernel means, witness functions and product kernels are established methods;
see [Gretton et al., 2012](https://www.jmlr.org/papers/v13/gretton12a.html).
The use here does not import their universal consistency or testing guarantees.
Bound propositions versus final model outputs are also an established
distinction; see [Feng, Russell and Steinhardt, 2024](https://arxiv.org/html/2406.19501v1).

The pinned [OpenAI/math](https://github.com/openai/math) snapshot
`adc7f1241b42e322a6451854ab7e4b4c146bf78a`, the Universal Tensor Squares
preliminaries, inspired component-sensitive dual contractions. No symmetric-
group theorem is used as a theorem about text, personality or neural inference.
This packet contains no new mathematical theorem or first persona classifier.

The research contribution under investigation is the joined construction
procedure: qualified joint evidence, source-supported target meaning, cold
numeric application, and task-relative interpretation kept as separate maps.
Its value depends on explaining the observed transport changes, not on beating
the previous reader's accuracy or claiming advanced notation as novelty.

## 8. Next discriminating mechanism

First inventory existing conditional-response, coordinate-transport, qualification
and semantic-action experiments. A new experiment must distinguish competing
predictions about task-relative meaning using source-supported conditions and
fresh or separately withheld target contrasts. Preserve choice, state, dynamic
and signed joint components. Do not automatically fit within-task classifiers,
select pair sectors, average seeds to rescue scores, or sweep bandwidth/rank.

The full natural-text/personality construction question remains open. That is
the next missing map, not a reason to discard the executable joint function.
