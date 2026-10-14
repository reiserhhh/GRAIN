# Categorical Observation Geometry for GRAIN

2026-10-08. HM10B1 is a finite observation-law experiment, not a finished
personality instrument. The parent research goal and original scope remain open.

## 1. Three Different Maps

For a fixed query q and text x, the local observer computes

\[
x \xrightarrow{\phi_q} h_q(x)
\xrightarrow{W} \ell_q(x)
\xrightarrow{\mathrm{softmax}} p_q(x)
\xrightarrow{a^T} s_q(x).
\]

Here h is the final normalized hidden state, W contains the ten digit output
rows, and a=(0,1/9,...,1). The same W is used for all queries, but phi_q depends
on the query. These are conditional model readouts, not observed questionnaire
scores. The frozen block16 affine program is a separate approximation to s_q;
it still obtains actual question-token states from the local observer.

Consequently, this packet does not remove the transformer computation. It
identifies maps that later independent programs must preserve.

## 2. Exact Gauge and Observable Quotient

Let Pi=I-11^T/10. Then theta=Pi*ell is the zero-sum natural coordinate. Adding
the same scalar to all logits leaves p unchanged. This is an exact output
symmetry, not population centering and not removal of topic or author variation.

For Wc=Pi*W, the ten digit rows have rank nine in this observer. At the fixed
normalized-hidden-state domain,

\[
\mathbb R^{1536}=\operatorname{row}(W_c)\oplus\ker W_c,
\qquad d=P_Wd+(I-P_W)d.
\]

The quotient R^1536/ker(Wc) has nine observable coordinates. A displacement
in ker(Wc) cannot change the conditional digit probabilities at this head.
It may matter to another output head, an earlier layer, another task, or an
interaction with a later computation. Preserve it: head-null is not a synonym
for psychological noise. These equations do not authorize subtracting an
upstream activation before RMS normalization as an equivalent intervention.

SVD and a distinct difference-row QR construction agree to 1.94e-16 on the
actual four-corner displacements. Full-vocabulary probabilities are not reduced
to these ten rows; candidate mass is retained separately.

## 3. Link Curvature Versus Representation Interaction

For the four cells 00,10,01,11 of a frozen factorial contrast, define

\[
\theta_* = \theta_{10}+\theta_{01}-\theta_{00},\qquad
d=\theta_{11}-\theta_*.
\]

With S(theta)=a^T softmax(theta), the exact identity is

\[
\Delta S=
\underbrace{S(\theta_*)-S(\theta_{10})-S(\theta_{01})+S(\theta_{00})}
 _{\text{link-only curvature}}
+\underbrace{S(\theta_{11})-S(\theta_*)}
 _{\text{action of representation interaction}}.
\]

The second term equals

\[
\int_0^1\{p_t\odot(a-S_t\mathbf1)\}^T d\,dt,
\qquad p_t=\mathrm{softmax}(\theta_*+td).
\]

Both terms are signed and can cancel. Their RMS values cannot be divided into
psychological information percentages. The partition depends on the declared
three-corner endpoint; it is not a unique causal attribution. In particular,
score nonadditivity alone does not establish a new psychological interaction.

The log-partition Hessian is diag(p)-pp^T. The experiment checks the exact
KL/log-partition Bregman identity and the local Fisher quadratic separately;
the latter is not exact finite-displacement KL. These are the finite categorical
specialization of [Wainwright and Jordan (2008), Section 3.5.1](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf),
not a new theorem or a full implementation of their variational algorithms.

## 4. Actual Findings

HM10A supplied 1,024 records / 768 unique texts, 128 cubes and lexical families
source/validation/application=8/4/4. This packet added 2,304 actual forwards for
three fixed queries, no target refits or new human labels. It is a new observer
experiment on old materials, not independent new-text replication.

| Query | Total interaction RMS | Link RMS | Representation action RMS | Affine/native level r | Affine/native interaction r |
| --- | ---: | ---: | ---: | ---: | ---: |
| adapted_na8 | .006845 | .001878 | .005717 | .8126 | .6681 |
| adapted_dejected5 | .011123 | .002155 | .009701 | .7680 | .6541 |
| pre_trust_positive | .008951 | .003150 | .007841 | .9579 | .8271 |

The source-conditional logit correction changes application RMSE from
.005557 to .005437 for anxiety, .009920 to .006537 for dejection, and .008680
to .009677 for trust. All three independently source/application-family
bootstrap MSE-gain intervals include zero. Keep all six declared arms, including
the wrong-key control. This is cold 11-cell completion using cold 00/10/01,
not a predictor with only one new text.

Only 2.33%, 1.41%, and 3.50% of the pooled squared joint displacements are in
the fixed digit head's observable row space. These are coordinate-domain
Euclidean energy fractions, not personality variance shares. High hidden
dimension alone does not make a small visible fraction surprising.

Cross-query natural-interaction cosines are .7873 (anxiety/dejection), .7049
(anxiety/trust), and .6597 (dejection/trust). Corresponding scalar-action
correlations are .5625, .2368, and .0274. Similar natural-coordinate directions
can have different score action because p and grad(S) depend on the query and
text. These descriptive comparisons do not establish shared psychological
traits or independent replications.

## 5. Evidence Qualification Is a Separate Observation Object

The texts have no direct AI-trust evidence. The trust query nevertheless
returns .5862-.7825, has same-word incidence delta RMS .01533, and is the
best-replayed level query. A faithful surrogate can therefore faithfully
reproduce inference made without direct target evidence.

Do not fix this by setting trust to zero, subtracting its score from other
targets, or deleting its hidden directions. Keep the native response for
mechanism research while returning an unobserved direct-evidence status at
the author measurement interface. The observation object should include

\[
\mathcal O_q(x)=(D_q(x),\ \theta_q(x),\ s_q(x),\ \text{event/subject addresses}),
\]

where D records observed, unobserved, or unresolved evidence. Evidence status
must use independently defined event/content/subject support, not a high score
threshold. Off-target inference may be scientifically meaningful, but it is
not interchangeable with an explicitly supported observation.

## 6. Next Scientific Decision

HM10B2 will use new controlled content that independently crosses anxiety
evidence, AI-trust evidence, ownership and relation structure. Freeze old
queries and units. Include self/other same-word controls, no-evidence controls,
lexical source/validation/application separation, and original all-score output.

First ask which query-response components follow direct target evidence and
which follow shared structure/default inference. Then test source-supported
shared and target-specific operators without optimizing queries or thresholds
on application outcomes. Do not replace actual content with arbitrary
simulated psychological coefficients or continue a grammar/RMSE rescue grid.

Personal response B_u requires independent same-author observations and an
identified source design. It remains a separate open layer. Naturally licensed
author data and the original 18/100/44 scope remain open, not moved wholesale
to a human-only backlog.

## 7. Reproduction and Provenance

Canonical primary: results/grain_math_response_v1/native_joint_r3. Preserve r1
(directory initialization failure) and r2 (source-hash freeze rejected after
formatting), both with zero forwards. The primary and its input/source hashes
remain frozen.

Companions are explicitly result-after: native_joint_verification_r1,
native_quotient_diagnostic_r1 and native_specificity_diagnostic_r1. The
alternate-formula verifier checked 4,632 artifact hashes, 15 source hashes,
all native scores, affine projections, contrasts, completions and six interval
endpoints. Maximum algebra error is 6.22e-15. Output-head float32 replay error
is 3.87e-6 and is reported separately. This is same-agent alternate arithmetic,
not third-party independent validation.

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
"$PY" scripts/verify_grain_math_native_joint.py --out /tmp/grain-native-check-new
"$PY" scripts/diagnose_grain_math_native_quotient.py --out /tmp/grain-quotient-new
"$PY" scripts/diagnose_grain_math_native_specificity.py --out /tmp/grain-specificity-new
"$PY" -m pytest -q tests/test_grain_math_native_joint.py tests/test_grain_math_readout_quotient.py
```

All companions require a fresh output path and perform no new observer calls.
No private MEPS reads, API, new neural training, Claude restart, commit, push,
tag, lockbox opening or historical-queue replay occurred.
