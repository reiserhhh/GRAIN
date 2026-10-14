# HM124: GRAIN-represented versus unrepresented native response

## Question and decision

Do the current anonymous GRAIN coordinates retain the source vector component
that changes native target judgments, or is that component in their linear
complement? This is a measurement-interface question, not a fidelity search.
It extends HM119--123 from vector probes to actual frozen-model digit logits.

Use96 original HM115 source records and their three four-coordinate slices.
No held responses, logits, gradients or current states enter the source fit.
For source current value matrix X and A=[1,GRAIN three-slice coordinates],

    P = A A^+; X_G = P X; X_R = (I-P) X.
    D_(f,c) = average_(wording,b,owner) b*owner*X_(f,c,b,owner).
    D = D_G + D_R.

P is an orthogonal projector in the96-record SOURCE SAMPLE space, not a
projector in neural feature space. X_R is linearly orthogonal to A on source;
it is NOT guaranteed independent, free of GRAIN information or psychologically
meaningful. All components and contexts remain preserved. This is not LEACE.

## Fixed finite experiment

48 new public authored records: six behavior families x two paraphrases x two
polarities x two actors. Equal voluntary opportunities for both actors, two
prespecified equivalent opportunity wordings. No trait names in the bodies.
Source uses both original wordings and both conditions; application tests only
the source voluntary condition. No unrestricted cross-condition claim.

Each body has a native current carrier identical to source and12 actual target
questions (six targets x queried actor A/B). At layer17 current-token V, perform
five arms: baseline, subtract b*owner*D_G, subtract b*owner*D_R, subtract
b*owner*D, and a per-head-norm-matched random substitute for D_R.
All edits retain keys, padding, positions and original text. This is a fixed
hidden-state edit, not equivalent to erasing a natural psychological variable.
Physical amplitude is the original source response, not optimized on application.

The application FAMILY, polarity and actor indices are CONTROL DESIGN metadata
used to select the intervention. They are not inferred by GRAIN. This packet
cannot deliver autonomous personality inference from unknown natural text.

For each arm, retain the full6x6 family/target response matrix and score,
log-odds, candidate-digit probability mass and conditional-digit entropy.
Behavior x actor x queried-person coefficient uses all8 corners of b/o/focus.
Report changes from baseline, signed projection on baseline, norm and all
off-diagonal responses. Nonadditive whole-minus-parts is retained, not split
into a unique causal percentage. Source decomposition norms are not psychological
variance explained, and native digit scores are not human scale scores.

## Acceptance and stop

All48 records and240 native batches/2880 endpoints, or exact partial coverage.
Cumulative900 model seconds, including load and failures; never renew the cap.
Do not rerun ambiguous or terminal units. No thresholds/heads/ranks/doses selected
after viewing results. Source projected/remainder reconstruction and orthogonality
must hold numerically. Actual edit, key/pre-edit state and attention support must
match baseline; native units must independently replay from saved logits.

Mechanism decision uses magnitude and matched-random contrasts for BOTH
paraphrases and both score/log-odds units, retaining inconsistent cases. It is
not a human statistical-significance test or a requirement to get a positive
result. If full source actions fail to carry meaningful native action to new
expressions, neither component is declared sufficient: refine conditional
observation design, not this predictor's accuracy. No whole-goal promotion.

## Prior art and mathematics

Source least squares and factorial response accounts are established tools.
[Amnesic Probing](https://aclanthology.org/2021.tacl-1.10/) distinguishes encoded
information from information used in behavior. Our sample-space source split
is not its iterated classifier-nullspace erasure.
[LEACE](https://papers.nips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html)
shows why erasure guarantees require specific covariance conditions; we do not
claim perfect erasure or independence.
[Park et al.](https://proceedings.mlr.press/v235/park24c.html) distinguishes
representation, measurement and steering; no causal inner product is estimated.
OpenAI/math pinned matrix-jets section71--108 inspires testing whether the
admitted observations cover useful complementary directions. Its harmonic PDE,
constant-rank and uniform-inverse hypotheses are not transferred to text:
https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex

Potential contribution is the explicit GRAIN/native interface distinction and
what it implies for future author measurement, not new projection mathematics.
