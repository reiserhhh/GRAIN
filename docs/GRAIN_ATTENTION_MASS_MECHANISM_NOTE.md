# Attention mass as readout gain: a concrete follow-up mechanism

2026-09-08. Source-guided theory during S15, not an already measured sink claim.

## What the data prompted

The label-free S15 pilot shows that uniform-mass covariance badly mis-scales the
frozen reader, whereas exact attention masses make the same conditional
covariances much more faithful. The full cohort remains the comparison source.
This directs attention toward a multiplicative control, not only richer values.

Let B=sum_G P_G be all declared comment-body attention mass and r_G=P_G/B.
For positive finite softmax weights and nonempty groups, B>0, and exactly

\[
A(q)=B(q)\sum_G r_G(q)D_G(q),\qquad
D_G(q)=E[V\mid G,q]-E_G[V].
\]

The three objects are distinct: overall body gain B; relative allocation among
comments r_G; conditional relational response D_G. Native S15 exact masses
combine the first two, so their contribution is not yet separated by that run.
Replacing P_G by |G|/n changes both. A next experiment can retain B but set
r_G=|G|/sum_H|H|, keeping the same conditional covariance. This isolates total
gain from which comment is read, without fitting new coefficients or using Y.

This is also an exact refinement of GRAIN's conditional-method perspective:
preserve the conditions controlling a reading operation, then study the
information it reads. It is not a justification to remove topics or individual
deviations, and a computational gate is not a psychological emotion gate.

## Relevant primary research

- Xiao et al., ICLR2024,
  [Efficient Streaming Language Models with Attention Sinks](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html):
  early tokens can absorb attention without semantic salience; retaining their
  cached states stabilizes streaming behavior. Borrow the distinction between
  normalization function and semantic importance, not their long-context result.
- Gu et al., ICLR2025,
  [When Attention Sink Emerges in Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f1b04face60081b689ba740d39ea8f37-Abstract-Conference.html):
  investigates softmax normalization and key-bias-like sink behavior across
  pretraining settings. This motivates measuring denominator allocation instead
  of treating an attention map as a direct personality explanation.
- Ran-Milo2026, preprint,
  [Attention Sinks Are Provably Necessary in Softmax Transformers](https://arxiv.org/html/2603.11487v1):
  a trigger-conditional averaging/no-operation task yields sink necessity for
  its specified softmax model classes and input distribution, with a ReLU
  construction for contrast. The proof uses uniform small task error, continuous
  independent content coordinates and declared architecture assumptions. It does
  not prove every head in our pretrained Qwen must contain a sink.

The third source's useful idea is conditional suppression through normalized
weights, not a license to transfer its theorem to personality inference.

## Small next measurement, only after S15

Use the existing fixed24-author source pilot, not a fresh questionnaire search.
At the same common query, save per-head body mass, outside mass, first-token mass,
normalized comment-mass entropy, and contributions to the existing scalar reader.
Compare exact-mass covariance with body-gain-only covariance and uniform mass.
This determines whether the S15 gain comes mainly from a whole-head gate or
from different allocation among comments. Retain exact within-comment reading
as a replay comparator. No semantics need to be assigned to individual tokens.

Do not change S15 while it is running or preemptively add the fifth arm to its
frozen results. If the full result points instead to conditional higher moments,
develop that branch first. No new long queue is implied by this note.
