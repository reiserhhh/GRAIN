# VM06-RC5: exact task-sensitive readout geometry

2026-09-06. Freeze before RC5 private forwards. RC4 is still running; this packet
does not choose layers, cases, or metrics using its results. Reuse only RC1's
44 exposed replies, actual/matched preceding turns, and fixed assignment.
This closes the readout question, not the entire personality-mechanism program.

## Mathematical object

Use **post-final-normalization, per-token** hidden states, never mean states as
a replacement for next-token computation. At a given response-predicting
position let a and m be actual/matched contexts, z=W h+b, and A(z)=logsumexp(z).
For the observed next token y and delta z=z_a-z_m:

```text
Delta ell_y = delta z_y - [A(z_a)-A(z_m)]
g_m . delta z = delta z_y - E_{p_m}[delta z]
g_a . delta z = delta z_y - E_{p_a}[delta z]
Delta ell_y = g_m . delta z - KL(p_m || p_a)
Delta ell_y = g_a . delta z + KL(p_a || p_m)
g_a . delta z <= Delta ell_y <= g_m . delta z
```

These are exact softmax/Bregman identities, not new personality equations.
The hidden-state pullback metric is
`W.T @ (diag(p)-p p.T) @ W`; the directional curvature equals
`Var_p(delta z)`. Do not fit a Fisher matrix or call its directions traits.
Finite KL is not interchangeable with a local quadratic approximation.
Ordinary state cosine ignores W and p; high cosine alone does not constrain the
target probability change. Its failure would not imply hidden states are uninformative.

## Fixed observations and checks

Only tail tokens2..T are analyzed, using final body states0..T-2; the first token
is predicted by a prefix position and is excluded. The last body state does not
predict an observed response token and is excluded too. Replay RC1's native
mean/tail values within1e-5, then replay the selected token logp after independent
LM-head extraction within1e-5. Same cached model/template/positions and FP32/MPS.

Per token save coded position, never token ID/text: two logp values, their
difference, target-logit/partition shifts, two endpoint-gradient terms, both KLs,
matched-directional curvature, hidden displacement norm, and state cosine.
Compute vocabulary reductions in CPUfloat64 with stable logsumexp, using at most
eight token rows at a time after the existing MPS LM head. Full vocabulary
probabilities and hidden vectors are transient, not serialized.

Independently check both logp identities and KL definitions (<=1e-9), nonnegative
KL/curvature (tolerance1e-9), endpoint concavity, and scalar aggregation. Synthetic
controls include identity, common-logit-offset invariance, orthogonal hidden
coordinate transport, rank-deficient readout, and opposite target effects at
equal displacement norms. Algebraic identities passing are instrument checks.

Report every participant's **mean over tokens**, then equal-person averages
and descriptive1000-person bootstrap(seed731); do not pool1808 tail tokens as
independent people. Report the cosine of the two mean predictor states separately
from mean token cosine. No threshold, layer, seed, or favorable subset selection.
No absolute linear-term/remainder fraction when its denominator is near zero.
Do not claim vocabulary KL measures psychological uncertainty or a new score.

## Execution and boundary

Mac only; wait for RC4 to release MPS, never load both full Qwen copies at once.
RC1 six-forward numerical gate, then fixed EN/JA context pairs(two forwards each)
with all readout checks, before private parsing. Then88 private native forwards:
98 total, cap110,1800sec, sampled MPS<=9GiB. Count separate head-only batches
explicitly; no generation/fit, no extra model download, no paid API.

Private source/model/material replay and original MEPS whitelist apply. Mixed
answer containers are scanned/hashed to extract permitted text; questionnaire
fields and bridge outcomes remain unread. Save a fresh ignored scalar packet,
snapshot code/model/material identity, and perform an independent scalar audit.
No GRAIN production-score change and no raw text/token/full vector persistence.

## Decision

If hidden geometry is close while gradient/partition terms are substantial,
use task-sensitive readout quantities rather than arbitrary Euclidean variance
to interpret contextual displacement. If terms cancel, report cancellation rather
than no information. This is a standard readout mechanism in one small model on
short replies; it neither proves human personality reasoning nor licenses more
layer search on these44 people. Proceed to complete source-only feature/head
comparisons and frozen bridge evaluation after this bounded RC sequence.

Mathematical background: the relation between exponential families and Bregman
divergences is standard ([Banerjee et al., JMLR2005](https://www.jmlr.org/papers/v6/banerjee05b.html)).
This packet uses a direct final readout, not the potentially misaligned use of
an unembedding matrix on intermediate states; see
[Belrose et al., 2023, Tuned Lens](https://arxiv.org/abs/2303.08112).
