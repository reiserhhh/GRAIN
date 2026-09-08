# M25: matched query functions and fixed-reference author profiles

2026-09-09. Discovery packet following M23/M24/M24B. This is a construction of
the shared/private and author-position pieces, not an accuracy tournament.

## Frozen material and cost

Use the existing S1 panel: 24 authors, two views, three queries (T/F,
neuroticism, confirmation seeking), full/E1/E2 material. E1/E2 selections are
identical across queries for each view. Evaluation authors are the twelve M8
transport authors; the other twelve are the reference population. The old
two-author functional matrix G/g remains fixed and is not estimated again.
No human labels, questionnaires or lockboxes are opened.

Reuse 72 compatible M8 P/Q states and collect at most 360 prefix-only states
with the same local observer, template and query token identities. Bound the
new collection at 900 seconds. Retain complete packets for safe resumption.
No suffix scoring, gradients, external APIs or neural training. Native full
and edited outcomes already exist in S1 and are used only for evaluation.

## Functional coordinates

For each query q, P is its ordered question-token matrix; Q is the final token.
The old matrix defines a native-unit affine function, without clipping:

    f_q(P,Q) = b_q + <G_q, P-P_ref,q> + g_q'(Q-Q_ref,q)
    P-P_ref = 1*delta_m + delta_C,     1'delta_C = 0
    f_q = b_q + mean_q + configuration_q + final_q

Full/E1/E2 differences are evaluated by this SAME function. No separate fit
for changes. Configuration is a token-relative pattern, not an emotion label.

Compare functions on matched materials, not by aligning unequal query tokens.
For each edit/view stratum, subtract its REFERENCE-only mean. Standardize by
reference response SD, then for each query regress its configuration response
on the other two queries by OLS (pseudoinverse; no hyperparameter search).

    z_q = shared_q + private_q
    shared_q = z_-q * beta_q;   private_q = z_q - shared_q

The private part is an exact residual relative to these queries, this reference
and a linear projection. It is not exclusive psychological variance. Report
held-author correlations, residual energy and signed covariance accounting;
out-of-reference covariance between shared and private need not be zero.
Do not choose the query set or coefficients from evaluation outcomes.

## Author configuration and compact use

Keep distinct three-coordinate full level and six-coordinate E1/E2 response
profiles, plus their nine-coordinate concatenation. Obtain the fixed center/SD
from the twelve reference AUTHORS (average their two views first). Evaluate the
twelve other authors, preserving individual view scores. Compare paired-view
distances with mismatched authors, with author-permutation uncertainty. These
are cross-text views, not cross-day retest; report any exact text overlap.

For compact functional use, preserve mean and Q exactly; approximate the
zero-token-mean G configuration coefficient by SVD ranks 1/2/4/8/16/full.
Choose the smallest rank with relative RMS error <= .10 on reference full
configuration and on BOTH reference edit-response sets. This is a numerical
action-approximation tolerance, not a psychological success threshold.
Keep full-rank as an exact fallback. Freeze ranks before held-author evaluation.

## Deliverables and decision

- Hash-bound private states, functional parts and matched response ledger.
- Reference-only shared/private coefficients and author reference artifacts.
- Held-author shared/private, native-response and cross-view profile results.
- Reusable full/compact score API with exact accounting and batch invariance.
- Concise mainline ruling: what is shared, what remains query-specific, and
  which author configuration is useful enough to carry forward.

Stop after this packet and its compact construction; do not branch into new
queries, new source labels or another regressor grid. A weak author-view result
routes further work toward material coverage, not erasing the M24 mechanism.
