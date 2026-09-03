# GRAIN V6 Lineage Token-Slice Compatibility: Frozen Protocol

## Why this second audit exists

The event-disjoint V6 compatibility audit is deliberately stricter than the
historical PRED-1 token-slice estimator. It retained only 32 confirmation
authors at eight 128-token slices per view, so an inconclusive result cannot
confirm or overturn the earlier result. This stage rechecks the original
measurement unit without weakening the event-level audit.

## Estimand

For author `u`, selected top subreddit `a_u`, and construct `c`, compare
token-slice split-half precision in the fixed and mixed rind streams:

\[
\Delta_c = r(s^{a_u,L}_{u,c}, s^{a_u,R}_{u,c})
 - r(s^{\neg a_u,L}_{u,c}, s^{\neg a_u,R}_{u,c}).
\]

This is conditional-expression precision at a fixed token budget. It is not
an event-dynamic operator and does not establish independent occasions.

## Frozen construction

- PANDORA Tier-U local raw comments; only `author`, `body`, `created_utc`, and
  `subreddit` are read. Explicit personality-report comments are removed.
- Existing J1 hash confirmation authors only. No external labels are read.
- Fixed arm: author’s most frequently selected subreddit. Mixed arm: all other
  comments, requiring at least four other subreddits.
- Before any score is calculated, fixed and mixed full-arm calendar spans must
  have ratio at least `0.50`.
- Each arm’s time-ordered text stream supplies sixteen non-overlapping,
  evenly-spread 128-token windows. Alternating windows form left/right views,
  eight windows each. This matches token exposure exactly and samples across
  the whole observed arm rather than using chronological prefixes.
- The three historical target constructs are first-person, directive, and
  novelty. Tension remains descriptive because its historic apparent effect
  was audited away as temporal clustering.

## Decision rule

Author-cluster bootstrap with 999 draws. The historical PRED-1 direction is
upheld for this token-slice implementation only if all three target construct
confidence intervals have lower bounds above zero. A negative upper bound for
any target is an implementation-level contradiction. Everything else is
inconclusive, not an automatic reversal of the old result.

## Nonclaims

No result can identify a personality factor, clinical state, external-scale
relationship, causal topic effect, or a V6 dynamic-order operator.
