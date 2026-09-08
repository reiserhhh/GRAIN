# M14: locate the conditional-baseline association by an actual pairing intervention

2026-09-08, frozen before collection. M12/M13 b readers mainly correct offset;
source N association remains in the actual conditional anchor b(E). Do not add a
prediction hyperparameter sweep. Ask whether the previously established KV-pairing
path also operates when every author's P/Q boundary is the same reference.

## Fixed experiment

Same M11 48 source authors,24/query,3views each. No new label/material access.
At the cached block16 boundary replace P/Q with the original two-reference mean.
Compute layer17 QKV and all-head attention from exactly that state. For each
original comment G, retain its attention mass and value mean, remove only:

```
d_h = sum_G sum_(t in G) [p_ht - P_hG/|G|] v_ht
head_no_pair = head_anchor - concat_h(d_h)
```

This is the expected head output under independent within-comment uniform value
permutations with keys/query/attention fixed. It does not remove comments, change
wording, or claim that an actual shuffled human sentence has that state.

Patch only the final-position layer17 pre-W_O head, then run the frozen suffix.
One patched suffix/view =144 calls, plus first-view native-anchor replay for each
query =2calls. Maximum146 suffix,600seconds;0prefix/gradient/training. Capture
actual incoming head and require its match with mathematical reconstruction.

## Exact ledger and interpretation

`s = b_no_pair + (b-b_no_pair) + J_ref*d_x + c`.
Freeze author-mean intervention scores before the existing sourceY join.
Report b_pair and b_no_pair source associations/covariances, paired effect
magnitude, retained covariance, and old S9 C relation. Same48 diagnostic panel,
not independent confirmations or percentages of psychological variance.

If b_pair retains the source association, next transport this specific
conditional vector functional instead of the whole b value. If it does not,
the missing baseline is routed elsewhere in E/remaining network; preserve the
earlier native-query pairing success. Conditional absence does not refute it.
This packet has no data-fitted score, coefficient search or new personality axes.
