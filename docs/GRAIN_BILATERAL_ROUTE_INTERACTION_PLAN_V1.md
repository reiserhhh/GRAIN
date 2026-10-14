# Bilateral route interaction

HM44, planned only, 2026-10-09 JST. Follow-up to the verified HM43 endpoint
carrier change. Codex executes; goal ACTIVE. Do not modify or resume HM42/HM43.

## Decision-changing question

HM43 measures only the H-leg marginal. Which P-leg route does the application
endpoint carrier couple to? Does a joint route matrix preserve the contrasts
that source body-only summaries lose? H- and P-side marginals do not determine
their joint matrix; do not fill its cells using independence or probability
normalization. All actions are signed and C may be indefinite.

## Fixed scope and mathematics

Use24 inherited HM39 midpoints: three queries, source/application, wordings0/2,
only the two predeclared crossed corners HBQ=000 and111. Path IDs:

```text
source:0,7,8,15,16,23,24,31,32,39,40,47
application:48,55,56,63,64,71,72,79,80,87,88,95
```

These corners are chosen as paired design controls, not from successful scores.
This is not all8-cell transport and not independent new natural text sampling.
Retain both H arms and each actual P displacement. Original eager/MPS frozen
Qwen reader, block16 leaf/block17 complete-state cut, ten digit logits and
exhaustive body/early_query/final-remainder masks remain fixed.

For each digit logit: one gradient to S, detach, then three masked VJPs to the
same leaf, extracting both H and P coordinates from each resulting leaf
gradient. This gives J_H,G and J_P,G together with four first calls per digit,
not seven; each backward produces all leaf positions. Thus40 first calls/node,
24 forwards/960 first/0 second calls,600 cumulative model seconds and180 CPU
analysis/audit seconds including imports, failures and readbacks. No retries,
cap changes, new training or automatic model switch.

For every pair of groups G,K, calculate in float64

```text
h_G = J_H,G d
p_K = J_P,K e
A_GK = h_G^T C p_K
H_margin_G = sum_K A_GK
P_margin_K = sum_G A_GK
link = sum_GK A_GK + measured numerical remainder.
```

Save full raw covectors and the nine physical cells. Do not differentiate C,
the query displacement or downstream scalar weights. J_P,body must vanish in
a causal decoder because P begins after all body positions. This is a causal
graph control, not a claim that topics contain no personality information.
Validate it on synthetic masked-attention and bypass examples before freezing.

Original logits/native/score, total uncorrected physical action and H marginals
against HM43 must pass1e-8 absolute in original units. Report coordinate
discrepancies but do not claim that HM42's full digit-coordinate gate passed.
This packet's declared observation is the bilateral action. Preserve raw failed
nodes before stopping. Old decoder and old roundoff remain separate. For a
group with near-zero total, use absolute actions/crosses rather than unstable
percentages. No coordinate, layer or normalization tuning to create success.

Freeze code, plan, all24 addresses, inherited factors and masks before calls.
Acquire12 source, hash-seal, then12 application. Disclose inherited application
C and diagnostic factors; this is not source-only fitting or an LLM-free text
predictor. No application-dependent rank/route selection. No fit is needed.

## Outputs and acceptance

Save actual attempted/completed lifecycle, observer manifests, patch audit,
cut-state/digit gradients,3-group H/P Jacobians, original donor provenance,
nine scalar cells, both margins, all components and sum errors. Produce48
observations,48 wording mean/contrast rows and36 query/role/arm/measure groups
(word4 points; mean/contrast2 corners). Retain all36 signed cross terms between
the nine route cells, plus link-decoder coupling. These are action moments,
not psychological variance. Tests: chain/bypass, noncommuting C ordering,
indefinite sign, causal body zero, two patches extracted from one VJP, exact
marginal/total identities, source isolation, caps and unstable denominator.

Alternate verifier uses independently ordered digit contractions, all raw
arrays/tables/frozen hashes and source chronology,0 new model calls. Same-agent
checking must not be called independent neural replication.

If endpoint H action is mostly an endpoint-to-early-query interaction, the next
compiler must preserve a joint target-conditioned carrier. If multiple cells
cancel or exchange roles between texts, preserve the matrix as a signed
bilinear measurement object before source-supported compression. If original
physical action fails, save it and identify graph/factor mismatch; do not add a
precision search. Report which finite branch changes the next mainline design.

No API/private text/Claude/automation/neural training/commit/push/tag/lockbox/
old queue. Original18 protocols/100 cards/44 requirements and natural-author
B_u/psychological interpretation/standalone cold text measurement stay open.
