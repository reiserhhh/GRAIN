# Bilateral route interaction

HM44 executed and verified, 2026-10-09 JST. Goal ACTIVE. Original plan, code,
inputs and numerical results remain frozen. This document records interpretation.

## Signed joint observation

H is the four-token text-property patch, P the ordered early-query patch,
and S the complete block17 output after the block16 leaf patch. With the
original ten digit logits, score Hessian C and physical displacements d/e:

```text
h_G = J_H,G d, p_K = J_P,K e
A_GK = h_G^T C p_K, G/K in {body, early_query, remainder}
H_G = sum_K A_GK, P_K = sum_G A_GK
raw_link = sum_GK A_GK
link = raw_link + cut_roundoff
full = link + inherited_decoder + inherited_old_roundoff.
```

Each digit uses one derivative to S and three masked VJPs to the same leaf;
each VJP supplies both H and P coordinates. C is a symmetric score Hessian,
not a positive probability coupling. Tests check the ordering of a general
bilinear C, signs, causal structure and sum accounts. No second calls or fits.

The H/P margins do not identify the joint matrix: a signed four-cell cycle
can change its cells while leaving both margins fixed, including when the
causal first column is zero. Do not multiply margins as an independence fill.
Nine cells, all36 between-cell crosses and30 other component crosses remain.

## Finite result

24 inherited HBQ=000/111 midpoints:12 source, hash barrier,12 application;
three queries, two wordings, both H arms. P-to-body is exactly zero in every
capture: query inputs begin after body positions. This is a causal graph
control, not a claim that body content lacks psychological information.

Signed wording means for unseen_source, in units of1e-5 original link action:

| Target/role | H body: early/final P | H early: early/final P | H final: early/final P |
| --- | ---: | ---: | ---: |
| Negative affect/source | -1.29669 / -0.99366 | +0.20354 / +0.23010 | -0.20610 / +0.08161 |
| Negative affect/application | +4.96558 / +5.76653 | -0.32804 / -0.47417 | +4.25743 / +4.06332 |
| Low mood/source | -3.63100 / -0.16940 | +0.15869 / +0.02605 | +0.06299 / +0.12011 |
| Low mood/application | +1.07554 / -0.04204 | -0.23117 / -0.10024 | -1.55417 / -1.11298 |
| Trust/source | -1.26504 / +1.90847 | -0.01145 / +0.01988 | -0.57667 / +0.83544 |
| Trust/application | +5.09828 / +7.73404 | +0.10287 / +0.21055 | +3.73764 / +7.45000 |

The omitted P-body column is saved and exactly zero. Negative affect's endpoint
couples to both query regions; low mood's endpoint inhibits positive body
action; trust changes from compensating early/final routes to reinforcing
routes. Application trust endpoint RMS early/final is4.25647e-5/8.82941e-5;
low mood is1.84701e-5/1.54044e-5. Endpoint action is not a universal single edge.
Low-mood application endpoint half-contrast means are+9.91165e-6/+1.05058e-5,
opposite to the wording means. All known_mean-arm rows, contrasts and decoder
accounts are retained. These local operations are not new psychological factors.

24 forwards/960 first/0 second,80.7320 model seconds,0 fits. Original logits,
native and score replay exactly. Raw link discrepancy3.33091e-11; H-margin
discrepancy against HM43 is8.54534e-10, below the unchanged1e-8 physical gate.
Coordinate discrepancy1.75089e-7 is reported, NOT accepted as coordinate
equivalence. HM42's coordinate failure remains failed.

Alternate verifier:25080 numbers,106 output hashes,11464 ancestry/code hashes,
1968 actual lifecycle records; maximum8.54534e-10 includes original H-margin
replay, not pure alternative table arithmetic.48 observations/48 paired rows/
36 groups; same-agent arithmetic, not independent neural replication.14 new
tests, fresh math669 tests/6.59sec. Analysis plus verification48.8316/180sec
before final readback/reserves; final accounting is in parent HM44_FINAL_READBACK.

Manifest discloses inherited application C/d/e prepared before calls. Thus
source-first new acquisition is not source-only fitting or cold prediction.
Two selected corners are not the full eight-corner design or fresh natural texts.

## Alignment and next decision

This turn fully read198-line curvature-change.tex at OpenAI/math commit
adc7f1241b42e322a6451854ab7e4b4c146bf78a. Its RCD result keeps the metric
fixed under bounded reweighting but requires the correct generator domain.
The useful design lesson is preserving the operator/domain when transporting
representations, not treating reweighting as automatic invariance. Its RCD
assumptions are not established here; the curvature theorem is not applied.
[Pinned manuscript](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Weak-Hessian-bounds-along-every-geodesic-in-RCD-spaces-September-24-2026/build/sections/curvature-change.tex).

HM45 will compile source-only affine logit potentials of the full receiver
cut followed by the known nonlinear digit score, with final-only omission,
source-action mean and zero controls. Application upper logits/gradients are
prohibited inputs. Preserve lower internal curvature and link curvature rather
than replacing only C. See GRAIN_SOURCE_CUT_READING_COMPILATION_PLAN_V1.md.

Results: results/grain_math_response_v1/bilateral_route_interaction_r1.
Verification: bilateral_route_interaction_verification_r1/summary.json.
Original18/100/44, natural-author B_u, psychological interpretation and fully
standalone raw-text measurement remain open. No API/private text/Claude/
automation/neural training/commit/push/tag/lockbox/old queue.
