# Fixed-observable route action

HM43 executed and verified, 2026-10-09 JST. Goal ACTIVE. The execution plan
and scientific inputs remain frozen; this separate document records the result.

## Object and calculation

At each inherited midpoint, H is the four-token text-property patch, P the
ordered early-query patch, and S the complete output of block17 after the
block16 leaf patch. The model, token sequence and ten-digit score are unchanged.
With the inherited digit-score Hessian C, P Jacobian J_P and physical query
displacement e, freeze w64=C^T J_P e before the new forward. Its float32 copy is
the fixed coefficient of L_w=w32^T logits. It is not recomputed from the new
outputs and not differentiated.

Let g_S=D_S L_w, detached, and M_G be one of three disjoint position masks.
The measured H covector and its action are

```text
j_G = (D_H S)^T M_G g_S
A_G(d) = <j_G,d>, G in {body,early_query,remainder}.
sum_G A_G(d) = (J_H d)^T C (J_P e) + numerical discrepancy.
```

The same three covectors serve both original H directions. One scalar
gradient plus three VJPs per node replaces the ten separate scalar families
used by HM42. It does not establish equality of every digit-Jacobian entry.
HM42's coordinate-gate failure remains unchanged.

The exact numerical accounts retained with untouched raw output are

```text
weight_delta = (w32-w64)^T J_H d
arithmetic_delta = sum_G A_G - inherited_link - weight_delta
inherited_link = sum_G A_G - weight_delta - arithmetic_delta
full = inherited_link + inherited_decoder + inherited_old_roundoff.
```

The last identity is bookkeeping. Physical acceptance used the uncorrected
raw sum, not the corrected link. Signed means, wording half-contrasts and all21
cross terms of the seven component accounts are kept. Neither squared actions
nor cross terms are psychological variance or information percentages.

## Actual result

All96 nodes were acquired:48 source, hash barrier,48 application. Logits,
native readout and original scores replay exactly. The maximum raw physical
action discrepancy is8.34527744e-10 under the unchanged1e-8 gate. Four actual
first derivatives per node give96 forwards/384 first/0 second derivatives,
77.7324 model seconds. No fit, generation, API or training occurred.

The remainder contains exactly the final Q endpoint in all96 original layouts.
This is an observed token-address fact, not a semantic label inferred from
the route's size. At the block17 cut, its contribution changes substantially
between the source and application text families:

| Query | Source body-only word omission U/M | Application body-only word omission U/M | Application contrast omission U/M |
|---|---:|---:|---:|
| Negative affect |25.73%/18.88%|40.87%/37.03%|30.97%/79.72%|
| Low mood |15.30%/13.17%|114.42%/166.93%|87.75%/121.78%|
| Trust |24.98%/24.03%|47.96%/49.71%|51.95%/51.32%|

Here omission=sqrt(mean((link-body)^2)/mean(link^2)), computed for the
predeclared wording or contrast family. Ratios above100% are possible under
signed cancellation and are not more-than100% information. U/M name the two
original physical H arms, not two new author samples.

Source trust word action RMS body/remainder is9.9733e-6/3.0142e-6 in U;
application becomes1.40745e-4/1.25296e-4. Application low-mood body/remainder
is1.12094e-5/3.37738e-5 in U. Thus the source's large body route is not a
universal sufficient carrier. Query-region inhibition and endpoint carriage
can change both the level and wording contrast. The new observation motivates
a joint conditional reading object, not an averaged or largest-norm coefficient.

## Mathematical alignment and boundary

OpenAI/math's calculus section works with fixed test gradients and weights in
weak Hessian pairings, on full-support RCD(K,N) spaces with1<N<infinity.
The useful design lesson here is to define and freeze the observable before
testing it. Its geometric regularity and convergence theorems are not assumed
for a Transformer. This packet uses finite-dimensional chain rules and dual
pairing, not an RCD application or a new general mathematical theorem.
[Fixed source](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Weak-Hessian-bounds-along-every-geodesic-in-RCD-spaces-September-24-2026/build/sections/calculus.tex).

The all96 inherited C/J_P weights, including application weights, were prepared
before model calls. This is explicitly disclosed in the manifest. Source-first
new acquisition does not turn them into source-learned cold prediction inputs.
These are controlled local reading actions on reused texts, not natural-author
personality accuracy or unique attention circuits. The arithmetic verifier is
same-agent, not independent neural replication. Original18/100/44, natural
author response B_u, standalone measurement and psychological meaning remain
open. Old failed and frozen results are preserved.

## Next decision

Measure the two-leg3x3 interaction A_GK=(J_H,G d)^T C(J_P,K e), not merely the
H-leg marginal. An endpoint H route may couple primarily to early query
positions or to the query endpoint; the H marginal alone cannot determine
this. Retain signed cross-route cells and separate decoder contribution.
The next finite packet is24 predeclared corner nodes, not a96-node precision
retry or a layer/head/rank search. See GRAIN_BILATERAL_ROUTE_INTERACTION_PLAN_V1.md.
