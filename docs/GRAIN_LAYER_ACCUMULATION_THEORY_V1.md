# GRAIN: directional response formation across model depth

2026-09-10. This is a computational interface between fixed GRAIN directions
on the same material and a query-conditioned model. It is not a new scale or
a claim that transformer depth is psychological time.

## 1. Common material, different observations

Let H be the complete hidden sequence after block16. The frozen GRAIN factor f
defines D_f on the same BODY positions, with zeros elsewhere. Its construction
and normalization are those of NEXT41. For this directional derivative D_f is
held fixed; differentiating the direction-construction rule would be a different
experiment. Query, layout, source reference and all intermediate positions remain.

GRAIN thus supplies a scored expression direction; the model suffix supplies its
query-dependent action. Neither view is treated as the other's factor label.
The local result is about a small vector displacement, not automatically a
grammatical text edit, a whole-author score or an interpretation of motivation.

## 2. Signed update identity

Write the actual attention and MLP residual updates in block j as:

    U_j = H_j + A_j(H_j)
    H_(j+1) = U_j + M_j(U_j).

Norms, RoPE, causal attention and grouped key/value heads are inside A_j/M_j.
For an initial tangent D_f:

    dU_j = (I + DA_j[H_j]) dH_j
    dH_(j+1) = (I + DM_j[U_j]) dU_j.

Let E select the final token, and define signed final-position increments
a_j = E DA_j[H_j]dH_j and m_j = E DM_j[U_j]dU_j. Then exactly:

    v_k = E dH_initial + sum_(j=17..k)(a_j + m_j)
    v_full = v_26 + a_27.

Here E dH_initial=0 because D_f has BODY support only. The representation still
has nonzero values there; only its initial tangent is zero. The twenty-one
updates include their actual upstream transport. Removing an isolated block
would change downstream inputs and is not equivalent to subtracting its update.

## 3. Source-fixed reading and an exact error account

Use the previously frozen source cotangent tbar, not the current author's t.
The four fixed readouts are tauhat_k=tbar^T v_k for k20/23/26/full. Current
native local action is tau=t_u^T v_full. Their difference decomposes as:

    tauhat_k - tau
      = (tbar - t_u)^T v_full - tbar^T (v_full - v_k).

The first term is source-readout mismatch; the second is omitted signed
downstream action. A later exit may reduce, amplify or cancel the first error.
Therefore performance need not improve monotonically, and similar final
correlations do not show that the intervening operations were all zero.

For descriptive stage accounting, over the fixed actions i, define

    share_j = sum_i(update_ij * fullhat_i) / sum_i(fullhat_i^2).

The shares sum to one up to numerical error here. They can be negative or exceed
one, and are neither independent causal effects nor percentages of personality
information. Individual signed paths are kept in private numerical artifacts.

## 4. What can actually be removed

An after20 executable runs only blocks17..20 on cached H16, then applies tbar.
It does not compute later blocks, model logits, the current score or gradients
of that score. Forward automatic differentiation is still native model
computation. Constructing H16 from raw text still requires original blocks0..16.
Reduced suffix depth is a genuine dependency reduction, not LLM independence.

Fixed exits avoid selecting a depth from questionnaire labels. Own-query and
pooled-query source means are separate full comparisons, never a best-per-query
mixture. A useful exit can motivate a smaller program; an unsuccessful one can
motivate source-only coordinate translation or necessary contextual computation.

## 5. Relation to prior methods and the user's interpretation proposal

[Tuned Lens, sections2-3](https://arxiv.org/html/2303.08112v6) describes residual
updates and layer-specific affine translation to read latent predictions. Our
experiment instead transports a fixed GRAIN tangent and keeps the source
cotangent unchanged across exits. No affine translator or vocabulary probe is
trained here; poor early reading alone would not establish absent information.

The common-material interface supports the user's proposal to examine where a
GRAIN-scored expression direction participates in a query-related path. It does
not require treating a slice as a psychological item. Context-linked evidence
can later inform a separate, source-grounded interpretation rubric, including
transactional or psychodynamic hypotheses where permitted. That interpretation
must reference observable material and alternatives, not rename this layer
accumulation as a clinical mechanism. No new such coding or clinical assessment
is performed in NEXT42.

Execution: [frozen plan](GRAIN_LAYER_ACCUMULATION_PLAN_V1.md). Results and limits:
[aggregate report](../reports/grain_vector_mechanisms/LAYER_ACCUMULATION_REPORT_V1.md).
