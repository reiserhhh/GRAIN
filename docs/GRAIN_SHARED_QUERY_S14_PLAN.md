# S14: shared reference query versus author-conditioned query

2026-09-08. Mainline discovery after S12/S13; no new target reader fitting.

## Question and decision

S12 recovered part of N's useful configuration response from a joint KV vector.
Does that readout require a different query for every author, or can one
author-out reference query read the useful relationships already in their keys
and values? This tests a concrete shared/private split, not whether different
people have identical psychological traits.

Use all213 source N authors/3views and the existing five author folds.
Construct each head's pre-RoPE query from each saved block16 query state via
the frozen layer17 input normalization and Q projection. For fold f, average
three views per training author, then average the training authors, never the
held-out fold. The resulting common raw query has12x128 coordinates.
Apply each held-out prompt's own RoPE endpoint and attention scaling.

For alpha=0,.5,1, set raw_q(alpha)=q_reference+alpha(q_author-q_reference).
Keep each held-out author's keys, values, token positions and comment groups fixed.
Read all three KV-pairing vectors using their **existing S12 KV/PCA C reader**.
Do not retrain it, tune alpha, select signs, or choose users from questionnaire Y.
Native alpha1 replays S12; alpha0 tests common geometry with all author-specific
information in KV retained. Alpha.5 checks a fixed intermediate dose.

## Exact accounting

Let Vtilde_i=v_i-mean_G(v) for body tokens i in comment G, and0 otherwise.
Then the complete within-comment pairing vector at a head is

\[
C_{\rm head}(q)=E_{p_q}[\widetilde V],\qquad
dC_{\rm head}/dq=\operatorname{Cov}_{p_q}(\widetilde V,K).
\]

This derivative includes changing comment masses P_G; it is not the version
holding P_G constant while integrating only conditional within-group weights.
Centering is fixed by the captured V and declared groups, not fitted to Y.
The source reader contrast obeys

\[
\widehat C(q_u)=\widehat C(q_{-f})+
 [\widehat C(q_u)-\widehat C(q_{-f})].
\]

Linearity gives the same sum for Cov(Y,.), but the terms need not be orthogonal.
The second term is author-conditioned **query contribution given KV**, not all
author-specific information. The reference query still belongs to the same
pretrained model and N question; it is not a human population mean personality.

S10's exponential-family derivative (Wainwright & Jordan,2008) supplies the
mechanism; the S12 finite linear-reader result supplies the target and fixed map.
[Exponential Family Attention, Wibisono & Wang2025](https://arxiv.org/abs/2501.16790)
offers a context-dependent factor-model precedent, not an identifiability
guarantee for these human traits. No new theorem is asserted for the algebra.

## Analysis and resource contract

Collector:639 prefix calls,0 full decoder scores,0 backwards;35-minute cap.
Build and freeze reference queries before collection. One neural collector only.
Primary: teacher-C r/MSE of alpha0/.5/1 and paired differences vs alpha1.
Native endpoint vector max-error1e-5 and reader max-error1e-6 against S12.
Report reference-query dispersion and actual angular differences, never assume
prototype replacement is a large perturbation. Save per-author three-dose scores.
Freeze scores/hashes before joining previously exposed source Y.

Report all doses' source N associations, paired bootstrap intervals and
Cov(Y,shared)+Cov(Y,query_increment). Preserve native S12 and T/F controls.
If alpha0 preserves recovery, prioritize a common query operator on author KV.
If personalized query adds useful response, retain it and localize its geometry.
Either finding is a mainline result; neither alone proves all author identity is
query-based. Do not run new labels, API, LoRA, old queues or external lockboxes.

Outputs: results/grain_vector_mechanisms/shared_query_s14_r1/
Report: reports/grain_vector_mechanisms/S14_SHARED_QUERY_REPORT.md.
