# Typed Evidence Binding for GRAIN

2026-10-08. Finite development packet HM-03, not a complete personality theory.
Canonical outputs: `results/grain_math_response_v1/binding_r1` and
`results/grain_math_response_v1/binding_geometry_r1`.

## Observation Object

An event is not an unordered set of words. Its action and object occupy distinct
argument roles. A target Q observes a learned relation on these roles:

\[
r_Q(a,b)=c_Q+u_Q^Ta+v_Q^Tb+a^TW_Qb.
\]

The augmented typed feature is `(1,a) tensor (1,b)`. Its kernel is
`(1+<a,a_ref>)(1+<b,b_ref>)`, implemented without explicitly allocating the
tensor coordinates. The Gram matrix and its centering reference are learned
only on source events. Applications do not change the reference population.

Action-object types and eligibility are separate. Relevance does not establish
author ownership, actuality, or verified real-world behavior. The contribution
combines observed relevance with independent polarity/ownership/modality slots:

\[
s_Q(e)=\hat p_e g(r_Q(a_e,b_e))g(\hat o_e)g(\hat a_e),\quad
g(t)=\operatorname{clip}((1+t)/2,0,1).
\]

Missing eligible evidence remains unobserved. Event aliases do not create new
evidence. Context/choice composition is retained in its own account, not erased
by this within-event computation. The present compound controls share p/o/a;
heterogeneous event-local eligibility is the next HM-04 mechanism.

HM-04 and HM-05 have now been executed without changing the HM-03 source models.
The extension, actual results, and purpose-dependent qualification boundary are
documented in `docs/GRAIN_EVENT_LOCAL_OBSERVATION_V1.md`; this paragraph's shared-
role compound controls describe the historical HM-03 scope, not the new packet.

## Coordinate-Free Action and Its Limits

With row coordinates `a'=a T_A`, `b'=b T_B`, transported metrics are
`G_A'=T_A^{-1} T_A^{-T}` and `G_B'=T_B^{-1} T_B^{-T}`. Then:

\[
a'G_A'a_{ref}'^T=aa_{ref}^T,
\qquad b'G_B'b_{ref}'^T=bb_{ref}^T.
\]

Thus the cross-Gram, source centering, and frozen dual score are unchanged.
This identity is a same-reader change-of-basis result. It does not establish
that a new neural reader or domain is an invertible coordinate change.

Pointwise multiplication `a*b` identifies coordinates across role spaces.
That identification requires justification. An outer product keeps role types
separate and contracts them through a target-specific functional.

## Actual Finite Evidence

- 1,536 public single-event texts, 32 same-word argument-swap controls.
- Two frozen readers, 1,609 actual inputs each; 1,568 frozen Stanza parses.
- Typed tensor contribution RMSE improves over slot-additive in all eight
  cold reader/target/domain point comparisons, with six fixed-program crossed
  bootstrap intervals above zero. Whole-sentence coordination retains an
  advantage on unseen combinations of familiar slots.
- Automatic argument routing distinguishes 32/32 relation cases per reader;
  slot-additive also succeeds here. The tensor advantage is primarily score
  calibration, not a uniquely necessary classifier in this material bank.
- Same-reader basis/metric transport: 32 transformations, 12,288 actual
  readouts, maximum change 8.88e-14. Wrong reference/metric transport changes
  scores substantially.
- Source-fitted vectors and programs contain no application labels at scoring.
  No API, new neural training, private texts, publication, or lockbox action.

## Predecessors and Contribution Boundary

[Coecke, Sadrzadeh and Clark (2010)](https://arxiv.org/html/1003.4394v1) develops
typed compositional distributional semantics.
[Grefenstette and Sadrzadeh (2011)](https://aclanthology.org/D11-1129.pdf) tests
relational word representations and composition. The present kernel is a
limited implementation of role-sensitive interaction, not a full categorical
grammar and not an original tensor-product theorem.

OpenAI/math's pinned priority-subspace transport construction motivates
protecting the target action during representation changes. Its theorem is
not implemented or established by these experiments. The contribution here
is a tested observation-interface step for GRAIN: role-bound semantic action,
qualification, fixed-reference scoring, and explicit coordinate transport.
Psychological meaning must be connected to authors and independent criteria,
not inferred from grammatical accuracy.

## Code and Reproduction

`grain_research/construction/math_binding.py` implements source-only fitting
and cold scoring. `math_binding_geometry.py` implements metric transport.

```sh
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
"$PY" scripts/run_grain_math_binding.py --device mps
"$PY" scripts/run_grain_math_binding_geometry.py
"$PY" -m pytest -q tests/test_grain_math_binding.py tests/test_grain_math_binding_geometry.py
```

Changed protocols require new output directories. Original code snapshots,
source models, failures, and result tables remain available. The original
18-protocol/100-card/44-question scope is not closed by HM-03.
