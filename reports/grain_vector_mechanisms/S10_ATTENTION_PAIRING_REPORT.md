# S10: a measurable key/value pairing route to trait-query judgments

2026-09-08. COMPLETE. Same eight S1 authors, view0, three queries, all12 heads
at layer17.72 forwards,0 backwards,104.8seconds. No labels read. The protocol
was committed before inspecting S9 outcomes.

## Finding

Holding each comment's attention mass, all keys, the original query and other
tokens fixed, replacing its value pairing by the exact within-comment permutation
average changes final trait-query scores. This identifies a concrete, nonzero
relational contribution rather than only a difference in vector means.

The matched N/confirmation effects share r=.8070, author-bootstrap95%
CI[.5252,.9710]. This is shared computational response on eight authors, not
a correlation between two human psychological scales.

| Query | Final-score effect RMS | Mean absolute effect | Median comment attention mass | Median pairing-vector norm |
|---|---:|---:|---:|---:|
| T/F | .004018 | .003373 | .02688 | .07401 |
| Neuroticism | .001781 | .001594 | .04249 | .11376 |
| Confirmation query | .001301 | .001006 | .04131 | .10605 |

Native-vector self-copy score error is exactly0. CPU reconstruction of each actual
attention head differs by at most about2e-6. The score effects are therefore not
the numerical reconstruction error. Low attention mass does not make the
value-weighted contribution zero, and mass alone is not a psychological importance
coefficient. This single-layer intervention need not reproduce the larger S6/S9
whole-suffix configuration effects.

## The direct mathematical link

For fixed position-resolved K,V and effective scaled query q,

\[
A(q)=E_{p(q)}[V],\quad p_i(q)\propto e^{q^T k_i},\quad
\partial_q A(q)=\operatorname{Cov}_{p(q)}(V,K).
\]

Numerically integrating that covariance from zero query to the actual query
reconstructs the captured head-vector change:32-point relative MSE median
8.46e-14/1.17e-13/1.16e-13 for TF/N/confirmation; worst head6.46e-13.
The zero-query linear approximation has median error ratios .714/.609/.662.
Thus the exact nonlinear covariance path is computable from captured KV without
running the decoder or generating language again. It still uses learned model
representations, and does not replace the rest of the inference pipeline.

For a comment G, the pairing contribution is
C_G=sum_{i in G}(p_i-P_G/|G|)v_i. Its subtraction exactly averages the **head
vector** over uniform within-comment value permutations with keys/weights fixed.
The downstream decoder is nonlinear: the resulting final score is not the
average final score over those permutations. That distinction is preserved.

The identities use finite exponential-family calculus and kernel-attention
structure, not a new attention theorem. Sources and implementation choices are
in [S10 plan](../../docs/GRAIN_QUERY_COVARIANCE_S10_PLAN.md).

## Meaning for the mainline

S9 shows a suffix configuration contrast relates to observed neuroticism; S10
shows one exact vector-pairing operation inside that class can alter the relevant
queries with shared N/confirmation response. The experiments use different source
panels, so S10 is not yet a quantitative attribution of S9's .001046 covariance.
The useful next object is a query-conditioned joint vector function, rather than
another unconditioned mean-only encoder grid or an assumed literal keyword chain.

## Artifacts and cost

Runner: `scripts/run_grain_attention_pairing.py`; algebra: `query_tilt.py`.
Private output: `results/grain_vector_mechanisms/attention_pairing_s10_r1/`.
Includes frozen `manifest.json`, exact captured `private_kv/*.npz`, author packets,
`private_effects.csv`, `private_head_diagnostics.csv`, aggregate `metrics.csv`,
`summary.json`, and terminal `run_status.json`. Eight algebra tests and two tiny
random-Qwen CPU integration tests pass. No full-model weights were changed.
