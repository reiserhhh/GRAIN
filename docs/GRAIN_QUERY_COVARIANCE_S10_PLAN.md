# S10: query-conditioned key/value pairing as a concrete vector link

2026-09-08. Mathematical preparation while S9 completes; run no second neural
collector concurrently. No new questionnaires, training, API, or lockbox.

## Question and connection to the mainline

S6-S8 localized useful configuration response, but not its elementary operation.
S9 connects that intervention to exposed source outcomes. A complementary local
mechanism is whether a query's attention readout depends on the **joint pairing**
of keys and values, beyond their separate means or separate marginal bags.
This extends the earlier evidence-state theory, not a new claim that every
reasoning operation is one attention head.

For one frozen, position-resolved head, use the effective scaled query q and
paired token rows (k_i,v_i): p_i(q)=exp(q'k_i)/sum_j exp(q'k_j), A(q)=sum_i p_i v_i.
Direct differentiation gives

\[
\frac{\partial A}{\partial q}=\operatorname{Cov}_{p(q)}(V,K),\qquad
A(q_1)-A(q_0)=\int_0^1\operatorname{Cov}_{p(q_t)}(V,K)(q_1-q_0)\,dt.
\]

This covariance is generally rectangular, neither a PSD matrix nor a human
personality covariance. Grouping tokens by comment gives the exact total
cross-covariance split into between-comment means and within-comment pairing.
Two token collections can have identical marginal K and V bags but opposite
readouts if their pairing differs. A marginal distribution student cannot
distinguish that constructed case; this does not prove E5 fails for that reason.

## Primary sources and actual use

- Wainwright & Jordan, *Graphical Models, Exponential Families, and Variational
  Inference* (2008), section3.5.1. Derivatives of the log normalizer give moments
  and covariance; adding value coordinates gives the cross derivative above.
  [Author manuscript, university-hosted](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/WainwrightJordan2008.pdf).
  This is a finite exponential-family identity, not an assumption that authors
  or their psychology have an exponential-family distribution.
- Choromanski et al., *Rethinking Attention with Performers*, ICLR2021,
  [paper](https://arxiv.org/pdf/2009.14794), section2. Kernelized attention uses
  query-key interactions and value-weighted moments. We are not replacing Qwen
  with FAVOR+ or transferring that architecture's accuracy guarantee.
- Schlag, Irie & Schmidhuber, ICML2021,
  [paper](https://proceedings.mlr.press/v139/schlag21a.html). The fast-weight view
  motivates preserving key/value associations rather than two unrelated bags.
  Its equivalence applies to linearized attention, not the full nonlinear decoder.

## Small neural experiment, frozen before S9 outcomes

Use the same eight S1 panel authors, view0, three existing queries. Layer17,
the first attention layer after the block16 intervention site. All heads; no
selection using observed effect, trait labels, or favorable head names.

1. Capture actual rotated keys, values, scaled last-position queries, and the
   native pre-output-projection head vector in one unchanged forward per prompt.
   Reconstruct the vector using fixed KV attention and report numerical error.
2. For each of12 comment groups G, compute P_G=sum_{i in G}p_i and the pairing
   contribution C_G=sum_{i in G}(p_i-P_G/|G|)v_i. Retain other tokens, every key,
   token position, group mass and original query. Subtract sum_G C_G only from
   the selected last-position pre-output-projection vector, then score the suffix.
   This exactly averages the **head vector** over within-comment value
   permutations; it is not an average of nonlinear final scores or a natural edit.
3. A native-vector copy is a self control.24 origins+24copies+24interventions:
   maximum72 forwards,0 backwards,15-minute cap, resumable per-author outputs.
4. On captured KV only, compare small-query linear covariance approximation with
   integrated covariance readout at the native query. Report all12 heads, errors,
   comment attention mass, pairing-vector size, final-score effect and matched
   N/confirmation response correlation. No requirement that every query uses
   identical coupling. This does not yet identify a minimal causal circuit.

Algebra implementation: `grain_research/vector_mechanisms/query_tilt.py`.
Eight synthetic tests check finite differences, integration, matched marginal
counterexample, full small-group averaging, total covariance, group validity,
and the comment-conditional integral with native group mass fixed.
Two tiny random-Qwen CPU tests additionally validate the actual hook/reconstruction
path without source data or loading the full pretrained checkpoint.
Outputs will remain under ignored `results/grain_vector_mechanisms/` with a
tracked aggregate report. Proceed only after S9's live collector has ended.
