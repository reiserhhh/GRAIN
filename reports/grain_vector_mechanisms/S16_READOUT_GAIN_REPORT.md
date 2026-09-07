# S16: gain, allocation and conditional curvature each have a role

2026-09-08. COMPLETE: fixed24 source authors,72views,72prefix calls,
0 suffix/backward calls,67.12seconds. No questionnaire join or new fitting.

## Finding

Total body gain fixes most of the catastrophic uniform-weight scale problem,
but comment allocation still matters. One fixed second-order conditional term
substantially improves the remaining covariance approximation.

| Formula | r with exact reader | MSE/exact author variance | Mean error |
|---|---:|---:|---:|
| Exact source |1.000000|0|0|
| Body gain + uniform comment allocation + exact conditional response |.857726|.339773|-.002010|
| Exact mass + conditional covariance |.772988|.741919|+.004178|
| Body gain + uniform allocation + covariance |.651014|.873235|+.003782|
| Exact mass + covariance + second conditional term |.947722|.222261|-.000912|
| Uniform total mass + covariance |.368250|902.601261|+.045095|

All rows use the same frozen reader, query and source folds. No bias correction,
sign flip, or fitted calibration is hidden in the comparison. These are model
readout reconstruction correlations, not human-personality prediction scores.

## Mathematical interpretation

The exact head vector is A=B sum_G r_G D_G. B is total comment-body attention,
r_G the conditional allocation among comments, and D_G the tilted value mean
minus its unweighted comment mean. S16 independently replaces r_G with token
proportions or D_G with its local conditional expansion:

\[
D_G(q)\approx\operatorname{Cov}_G(V,L)
+\tfrac12 E_G[(V-E_G V)(L-E_G L)^2],\quad L=q^TK.
\]

The second term is a mixed third cumulant (one value and two key-direction
factors), despite being second order in query strength. No Gaussian model or
outcome supervision is needed to compute it. It is a finite-step approximation,
not a universally convergent truncation for every query/trait.

The gain-only exact arm shows that relative comment allocation affects the
readout even after body gain is held fixed. The gain-only covariance arm
demonstrates that overall gain rescues scale without restoring full ordering.
The second-order arm then isolates useful conditional curvature that a first
covariance alone misses. All three operations remain in the working formula.

## Attention mass is not synonymous with a semantic contribution

Across12heads, mean body mass ranges.0084-.1576; first-token mass ranges
.0012-.6030; normalized comment entropy ranges.8214-.9507. Substantial mass
goes outside the declared comment bodies, but the first token is not the
dominant destination for every head. This measures allocation, not proof that
one token has a particular psychological meaning.

For example head2 has body mass.0103 yet signed reader-contribution SD.00244;
head10 has body mass.1227 but contribution SD.00170. Mass and downstream
coefficient/value effects are different objects. These descriptive SDs combine
the fixed author-fold readers; do not rank heads as unique psychological factors
or causal personality components from this table.

The distinction is motivated by
[Gu et al., ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f1b04face60081b689ba740d39ea8f37-Abstract-Conference.html)
and [Ran-Milo2026](https://arxiv.org/abs/2603.11487), which examine normalization
and conditional no-operation behavior. Their model/task-specific claims are
not imported as theorems about this frozen Qwen. The decomposition above is
ordinary conditional expectation and finite exponential-family calculus.

## Decision and reproducibility

Freeze the second-order formula and measure it on the remaining189 authors of
the already-authorized source cohort. Reuse these24 completed units, then report
the new189 and all213 separately. Source outcomes are connected only after all
candidate scores are frozen. This is a focused formula-development step, not
an expansion to new traits or a new questionnaire-selected hyperparameter grid.

Code/protocol at0c8ed6b, plan `docs/GRAIN_READOUT_GAIN_S16_PLAN.md`.
Output `results/grain_vector_mechanisms/readout_gain_s16_r1/` (ignored):
manifest, run_status, summary, metrics, head_summary, private author/head records.
Three source arms replay within the recorded numerical tolerance. Three new
tests verify old-arm reconstruction, single-group behavior, local second order,
head contribution sums and constant-value null. No protected labels or cloud API.
