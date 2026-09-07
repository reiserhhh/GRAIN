# S23: connect retained functional components to source N

2026-09-08. After S21/S22, no new source Y has been used for this split.
N's control-orthogonal component still reconstructs transfer pairing effects
at r=.9247. Test where the real source association lies, using no neural calls.

Preserve S20's successful24-author reference g_N exactly, rather than replacing
it with S21's12-author estimate. Use S21's frozen quantitative-content covector
g_c and its12-author/four-recipient-query reference displacement distribution.
As in S22, compute lambda=Cov_ref(g_N^T D,g_c^T D)/Var_ref(g_c^T D),
with within-recipient-query author centering and equal author/query weights.

For all213 cached native N authors score three-view means:
z=mean_v g_N^T d_uv,
z_aligned=mean_v lambda*g_c^T d_uv,
z_orthogonal=z-z_aligned.

Neither component is labeled noise or pure personality. Orthogonality is in the
declared pooled reference function metric, not necessarily on the source N
population alone. The quantitative-content direction may carry meaningful
personality-related choice. Retain both, in their original signed contrast units.

Before joining existing exposed source Y, freeze every score and verify z
replays S20 within1e-12. Then report all three components on primary189 outside
the24-author gradient reference, discovery24, and all213. Pearson/Spearman,
Fisher95%CI, paired author bootstrap differences versus total, component
covariances and the exact identity Cov(Y,z)=Cov(Y,z_aligned)+Cov(Y,z_orthogonal).
No rescaling, label-based sign choice, hyperparameter search or removed authors.
These are exposed development-source associations, not a new untouched benchmark.

This uses S22's empirical L2 orthogonal projection and coordinate-covariant
functional split. Primary background remains
[Park et al.2024](https://arxiv.org/html/2311.03658v2) for metric/representation
distinctions and [Haufe et al.2014](https://pubmed.ncbi.nlm.nih.gov/24239590/) for
covariance-aware interpretation, without importing causal-separability guarantees.

Zero new model calls, two CPU threads,60-second analysis budget. No protected
bridge/MEPS labels, lockbox, LLM training, external transmission, or old queues.
One script: scripts/run_grain_functional_source_split.py.
Output: results/grain_vector_mechanisms/functional_source_split_s23_r1/.
If the orthogonal part carries source association, proceed with a shared-plus-
specific reader family. If the aligned part carries it, study that shared route
instead of declaring the association spurious by its name. Report negative or
mixed outcomes equally. Keep S20's total and T/F positive control unchanged.
