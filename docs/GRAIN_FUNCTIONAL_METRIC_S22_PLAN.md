# S22: measure query sharing in the space of actual readout functions

2026-09-08. Zero-neural-call follow-up specified after the S21 four-reference
pilot, before the full S21 transfer matrix is available. That pilot gives
N/confirmation mean-gradient cosine.797 and N/content-control.422.
The goal is to quantify shared versus distinct reading without relying solely
on coordinate angles. It does not replace S21's frozen actual-effect experiment.

## Mathematical basis

Let D_uq be each reference author's pairing displacement under recipient query q,
and g_r the frozen mean covector for reader r. Define f_r(D)=g_r^T D.
Use equal weight per reference author and recipient query. Center D only within
each recipient query to define a covariance metric, not to delete it from scoring:

K_rs=mean_(u,q) [(f_r(D_uq)-mean_u f_r(D_uq))
                (f_s(D_uq)-mean_u f_s(D_uq))].

Equivalently K=G Sigma_D G^T. This is an empirical L2 Gram matrix, positive
semidefinite, and can be computed from the12x4x4 response tensor without a1536
covariance inverse. A null direction is unexcited on this reference support,
not globally inactive. Query blocks share authors: n=12, not48 independent people.

For any invertible coordinate change D'=A D, g'=A^-T g, both the scalar response
and K stay unchanged. Native Euclidean gradient cosine need not.
[Park, Choe & Veitch2024](https://arxiv.org/html/2311.03658v2) motivate distinguishing
geometries and warn against treating arbitrary Euclidean inner products as
semantic. Their causal-inner-product identification has separability and
unembedding assumptions; our empirical displacement covariance is NOT their
causal metric. [Haufe et al.2014](https://pubmed.ncbi.nlm.nih.gov/24239590/) similarly
motivate separating decoding weights from data-dependent activation patterns.
The Gram calculation and invariance below are finite linear algebra, not a
new claim of a universal causal inner product.

## Shared/specific decomposition, retaining both parts

Use the predeclared quantitative-content reader c as a comparison direction,
not as psychological noise. For each r, fit only in the reference response metric:

lambda_r=K_rc/K_cc,
g_shared,r=lambda_r*g_c, g_specific,r=g_r-g_shared,r.

Then the two components sum to g_r exactly, and their reference covariance is0.
No human labels or measured decoder endpoints are used to fit lambda.
If K_cc=0, report the comparison unavailable. Do not add an arbitrary ridge or
select a different anchor. Orthogonality on the reference is not a guarantee
on other authors. Never erase the shared component from GRAIN by calling it noise.

On the fixed S21 transfer12, report shared, specific, and total predictions of
each query's actual pairing effect; component RMS and effect correlation; total
reconstruction identity; and transfer covariance. The control's specific part
is exactly zero and its correlation is undefined. Keep it in the table.

Also report normalized K, its spectrum, and the native Euclidean cosine matrix.
The eigenvectors are query-response combinations, not named personality factors.
Different spectra tell us which native-coordinate similarities matter for
observed text variation.

## Execution

After S21 completes, freeze its summary and all packet/reader hashes.
Load existing reference gradients, pairing vectors and transfer endpoints;
no model load, external API, questionnaire, lockbox, or new outcome selection.
Run synthetic tests for positive semidefiniteness, coordinate-change invariance,
exact component sum and reference covariance orthogonality first.
Two CPU threads; at most two minutes; no new model fitting or hyperparameter grid.

Files: functional_metric.py, run_grain_functional_metric.py and their tests.
Output: results/grain_vector_mechanisms/functional_metric_s22_r1/.
Report all queries, not just N. If distinct response functions survive shared
directions, the next source experiment keeps both channels and tests their
source association separately after freezing; it is not a new denoising deletion.
