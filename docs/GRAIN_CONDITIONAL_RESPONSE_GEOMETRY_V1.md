# Conditional Response Geometry V1

HM-09A is a finite observer-mechanism experiment, not a personality scale or
a new human-validity result. The HM-03 tensor models and their thresholds remain
fixed. Geometry is calculated separately inside each frozen E5/BGE space.

## Exact Functions And Reference Law

For target Q, the source-centered kernel has the exact expansion

\[
k_Q(a,o)=c_Q+c_{A,Q}^{\mathsf T}a+c_{O,Q}^{\mathsf T}o+a^{\mathsf T}T_Qo,
\quad J_Q(a)=c_{O,Q}+T_Q^{\mathsf T}a.
\]

Consequently, for a fixed observed action,
\(\Delta k_Q=J_Q(a)^{\mathsf T}\Delta o\) is exact, not a small-step Taylor
approximation. This derivative is an **observer response**, not an author's
psychological sensitivity. Author/experiencer, mode and target qualification
are retained downstream; a relevance response alone is not an author score.

The common empirical reference contains 16 distinct source-role actions and
16 source-role objects, uniformly enumerated, with 120 unordered object
displacements. It is not the distribution of natural language. No application
labels enter the reference, source unit conversions, SVD ranks or geometry.
Ranks use eps * max(shape) * largest singular value, not a rank search.

## Three Different Objects

1. Euclidean gradient-subspace principal angles disregard sign and magnitude.
2. Reference-action geometry uses the response fields J_Q(a).Delta o. Its metric
   is the empirical object-difference covariance; it can be singular.
3. Cold action transport applies one source-only, zero-intercept beta to actual
   new displacements. It is judged against identity and zero controls, in fixed
   target units. A close subspace does not imply a useful scalar conversion.

For two targets, the mean and contrast reconstruct both responses exactly:
\(r_1=s-d, r_2=s+d\), where s=(r_1+r_2)/2 and d=(r_2-r_1)/2. This is a
reversible coordinate decomposition, **not** a discovered common trait or an
orthogonal division of psychological information. Blindly dropping d is a
negative control. The actual cold gate test averages k scores, not these r's.

## Two Mechanism Findings

Both targets have reference-field rank 8 in an ambient rank-15 reference.
The Grassmann dimension identity forces an intersection of at least one
dimension: 8+8-15=1. Both observed top principal cosines equal 1, and the
alternate formula confirms intersection rank 1. This overlap cannot itself
be offered as evidence of shared psychological meaning.

Let mu be the source object mean and P the projector onto centered source
objects. The family

\[
k_{Q,\lambda}(a,o)=k_Q(a,o)+\lambda J_Q(a)^{\mathsf T}(I-P)(o-\mu)
\]

agrees with k_Q on **every source object**, for every action. New objects can
separate it. The result-after companion compares lambda=-1 (the anchored
projected alternative) with lambda=0 (the unchanged original); neither is
selected as better. Actual source score error is <=6.07e-14. Cold E5 verification
scores differ by up to 0.223184 and 14 legal-view gates change. Thus source-fit
equivalence does not establish transport equivalence. The complement is
unidentified by this reference, not established to be noise or personality.

## Actual Scope And Results

- New material: 768 controlled public English expressions, four new object
  families, 16 source-known actions, actual/desire modes, self/other subjects.
- Actual Stanza: 768 calls; complete documents 765. Three verify/want constructions
  remain unresolved and are excluded with explicit IDs, not repaired after test.
- Offline frozen encoders: 784 new inputs each, 1,568 total, MPS. No neural
  training, API, private corpus or new source-target fits.
- Legal-view target-gate errors: E5 80/1530, BGE 30/1530. Blind shared gate:
  128/1530 each. These are supplied relation-design codes, not personality accuracy.
- Own-qualified errors: E5 40/1530 vs blind-shared 64/1530; BGE 15/1530 vs 64/1530.
  No qualified other-actor event in either reader. Coverage is reported separately.
- E5's source conversion improves some coordination-context response comparisons
  but fails other domains; BGE conversion usually approximates zero. No universal
  unit conversion is promoted. HM-08 cache analysis is explicitly result-after.

## Reproduction

From the implementation root, using the existing Project Persona environment:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 '/Volumes/mobile3/projects/project persona/.venv/bin/python' scripts/run_grain_math_response_geometry.py --stage all --out results/grain_math_response_v1/response_geometry_replication --device mps
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 '/Volumes/mobile3/projects/project persona/.venv/bin/python' scripts/verify_grain_math_response_geometry.py --out results/grain_math_response_v1/response_geometry_replication --dest results/grain_math_response_v1/response_geometry_replication_verification
```

The original completed result refuses rerun/overwrite. This requires the fixed
local source models, frozen HM-08 inputs and encoder/parser caches. It is not a
standalone download-and-run package. A separate result-after witness lives in
`scripts/diagnose_grain_math_response_support.py`. Existing primary artifacts are
not changed by either companion.

## Mathematical Sources And Remaining Dependency

[Bjorck and Golub (1973)](https://www.ams.org/mcom/1973-27-123/S0025-5718-1973-0348991-3/)
provides the standard principal-angle/SVD method; the two PDF fetches failed
with 403 in this packet, so no full-paper reading is claimed.
The OpenAI manuscript [Random-subspace tests and trace smoothing for coordinate
sweeps](https://github.com/openai/math/tree/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Random-subspace-tests-and-trace-smoothing-for-coordinate-sweeps-September-26-2026)
was read from the pinned local TeX, introduction 1-160 and proof 1-325. Its tagged
projection-overlap strategy motivates comparing overlap with action; its Thorp
shuffle, representation and complex random-plane bounds are **not** imported
as theorems about language. The Gaussian orientation control here is real,
norm-matched and diagnostic, not the manuscript's complex random-plane test.

HM-09B multi-trigger state exposure is still pending. The original 18 protocols,
100 cards, 44 computational questions, source uncertainty, psychological linkage
and natural author scope retain their actual open statuses.
