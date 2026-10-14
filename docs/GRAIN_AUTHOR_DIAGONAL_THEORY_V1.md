# Author identity and target-indexed cross-view moments

HM239, 2026-10-13 JST. A finite SOURCE mechanism diagnostic, not a new
U-statistic theorem, individual predictor or psychological variance model.

## 1. Why this belongs on the measurement mainline

The HM185 atlas preserves all32 inherited five-target profiles, both disjoint
text views and three fixed observation spaces. Its within-author-family
cross-view products are positive for all31 nonconstant characters, whereas
its disjoint-family statistic is positive for16/31 in full space. These
different estimands were already explicit; this work does not discover the
same-family/disjoint-family distinction for the first time.

HM239 resolves the actual author-index terms entering those products. It adds
a distinct-author population-moment interface without constructing a sparse
probability law for every complete text path. The old scorer is unchanged.
Same-author covariance is retained as a measurement object, not removed from
the text vectors or called psychological noise.

## 2. Fixed reference and exact equations

Let g run over the same32 observed target profiles, with n_g SOURCE authors
and two text-view vectors x_iA,x_iB. For each of the32 real characters chi_S,

\[
m_v(g)=\frac1{n_g}\sum_{i:g_i=g}x_{iv},\qquad
B_{vS}=\frac1{32}\sum_g\chi_S(g)m_v(g).
\]

For a fixed observation space define K_ij=<x_iA,x_jB>. The inherited product
matrix is V_ST=<B_AS,B_BT>. Its actual same-author contribution is

\[
D_{ST}=\frac1{32^2}\sum_g\chi_S(g)\chi_T(g)
              \frac{\sum_{i:g_i=g}K_{ii}}{n_g^2},\qquad R=V-D.
\]

Since chi_S(g)^2=1, D_SS is exactly identical for every character under the
fixed partition. It includes the vector origin, conditional mean geometry
and within-author cross-view structure. A D/V ratio is not a psychological
information or removable-noise fraction. R uses the original pair weights;
simply discarding D is not an unbiased population comparison.

If every n_g>=2, define the established distinct-index within-profile moment

\[
u_g=\frac{\sum_{i\ne j:g_i=g_j=g}K_{ij}}{n_g(n_g-1)},\qquad
U_{ST}=V_{ST}+\frac1{32^2}\sum_g\chi_S(g)\chi_T(g)
                    [u_g-\langle m_A(g),m_B(g)\rangle].
\]

Unbiasedness for a population target-mean product additionally requires
independently sampled authors within profiles. It does not require the two
views of the same author to be independent. The sampling assumption is not
proved by this historical dataset. A cell with one author makes the complete
U unavailable; it is not set to zero or deleted.

Writing the empirical within-profile cross-view covariance trace as

\[
c_g=\frac1{n_g}\sum_{i:g_i=g}K_{ii}
                 -\langle m_A(g),m_B(g)\rangle,
\]

gives the exact diagonal relation

\[
V_{SS}-U_{SS}=\frac1{32^2}\sum_g\frac{c_g}{n_g-1}.
\]

This common finite correction explains how many target modes can receive a
similar positive cross-view contribution from reused authors. It is signed,
not a variance-component fraction or a prescription for erasing author
differences. For independently sampled authors its expectation is the
within-profile covariance trace divided by n_g and32^2, summed over g.

## 3. Origin and identity are separate tests

Subtracting the same SOURCE-author mean from all vectors in each view leaves
the nonconstant B, V and U blocks unchanged. It changes D and R. This is a
coordinate-origin diagnostic, not condition-centering or a denoising model.
The common character and its mixed rows need not be invariant.

A deterministic cyclic re-pairing of viewB authors within each profile keeps
all profile means and V unchanged by construction. It changes which K_ii
terms represent the assigned paired identity, and therefore can change U.
This counterfactual audit does not create new human observations. It shows
that the old mean-based atlas is not a test of individual A/B pairing.

The old cross-bank product averages all four view pairings. Its opposite-view
companion is also retained. Neither must equal pooled U in finite data.

## 4. Actual finite result

SOURCE386 authors, two12-comment views in1024-dimensional historical E5
space, fixed203/183 author banks and old1024x15 GRAIN mapping. Pooled profile
counts range3..22. Fold0 admits full U; fold1 has profile7 with one author,
so its U is missing in all three spaces and both coordinate origins.

| Fixed space | Pooled V positive modes | Pooled U positive modes | Joint U sum | Conditional upper rank |
| --- | ---: | ---: | ---: | ---: |
| Full |31/31|13/31|-0.00008276095|0.720|
| Retained15|31/31|15/31|+0.00002437318|0.430|
| Complement|31/31|13/31|-0.00010713413|0.990|

The registered primary is full-space U summed over26 joint characters. Its
null mean is+0.00000936614 using exactly199 inherited-seed whole-profile
reallocations within the fixed author banks. The primary does not show an
excess positive target-mean relationship under this conditional reference.
Negative estimates are retained, not declared negative personality variance.

Full-space single-character U is positive for N/A/O and negative for E/C;
its five-mode sum is+0.00004247970, upper rank0.200. Exploratory N+O and N+A+O
have per-mode upper ranks0.020/0.035 before multiple-comparison adjustment.
All96 space-by-character rows are reported; these are candidate relations,
not confirmation of two new factors or independent follow-up evidence.

The actual full-space D_SS changes from0.00245736147 to0.00005807378 between
the two origins, while the nonconstant V/U blocks differ by at most3.92e-17.
Thus even this actual dataset rejects using the raw diagonal as an intrinsic
psychological fraction. U_SS-V_SS stays-0.00005846578.

Within-profile identity re-pairing preserves V to1.12e-16 but raises the
full-space joint U sum by0.00168574541. The magnitude is an audit account,
not improved prediction or a new sampled population. A different identity
assignment changes which estimand is computed without changing the atlas.

## 5. What changes in GRAIN

Keep at least three distinct contracts:

1. Individual identity/within-author recurrence, with actual event pairing.
2. Target-indexed between-author relation under an explicit reference.
3. Individual target reading and evidence action in a new observation context.

Contract1 cannot be renamed contract2, and neither automatically supplies
contract3. The useful new capability is an executable identity-aware source
moment account alongside the unchanged atlas. It prevents an all-positive
repeatability catalogue from becoming a blanket reusable-personality claim.
It does not invalidate anonymous GRAIN texture, author identification,
conditional response or dynamic/state channels.

Next inventory HM171/173/184/192/209 and the existing condition-linked numeric
event interfaces. Prefer a direct source-supported relation or conditional
response to a complete32-profile mean atlas or positioned hard density.
Before new computation, specify a previously unavailable target-use capability
and competing mechanisms. Do not continue null draws, seed/partition/kernel
search, smoothing, rank, accuracy or LLM-answer fidelity grids. Preserve the
N/O exploratory relation as one follow-up candidate, not the whole mainline.

## 6. Prior art and reproducibility

Distinct-index kernel moments are established; see Gretton et al.(2012),
Lemma6, pp728-729, [A Kernel Two-Sample Test](https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf).
The present statistic is not their complete MMD test. Finite real characters
and covariance identities are also standard mathematics. OpenAI/math pinned
adc7f124, signed-tensor03b-isotypic lines1-120, inspired retaining signed
sectors and occurrence identity. Its symmetric-group/Specht-module domination
theorem and hypotheses have not been transferred to text measurement.

Canonical results:results/grain_math_response_v1/author_diagonal_r1.
Prepare/compute/verify/check executed once each. Compute0.2233seconds CPU-wall;
0 new neural/API/private/application-label/training calls. Historical SOURCE
codes were used; no new questionnaires, query poles or application outcomes
were decompressed. Full signed32x32 matrices and all unavailable cases remain.
2976 diagonal values were re-summed by literal author-pair loops, with192 U
values retained unavailable; maximum old-atlas replay error1.333e-15. Same
agent alternate audit, not blind replication. Nine new/27 initial focused
tests, final57 combined atlas/complement/chain/path tests and three Python
lint/format checks pass. Pre-seal test threshold/shebang issues
were corrected before SOURCE outcomes, not by retuning science.

```bash
python scripts/run_grain_author_diagonal_hm239.py check
python -m pytest -q tests/test_grain_author_diagonal.py
```

Do not rerun exclusive prepare/compute stages over sealed results. Goal remains
ACTIVE/nojob;original18 protocols/100 cards/44 computational questions OPEN.
