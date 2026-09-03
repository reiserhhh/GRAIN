# V6 Numeric Property Atlas

Status: 2026-07-14. This document is a text-blind evidence map. It describes
only what can be established about GRAIN V6 numeric objects in the current
PANDORA design. It neither names a psychological construct nor claims a
personality score.

## 1. Object hierarchy

After a discovery-fitted coordinate map and opportunity conditioning, the
current working decomposition is

\[
w_{u,r,t}=\mu_{C/O}(r,t)+a_u+B_u z_{r,t}+g_{u,r,t}.
\]

Current PANDORA can form numerical approximations to three object families:

| Family | Current numeric object | What the corpus can test | What it cannot identify |
|:--|:--|:--|:--|
| Static | opportunity-conditioned author configuration across condition cells | slice sensitivity, early/late relational geometry, representation perturbation | a named trait or causal source |
| Hybrid | condition-response configuration after static adjustment | whether broad row relations reproduce | an individual response operator `B_u` |
| Dynamic | run/order-derived path configuration after static and hybrid adjustment | coarse early/late order and selected nonlinear alternatives | stable motion style, recovery, inertia, or causal coupling |
| Residual | configuration not captured by the provisional factor inventory | rank, local dimension, geometry, representation sensitivity | “pure noise” or “hidden personality factor” |

All results below use only numeric author-by-coordinate matrices after the
representation boundary. Reports do not export raw text, terms, user IDs, or
external scale labels.

## 2. Static configuration: supported only at the geometry level

### Early/late residual geometry

For `n=1,640`, the static residual has effective rank `16.49/16.55` and local
intrinsic dimension `14.41/14.21` in early/late halves. Row-linkage tests find
linear CKA `.0396`, distance-rank concordance `.2597`, and top-10 neighbour
Jaccard `.0084`; all survive the 18-test FWER rule at `.018`. The object is
therefore distributed rather than a small catalog of transportable axes.

The multiscale profile is decisive about scale: local `k=5` neighbourhoods do
not reproduce, whereas `k=10..200` do. The normalized excess rises from
`.00123` at `k=10` to `.02628` at `k=200`; the same pattern remains after an
early/late condition-overlap sensitivity mask. This is a broad/mesoscopic
relation field, not an evidence-supported cluster system.

### Technical-replica robustness in real text

Every confirmation author (`n=1,640` before the fixed subsample) retained
within-condition technical replicas. On a fixed `n=384` confirmation subset,
both interleaved and temporally blocked splits pass every three-metric linkage
test after 12-test FWER correction (`p=.012`). Across the four rows:

| metric | range |
|:--|--:|
| linear CKA | `.126-.138` |
| distance-rank concordance | `.350-.404` |
| top-10 neighbour Jaccard | `.050-.057` |

This establishes a restricted but useful fact: the static configuration is not
an artifact of one arbitrary within-condition comment boundary. It does **not**
establish stability across independent days, tasks, languages, or clinical
contexts.

### Representation perturbation

With independently discovery-fitted word/bigram and character 3--5 gram
TF-IDF/SVD maps, full static geometry passes all four correspondence tests at
same and cross temporal halves (`n=384`, 999 permutations, 32-test FWER
`p=.032`). Static residual geometry passes same-half comparison but only partial
cross-half comparison. Thus the broad full static arrangement is not tied to a
single lexical coordinate basis, while the residual’s fine geometry is more
time/representation sensitive.

### Nuisance-strength perturbation

The registered nuisance Ridge alpha is `10`. A fixed logarithmic local grid of
`1, 10, 100` was retained in full rather than optimized. In all six same-half
cross-alpha comparisons (`n=384`, 999 permutations, 24-test FWER), linear CKA
and distance-rank concordance were at least `.9999`; top-10 neighbour overlap
was `.972-1.000`; every adjusted p-value was `.024`. Mean standardized vector
norms were also essentially unchanged.

Therefore the static geometry is not an artifact of choosing `alpha=10` within
this local range. This is a **scorer perturbation** result, not proof that the
nuisance model has removed all topic, situation, opportunity, or state effects.

## 3. Hybrid configuration: high-rank and only partially reproducible

For `n=1,381`, hybrid full and residual configurations have effective rank
about `21-23` in a 24-dimensional space and local dimension about `17-19`.
Distance ordering (`.229-.233`) and very small neighbour excess survive
permutation correction, but linear CKA (`.015-.017`) does not. The result is
substantively unchanged after requiring early/late condition Jaccard >= `.10`.

The correct current classification is **partial coarse relational order**. The
two provisional hybrid factor axes do not transport and remain rejected. This
object is neither erased nor promoted to a response operator.

## 4. Dynamic configuration: no recoverable stable axis or tested manifold

Dynamic residuals (`n=635`) retain only coarse, large-scale ordering:
multiscale neighbourhood linkage appears at `k=100,200`, not at `k<=25`.
Linear CKA and local-neighbour tests do not reproduce. The earlier discrete
factor, path-signature, and conditional-kernel transition probes likewise did
not replicate on held-out PANDORA confirmation data.

To test the specific alternative that dynamics occupy a curved manifold rather
than a linear axis, four pre-specified self-tuning diffusion maps were tested:

\[
W_{ij}=\exp\left(-\frac{\lVert x_i-x_j\rVert^2}{\sigma_i\sigma_j}\right),
\qquad
\Psi_t(i)=(\lambda_1^t\phi_1(i),\ldots,\lambda_m^t\phi_m(i)).
\]

At `(k,m) in {(10,3),(25,3),(50,3),(25,5)}`, all dynamic rows fail 32-test
FWER at `n=384`. This does not prove that no nonlinear phenomenon exists; it
does rule out recovery by this fixed local diffusion family in the present
two-half PANDORA design.

## 5. Cross-block relation: concurrency, not direction

Static and hybrid configurations share a same-half relation geometry (all four
tests pass at 48-test FWER `.048`), but their cross-half comparisons are only
partial. Static-dynamic and hybrid-dynamic comparisons fail. Since all objects
are transformations of the same text-path coordinate field, same-half coupling
can be mechanical. With only two halves, no result identifies leading,
inhibition, recovery, or causal influence.

## 6. Current mathematical verdict

The most defensible PANDORA conclusion is:

\[
\text{reproducible text configuration}
\neq
\text{named low-dimensional factor}
\neq
\text{stable personality dimension}.
\]

The static configuration has passed three relevant robustness checks:

1. early/late relational linkage;
2. within-condition technical replica perturbation;
3. word-to-character representation perturbation.

Hybrid and dynamic objects are not discarded as “nothing,” but their observed
properties are too weak and too nonlocal to name. Dynamic interpretations are
currently refused because the registered two-epoch-by-two-technical-replica
design is available for only 9 PANDORA authors.

## 7. What must wait for new human data

No further factor rotation or nonlinear relabeling of current PANDORA can fill
these gaps:

- Same people, at least three independent occasions, under both free and fixed
  conditions.
- Repeated common conditions plus measured opportunity metadata, so individual
  `B_u` can be identified rather than imputed.
- A frozen multilingual/Japanese representation layer before the private MEPS
  fixed-condition pilot is scored.
- Independent technical replicas per condition/epoch for any change-point,
  inertia, recovery, or interaction-response conclusion.

The next empirical stage should therefore be a small, deliberately designed
same-person protocol rather than additional unbounded PANDORA factor mining.
