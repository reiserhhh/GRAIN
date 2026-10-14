# Joint target observation atlas: construction and empirical qualification

HM185, 2026-10-12. Finite SOURCE observation construction, not a new Fourier
theorem, a new personality scale, or an accuracy promotion.

## 1. The question and the actual evidence family

Can the conditional text observations for five target labels be assembled from
five independent global directions, or do reproducible joint target-indexed
components matter? HM182 had stable conditional directions across text views
of the SAME authors. HM185 separates that recurrence from agreement between
different source authors. Historical native TF/N queries have disjoint panels
of 40/24 authors: their intersection is zero. Query-pole vectors cannot supply
missing same-event native responses.

Use the authorized numeric HM184 cache: 386 SOURCE authors, 32 inherited binary
E/N/A/C/O profiles, two disjoint views of 12 comments, 1024-dimensional E5
vectors, and the frozen 1024-by-15 GRAIN linear map. No raw text, questionnaire,
application label, new neural forward, API call, or native-answer fit is used.
The upstream embedding and historical GRAIN construction remain learned.

## 2. A complete, declared reference rather than an independence assumption

For profile g in {-1,+1}^5, let m_v(g) be its author-balanced mean vector in
text view v. Define chi_S(g) = product of g_q over q in S. Under a DECLARED
uniform reference over the 32 observed profiles:

    B_(v,S) = (1/32) sum_g m_v(g) chi_S(g)
    m_v(g) = sum_S B_(v,S) chi_S(g).

The reference is not the empirical population prior. Neither the formula nor
the data asserts independent psychological traits. All 32 terms remain:
one common term, 5 single-target terms, and 26 joint terms of orders 2 to 5.
No smoothing, sign alignment, axis rotation, or selection of positive modes.

The conditional high-minus-low observation for target q, at other-label profile
b, is exactly:

    D_(v,q)(b) = 2 sum_(S contains q) B_(v,S) chi_(S minus q)(b).

Thus all 80 edges per view are read from ONE complete coefficient bank.
A joint S contributes to each of its indexed target families. This is shared
conditional-edge accounting, not evidence of a shared neural circuit or a
uniquely identified psychological cause. A singleton index does not guarantee
geometric or psychological specificity either. Order is label-profile order,
not temporal order, embedding nonlinearity, or neural path length.

## 3. Fixed GRAIN companions and new numerical use

With the historical map A = QR, split each B without deleting either part:

    B_parallel = QQ^T B; B_perp = B - B_parallel
    beta = R^(-1) Q^T B
    B^T x = beta^T (A^T x) + B_perp^T x.

The cold reader receives only the pooled coefficient bank, two semantic query
fields, numeric event vectors, and opaque addresses. It computes B^T mean(x),
then reconstructs every D_q(b)^T mean(x). It does not assign a new author to a
profile, read the author's labels, or output a calibrated trait probability.
The common component and complementary readings stay present. Norms and inner
products are not psychological information fractions.

The 480 historical HM184 frames now have 678 values each: 192 mode readings,
480 conditional-edge readings, and 6 semantic-pole readings. This is a new
complete numerical interface on old diagnostic material, not 480 new people
or an untouched generalization sample.

## 4. The crossed author/text comparison

Within each complete profile, sort original numeric source addresses by a
fixed salted hash and alternate assignments to two author folds. The fixed
folds contain 203/183 authors. Every profile appears in both; some cells have
only one author in one fold. No alternative split is searched.

Construct B_(a,v,S) for author fold a and text view v. The registered primary is:

    J = sum_(|S| >= 2) (1/4) sum_(v,w) <B_(0,v,S), B_(1,w,S)>.

The normalized cross-family coherence is 2J divided by the product of the two
stacked-view norms. It equals an ordinary cosine only when each family's views
are identical. Signed inner products, including negative values, are retained.
The 199 fixed nulls permute WHOLE five-bit profiles within each author fold,
using the same assignment in both views. They preserve joint profile counts
and within-author text dependence. The primary upper tail is
(1 + count(null >= J))/200. Other order/space tails are descriptive and
unadjusted; they are not additional promoted hypotheses.

A useful conceptual model, not a fitted variance decomposition, is
B_(a,v,S) = B_S + U_(a,S) + e_(a,v,S). A same-author-family cross-view product
can retain U as well as B, whereas independent-author products remove the
zero-mean U contribution in expectation under additional sampling assumptions.
This explains why within-family recurrence is a different estimand. It does
not uniquely identify fingerprint, condition, small-cell error, or personality.

## 5. Actual result and theory decision

Full-E5 joint primary J = -0.0000408165189008; cross-family coherence
-0.00585871638; exchange upper tail 0.565. The within-author-fold cross-view
joint inner product is +0.00298211639. All 31 nonconstant modes have positive
within-fold cross-view products, but only 16/31 are positive across authors;
the 26 joint terms split 13 positive / 13 negative.

Full single-target order: cross-author product +0.00004115049, tail 0.255.
Full orders 2/3/4/5 have tails 0.680/0.090/0.930/0.630. The positive order-3
pattern stays visible, but it is not selected as a confirmed common pathway.
Retained/complement joint products are +0.00004158870/-0.00008240522, tails
0.380/0.920. Their sum is the full product, not an additive personality effect.

Decision: the complete atlas is mathematically constructible and cold-readable,
but this finite mean-based source experiment does not support promoting the
joint bank as a transferable personality structure. HM182's same-author
recurrence remains intact; its interpretation is narrowed. The primary result
also does not prove that global independent axes suffice, or that nonlinear
and event-qualified pathways are absent.

## 6. What changes next

Stop this profile-mean atlas as a mechanism-promotion branch. No favorable
fold, rank, weight, query, or coefficient search. Return to event-qualified
target evidence: inventory existing condition/opportunity and source-anchor
interfaces before a new finite plan. Compare a target's common evidence action
with author/context-specific realization on the SAME units. Preserve condition
selection, state, and dynamics rather than labeling their failed group mean as
noise. A new native multi-target intervention needs actual same-event outputs;
different historic author panels cannot be combined to manufacture it.

This is a source-label observation test with sparse cells and historical learned
representations, not a general impossibility theorem or a rejection of GRAIN.
Original 18 protocols / 100 cards / 44 questions remain OPEN.

## 7. Prior art and verification provenance

The complete character basis, coefficient formulas, and discrete derivatives
are classical: O'Donnell, *Analysis of Boolean Functions*, Theorem 1.5,
Proposition 1.8 and Definition 2.16, author PDF (2014 / May 2021 edition).
https://www.cs.cmu.edu/~odonnell/papers/Analysis-of-Boolean-Functions-by-Ryan-ODonnell.pdf
The abstract of Chastaing/Gamboa/Prieur (2012) supplies a reminder that dependent
variable decompositions require their own assumptions; its generalized Sobol
theorem and indices are not implemented here.
https://arxiv.org/abs/1112.1788

Pinned OpenAI/math adc7f124, Signed-tensor-densities 03b-isotypic lines 1-175,
was reread for the full-sector/multiplicity retention idea. Its symmetric-group
density bounds are NOT transferred to this Z2^5 SOURCE construction. This is
design inspiration from a recent manuscript, not a new applicable theorem.

28 focused tests and four Python lint/format checks passed after pre-freeze
repairs. Literal coefficient/profile/edge and cold-coordinate replay maximum
error is 1.11e-15. Verification uses alternate arithmetic by the SAME agent,
not blind replication. Reconstruction identities verify implementation, not
empirical mechanism. The pre-freeze exact-float assertion and float32 pole
normalization guard failures are retained in the parent report; no scientific
output, tolerance, or method was edited after freezing.
