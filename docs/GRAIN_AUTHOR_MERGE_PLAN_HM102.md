# HM102: author evidence composition versus partition-weighted merging

2026-10-10. Finite CPU mechanism construction, not an accuracy experiment.
Parent goal and original18/100/44 remain open. HM101 is sealed and unchanged.

## Question and discriminating operation

NEXT70 retains normalized content and log mass; HM88 separates within-body
attention from author composition. Neither is a new result here. This packet
tests the cross-independent-context merge contract and makes it executable.

For evidence e, target t and attention head h, let

    Z_e = sum_a exp(q_e . k_ea), N_e = sum_a exp(q_e . k_ea) (d . v_ea),
    f_e = N_e/Z_e.

Compare declared author expectation E_P f with partition-weighted merge
sum P N / sum P Z, head by head before summing. They are distinct estimands.
Their difference is Cov_P(Z,f)/E_P Z. Retain all signs/targets/route arms.

Translate every key within one evidence unit by the same vector kappa_e.
The entire local normalized reading function is invariant for every query,
but N and Z are both rescaled by exp(q . kappa_e). Test whether the author
merge descends to the local-observer equivalence classes. This is an observer
equivalence, NOT an encoder symmetry, natural-text edit or psychological truth.
Keeping Z as an explicitly calibrated channel removes that equivalence.

## Fixed construction, not result-selected rescue

Use all864 HM101 captures, two public wording banks, two bridges, six action
families, three previous/current levels, four physical route arms, six targets.
Do not read private/natural text or generate new material. Input query bundle
and numeric layout alone first compile Q kappa = 1 for each KV-head/endpoint,
with fixed least-squares rcond1e-12. Record rank/condition/norm/residual.
If the all-target vector witness residual exceeds1e-8, fail that claim; no
new head/query/rank search. These solves are algebraic witnesses, not fitted
predictors. Only after this compilation export query-independent K/V arrays.

Fixed gauges:0, +/-.5*(previous-1), global+.5. No target/output labels select
them. Base and gauged records have exactly the same within-unit f. Use both
token-sum and unit-mean atom conventions; do not choose a winner. Do not
flatten independently positioned keys as if they formed one native context.
The compiled per-record query position remains registered explicitly.

Synthetic author laws P+/-=(1+/- .5D)/9 have identical current marginals.
In the physically neither arm all local reads ignore previous state, so the
declared expectation contrast must be zero. Any gauged merge contrast there
is a constructed aggregation ambiguity, not new personality information.
This counterexample complements, not replaces, the nonzero both-arm signal.

## Deliverable and checks

Pure NumPy cold scorer reads only compiled query JSON plus numeric K/V/metadata
NPZ, returns f/logZ and separate author expectations/partition merges. Stream
merging stores a positive lift (logZ,f); checks direct flattening, associativity,
reorder, split/rejoin and covariance identity. Finite-target queries are not an
exact finite compression for every future query. Token duplication changes
token mass but not unit-mean mass; repeated events are not independently
counted without an explicit event ledger contract.

Before cold outputs, seal code/plan/config/source and numeric exports. Then
replay HM101 local scores and the expected author contrasts; independently
verify direct token-weight sums and translated K/V versus scalar gauges.
Preserve failures and exclusive output paths. New neural calls/training/API,
parameter grids, psychological naming, Git/lockbox and old queue runs forbidden.

## Prior mathematical support and limits

- Vuckovic/Baratin/Tachet des Combes2020, A Mathematical Theory of Attention,
  section3.1: Boltzmann-Gibbs change of measure. https://arxiv.org/pdf/2007.02876
- Milakov/Gimelshein2018, sections3/3.1: stable online/parallel normalizer and
  associative merge. https://arxiv.org/pdf/1805.02867
- Katharopoulos et al2020, section3.2: numerator/denominator kernel attention;
  exact softmax does not have their finite feature map by assumption.
  https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf
- OpenAI/math pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a,
  Finite-Monoid-Computations.../build/sections/algebra.tex, lines1-160 read:
  composition must respect the declared representation, independently joined
  pieces require compatible meaning. Inspiration only: no finite/F2/star-height
  theorem is asserted for continuous attention.

The new candidate contribution is the GRAIN author-merge construction contract
and its observed finite consequences, not invention of normalization, monoids,
projective equivalence or a general mathematical theorem.
