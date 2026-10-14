# HM60: Cross-Target Evidence Response Operators

2026-10-09. Retrospective CPU reanalysis, not preregistration or independent
replication. Freeze this design and implementation before computing its statistics.
HM46-HM59 are immutable. No model calls, prompt tuning, API, private data,
training, rank search, publishing, or psychological gold labels.

## Scientific Decision

Does the six-target passive reader have only a generic evidence response, or
selective behavior-to-target interactions that persist across two behavior families?
Do event/narrative regimes scale one operator or change its structure?
This tests an observation mechanism, not human personality prediction accuracy.
M06 already established shared/query-specific actions; do not rediscover that as new.

## Observations and Units

Replay HM56's 24 isolated behaviors and 192 accounts, each read by all six queries:
144 atom + 1152 account readings. Bind each result to raw logits, metadata,
frozen input rows, and source hashes. Analyze both candidate expected score (0..1)
and high-five/low-five log odds; never select the favorable unit.

For query q, behavior c, family f, define

    D_atom[f][q,c] = y(c,f,1,q) - y(c,f,0,q).
    D_account[r,f,o][q,c] = y(r,c,f,1,o,q) - y(r,c,f,0,o,q).

The account companion is the OTHER family and OPPOSITE polarity. In balanced
accounts, changing polarity changes content placement, not the number of positive
events or a true author trait. Order is narrative order, not longitudinal time.
Alias/distinct/other changes retain context rather than treating it as removable noise.

## Fixed Matrix Roster

Per unit: atoms f0/f1/pooled (3); each of four regimes f0/f1 x order0/order1
and pooled (20); alias-balanced, distinct-alias, other-balanced, each four
family/order cells and pooled (15); order1-order0 for each regime f0/f1/pooled (12).
Total 50 matrices per unit, 100 total, 3600 entries. All are reported.

## Mathematical Diagnostics

On K6,6, let each query-content response be an oriented edge. The incidence
matrix B has +1 at query, -1 at content. For complete equal-weight observations:

    G[q,c] = row_mean(D)[q] + col_mean(D)[c] - grand_mean(D)
    C = D - G = P_query D P_content.
    B^T vec(C) = 0; <G,C> = 0; ||D||^2 = ||G||^2 + ||C||^2.

This is the standard graph gradient/cycle orthogonal decomposition, not a new
Hodge theorem. K6,6 has no triangles: use all 225 rectangular cycle contrasts,
not triangular curl. Cycle space dimension is 25. C is an interaction, not noise
and not psychological information. A multiplicative rank-one D can have C != 0;
therefore separately report all six singular values and rank-one energy/tail.
Numerical rank uses machine precision with a fixed 1e-12 norm floor, not a learned
factor-number choice. Ratios/cosines below that arithmetic floor are undefined;
near-perfect cancellations are not interpreted as signal. The floor is fixed
before computation and is never used to select a predictor.

Enumerate all 720 query-content bijections. Report the named diagonal's rank and
ties, not a population p-value: labels are fixed, not randomly assigned humans.
Report diagonal/off-diagonal contrasts, all off-diagonal responses, family cosine,
and cycle cosine. Shared semantic effects are not discarded.

For each account operator, decompose relative to pooled atom D0:

    alpha = <D,D0>/||D0||^2; R = D - alpha D0.

This algebraic projection is not a fitted predictor or evidence that whole author
accounts compose linearly. Zero signal returns undefined ratios, never perfect
similarity. Every matrix/contrast keeps its original measurement unit.

## Tests, Verification, and Interpretation

Synthetic controls: additive-only, rank-one multiplicative, selective identity,
constant/zero, simultaneous relabeling, incomplete/duplicate/nonfinite coverage.
Actual verification uses SciPy softmax/logsumexp; incidence least squares rather
than row-centering; graph rectangle vectors; Gram eigenvalues rather than SVD;
and direct raw-cell contrasts. Preserve all failures.

Accept this finite analysis only after raw binding, Cartesian coverage, hashes,
counts and independent-formula replay pass. Test count is not theoretical closure.
If selective structure crosses families, the next mainline object is a composable
multi-target evidence operator and its context-dependent transport. If generic
structure dominates, do not rescue it with prompt/fidelity optimization. A local
controlled evidence operator is not an author's latent personality vector.

## Sources and Boundaries

- Jiang et al., Statistical ranking and combinatorial Hodge theory, §§4.3/5.1:
  https://arxiv.org/html/0811.1067 . Graph projection only, not its ranking semantics.
- Campbell & Fiske (1959), original MTMM paper:
  https://www2.psych.ubc.ca/~schaller/528Readings/CampbellFiske1959.pdf . Separate
  trait/method variation; six queries from one reader are not independent methods.
- Anandkumar et al. (2014), §3.3:
  https://jmlr.org/papers/v15/anandkumar14b.html . Conditional independence and
  rank assumptions do not follow from querying one LLM six times. No latent-person
  identifiability claim or tensor-factor recovery is made.
- OpenAI/math, commit adc7f1241b42e322a6451854ab7e4b4c146bf78a, Determination
  of a metric and a unitary connection from one boundary patch (2026-10-05),
  introduction and beginning of boundary section: inverse-response strategy only.
  Ellipticity, smooth manifolds, full boundary operators and gauge-uniqueness
  assumptions are not established for the reader. Its theorem is not transferred.
