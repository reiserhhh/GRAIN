# HM46A: attenuation versus conditional reorganization

2026-10-09. This companion was specified while HM46 was running and before
inspection of its reader outputs. It is not independent preregistration.
No model calls, hyperparameter search or new psychological labels.

Let A[c,q] be self/actual polarity response and B[c,q] another account's
response. Three increasingly permissive mechanism models are:

    B = gamma A
    B = diag(r) A OR B = A diag(t)
    B = diag(r) A diag(t).

They represent common attenuation, content-only/query-only modulation, and
two-sided separable modulation. They are not probabilistic independence
models: entries can be signed and are effects, not probabilities.

The last model necessarily obeys, for every i<k and j<l,

    B_ij B_kl A_il A_kj - B_il B_kj A_ij A_kl = 0.

Proof: insert B_ij=r_i A_ij t_j; both products are the same
r_i r_k t_j t_l A_ij A_kl A_il A_kj. If A has no zero entries, the restriction
is also sufficient: the ratio matrix R=B/A has every two-by-two minor zero,
so rank(R)<=1 and R=r t^T (including zero B). With zeros in A, the quartics
can be vacuous and do not establish identification. Do not divide near-zero
individual cells or force a psychological conclusion from a zero polynomial.

Row/query-only laws have the corresponding quadratic cross-product identities.
Report unnormalized discrepancies and norm-relative discrepancies, with null
when their normalization vanishes. A scalar least-squares projection is a
descriptive finite matrix calculation, not an optimized predictor.

If the restrictions fail substantially across new examples and both surface
forms, context does more than reduce evidence strength: it changes content x
target relationships. That motivates a joint contextual observation operator
in GRAIN rather than a universal eligibility multiplier. Scores being small
does not by itself imply a failure of the algebra or a psychological absence.

Method lineage: Drton/Sturmfels/Sullivant, Lectures on Algebraic Statistics,
author-hosted draft https://math.berkeley.edu/~bernd/owl.pdf, Chapter1
Proposition1.1.2 and Definition1.1.3 (rank-one/minor characterization).
Only the elementary algebraic identity is used. Gaussian factor-analysis,
independent multinomial observations, chi-square thresholds and population
significance results are not imported into this deterministic reader design.
The general tool is established; our question is its application to conditional
text-evidence reading across constructs. No claim of a new general theorem.
