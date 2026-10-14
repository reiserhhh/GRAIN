# HM98 companion: probability geometry and exterior contraction

2026-10-10 JST. Registered after cold scoring, before inspecting new target
results. Does not alter the sealed primary specification or its outputs.
Use only the actual cold T matrices and evaluator identity/uniform controls.
No native application kernels, fitting, bandwidth or target choices.

Euclidean singular values of a row-stochastic T need not be <=1. Do not
interpret the primary unweighted diagnostic as a universal contraction bound.
Fix application reference p_i=1/n and its source image r=T^T p. All entries
are positive in the actual arms. With

    A = diag(sqrt(p)) T diag(1/sqrt(r)),
    C = A-sqrt(p) sqrt(r)^T,

Jensen gives ||T f||^2_p <= ||f||^2_r. A takes the source constant vector
sqrt(r) to sqrt(p), and its adjoint maps sqrt(p) back to sqrt(r). On their
orthogonal complements C has singular values s_i<=1. This is standard
Markov/conditional-expectation Hilbert geometry, not a new inequality.

A closed signed current D has D1=0. Its whitened form is

    D_bar = diag(1/sqrt(p)) D diag(1/sqrt(p)),
    D_source_bar = A^T D_bar A = C^T D_bar C.

On skew matrices, this is the exterior-square action. Its singular values
are s_i*s_j; the largest possible current-norm ratio is s_1*s_2. Check the
explicit136x136 compound matrix independently and all136 primary currents.
Uniform assignment annihilates current; identity preserves it. Probability
mass, point cardinality, full algebraic rank and current amplitude are thus
different properties. Nonzero rank alone is not an author-measurement guarantee.

The weighted reference r is an instrument reference derived from T, NOT a
new learned author population or a deletion of condition choice. Preserve
the actual probability/traffic accounts. A small current gain diagnoses
soft-assignment dilution; it does not imply zero human personality signal.
Native-versus-transported cochain changes remain the primary separate test.
