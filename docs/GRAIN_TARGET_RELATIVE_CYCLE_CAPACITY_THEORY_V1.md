# GRAIN: target-relative observation capacity and factor compression

2026-10-10 JST. Mathematical companion to HM97. All mathematics below is
standard finite graph/exterior algebra, adapted to a declared measurement
question. Empirical numbers belong in the HM97 report, not in the theorem.

## 1. Measurement sufficiency is relative to a target family

Let n declared evidence states have a complete directed pair support. A
stationary author's pair law P has P_ij >= 0, sum(P)=1 and P1=P^T1. Keep
opportunity/occupancy, symmetric traffic and directional current separately.
This is a local pair-law instrument, not a full multi-event LLM personality
answer or a claim that human writing is globally first-order Markov.

The closed-current tangent space is

    C = {D : D^T=-D, D1=0}.
    dim(C) = (n-1)(n-2)/2.

For observation cochains A_b and target cochains F_q define

    O_A(D)_b = <D,A_b>_F,
    O_F(D)_q = <D,F_q>_F.

Exact target preservation on this finite tangent space is equivalent to

    ker(O_A | C) subset ker(O_F | C),
    equivalently span(F_q_cycle) subset span(A_b_cycle).

Proof: all finite spaces use the same Frobenius inner product; taking
orthogonal complements exchanges kernel inclusion with row-span inclusion.
The positivity restriction does not invalidate the local criterion: every
D in C permits sufficiently small +/- perturbations of an interior P0.
This is not a novel sufficiency theorem. Its measurement consequence is
that "looks stable" or "retains direction" alone is not the criterion.

## 2. Graph cycles and the exterior square

For any pair kernel F, remove exchange-even terms first:

    F_odd = (F-F^T)/2.
    p_i = sum_j F_odd(i,j)/n.
    F_cycle(i,j) = F_odd(i,j) - (p_i-p_j).

This is the complete, unweighted graph projection, with zero row divergence.
It does not claim a physical Langevin force decomposition, thermodynamic
entropy, or the full curl/harmonic decomposition on an arbitrary complex.
The space C is the space of skew forms on the (n-1)-dimensional zero-sum
node space, hence is isomorphic to its exterior square.

For node coordinates h_i in R^d and fixed anchor z,

    A_z(i,j) = ((h_i-z)(h_j-z)^T-(h_j-z)(h_i-z)^T)/2.

Keep all coordinate planes. Changing z adds an exact graph gradient. On a
closed current, that change integrates to zero. In particular, after graph
projection A is the exterior-product bank of mean-centered coordinate
columns. If the affine node rank is r, its area-cycle rank is r(r-1)/2.

For n=3, this rank can reach the entire one-dimensional cycle space with
only two noncollinear coordinates. For n=18 the entire cycle space has136
dimensions; four coordinates can observe at most6. These are capacities,
not percentages of psychological information.

If r=n-1, a geometric coordinate bank can span ALL cycles. Even arbitrary
affinely independent coordinates have this capacity: exact in-support
reconstruction does not confer psychological specificity, natural-material
transport or a calibrated new instrument.

## 3. The actual frozen factor scorer has an exterior action

The inherited instrument maps a SVD row h to four standardized posterior
factor coordinates x through a fixed affine map

    x = hL+b.

L and b are derived directly from the saved feature scales, FA loadings,
noise covariance, signs and norm scales; they are not fitted to HM97 points
or target scores. Affine offsets cancel against the transformed anchor.

    A_factor = A_SVD * compound_2(L).
    compound_2(L)[(i,j),(p,q)] = L_ip L_jq - L_iq L_jp.

The same relation holds after closed-cycle projection. This pinpoints one
information-loss site: compression of event coordinates also compresses
their pair interactions. It is insufficient to keep only the four factors
and afterwards compute a more precise version of the same six areas.
Whether higher-order observations of the compressed points recover other
distinctions is a separate question, not ruled out by this area-only result.

## 4. Observed and invisible target action, with a legal witness

With a fixed Euclidean/Frobenius reference metric, project

    F_q_cycle = A_cycle w_q + R_q,
    <A_b_cycle,R_q> = 0.

The projection coefficients describe measured finite support; they are not
trained personality prediction weights. The energy split is a norm account,
not explained human variance or a causal percentage of reasoning.

Whenever R_q != 0, define

    P0_ij = 1/n^2,
    epsilon = 1/(2 n^2 max|R_q|),
    P+ = P0 + epsilon R_q,
    P- = P0 - epsilon R_q.

Both laws have positive masses, identical uniform occupancy and identical
symmetric traffic. Their chosen area summaries agree, while

    <P+-P-,F_q> = 2 epsilon ||R_q||_F^2 > 0.

This is an explicit target-invisible author-law direction for the bank,
constructed AFTER measuring F. It is not independent validation, a sampled
person, or proof that the target has human personality meaning. It proves
that this bank alone cannot preserve that particular finite target action.

## 5. Practical theoretical consequence

Important counterpoint: the four-factor node coordinates may all be DISTINCT
even though their six area summaries are target-incomplete. On finite distinct
nodes x_i, the standard cardinal polynomial

    phi_i(x) = product_(j != i) ||x-x_j||^2 / ||x_i-x_j||^2

satisfies phi_i(x_k)=delta_ik. Then the pair function

    F_q(x,y) = sum_(i,j) phi_i(x) F_q(i,j) phi_j(y)

reconstructs any measured finite target table. HM97's result-after companion
checks this exactly on its distinct nodes. It is a degree34 node-address
lookup, NOT an independently learned or transported mechanism. It establishes
the logical distinction between inadequate summaries, insufficient point
coordinates and unavailable generalization. No claim about all nonlinear
observations is justified by a six-area null space.

Likewise, full SVD area coverage on18 affinely independent nodes is an
algebraic capacity result. Neither exhaustive high-dimensional preservation
nor finite interpolation supplies the reusable target-specific observation law
required by the project. That law remains the actual next scientific object.

GRAIN should preserve a layered object rather than equate an anonymous
factor catalog with all available measurement information:

    event coordinates
      -> declared factor compression
      -> ordered/coupled observation channels
      -> author opportunity, traffic, endpoint and current accounts
      -> target-specific observation operators and their remainder.

This is a construction criterion, not an invitation to add every raw
dimension indiscriminately. A compact observer can be adequate for one
target family but inadequate for another. Future source-only constructions
must freeze their target family/support, then test new-material transport.
Psychological interpretation and real-author validity remain separate.

## 6. Literature and experiment links

- [Chevyrev/Kormilitzin](https://arxiv.org/html/1603.03788v2), sections1.2/2.1:
  signed area, log-signature Lie brackets, composition; known path mathematics.
- [Nartallo-Kaluarachchi et al.](https://arxiv.org/html/2409.07479v2), Methods2.1:
  graph gradient/cycle ancestry; our pair laws are not their rate-log-ratio force.
- [OpenAI/math pinned isotypic section](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Signed-tensor-densities-and-diagram-budgets-for-coordinate-sweeps-September-26-2026/build/sections/03b-isotypic.tex):
  sector preservation/projection-domain inspiration, not a transferred density bound.

HM97 finite plan: GRAIN_MULTI_CYCLE_CAPACITY_PLAN_HM97.md. Historical V6
signature depth1--3 test is retained; do not call this another successful
personality signature benchmark. Original18/100/44 and the full goal areOPEN.
