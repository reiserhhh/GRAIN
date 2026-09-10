# GRAIN: query response is accumulated and reweighted, not monotonically amplified

2026-09-10. NEXT45 explains the incoming direction located by NEXT44. It retains
the actual model's local derivatives, including attention, normalization and
MLP effects; additive bookkeeping is not an assumption of global model linearity.

## 1. What is now connected

The same author BODY/GRAIN direction is used under N and confirmation queries.
Let d_q be its propagated final-position tangent immediately before MLP25 and
w* the fixed equal source reader from NEXT44. The local query-contrast component is

    T = w*^T(d_N-d_C).

NEXT42 stored each actual signed update. Therefore, with initial tangent d_0:

    d_q = d_0q + sum_{s=0}^{16} u_sq
    T = T_initial + sum_s phi_s
    phi_s = w*^T(u_sN-u_sC).

The initial final-position tangent is exactly zero in all24 units: the frozen
GRAIN perturbation is inserted at BODY positions, not the final query position.
The prefix ends at attention25; MLP25's own output never explains its input.
The complete tensor includes all positions during native transport even though
these saved updates are evaluated at the final position.

## 2. Empirical form: accumulation followed by signed revision

For new8 own mode, cumulative contrast RMS is .031172 after block20, .208040
after block23, and .162922 before MLP25. The block21-23 group has projected share
1.043807; the last group, block24 through attention25, has -.169512. A share
above one is consistent with subsequent counteracting changes.

The largest positive stages are MLP22 (+.354303) and attention22 (+.290940).
MLP24 contributes -.245628, with negative projected shares on every new8 factor.
Its response is therefore not a failed observation to discard: it is part of
how the original query difference is revised. These are ensemble summaries,
not a claim that every individual trajectory overshoots identically.

The mechanism also depends on the author material. Source4 has MLP22 +.425493
and MLP24 -.219469, but attention22 -.035430 rather than new8's +.290940.
Keep the shared MLP pattern and the real cohort difference; do not convert one
sample's attention ranking into a universal circuit.

## 3. Signed covariance explains why a negative update need not be noise

For the stage vector phi and final T, use identical empirical weights and ddof=0:

    E[(sum_s phi_s)^2] = sum_{s,t} E[phi_s phi_t]
    Var(sum_s phi_s) = sum_{s,t} Cov(phi_s,phi_t).

The public matrices retain both triangles; off-diagonal pair contributions are
counted twice when reported as a single pair. New8 own MLP22/MLP24 cross-energy
normalized by E[T^2] is -.424472; attention22/MLP24 is -.330464. Other pairs
offset these, so net cross-energy need not be negative. Source4 has diagonal
energy .018986 and cross-energy -.013913, yielding total .005072.

This is arithmetic of the observed response, not independent layer ablation or
psychological variance attribution. A removed update would change downstream
computations; summing saved increments cannot predict that intervention in general.

## 4. A more explanatory recurrence: carry an existing difference, or introduce one

The next mathematical question is not only where an increment occurs, but why.
For a full-sequence tangent at stage s, let A_sq be the actual local Jacobian
of the residual update, including its identity connection. Then

    d_(s+1),q = A_sq d_sq
    delta_d_(s+1) = Abar_s delta_d_s + delta_A_s dbar_s
    Abar_s = (A_sN+A_sC)/2; delta_A_s = A_sN-A_sC
    dbar_s = (d_sN+d_sC)/2.

The second equality follows by expanding both products. It separates transport
of an existing query difference from an operator-contrast contribution evaluated
on the mean incoming response. It is a symmetric convention, not unique causal
mediation; its total is exact for the supplied Jacobians and responses.

Repeated substitution gives, with later factors on the left:

    delta_d_K = (Abar_(K-1)...Abar_0) delta_d_0
      + sum_{j=0}^{K-1} (Abar_(K-1)...Abar_(j+1)) [delta_A_j dbar_j].

An empty product is the identity. No inverse, commutativity, stationary matrix,
or continuous-depth limit is required. Two tests cover singular/noncommuting
matrices, identical initial directions and common operators. This mathematical
identity is distinct from having estimated every actual full-sequence Jacobian.
The empirical stage sums above do not identify all the A_sq.

For positionwise MLP24 this distinction can be tested economically using only
its local state/tangent and shared downstream w*. For attention, a last-position
vector is not a sufficient Markov state: responses at other tokens also enter.
Do not incorrectly compose last-position-only matrices as the whole model.

## 5. Alignment with prior mathematical approaches

[Elhage et al.2021](https://transformer-circuits.pub/2021/framework/index.html)
motivate analyzing residual contributions and interactions. Their additive
account does not itself prove that a new intervention has a globally linear
response. [Hewitt2026](https://www.cs.columbia.edu/~johnhew/residual-stream-not-linear.html),
an author research note rather than a peer-reviewed result, makes this distinction
explicit. Our use of actual JVPs and the exact derivative of RMS/SwiGLU preserves
the nonlinear context that produced each measured response.

[Olivieri & Perez Rodriguez2026, v2](https://arxiv.org/html/2605.25225v2), a preprint,
organizes activation interventions through response/Green operators and adjoints.
Its local propagation viewpoint closely matches our tangent/readout interface;
it does not establish GRAIN's psychological validity or all-model transfer.
Here we retain a finite discrete recurrence and author-aligned GRAIN probes,
rather than assuming a physical field or an already validated continuum limit.

The useful contribution of this packet is locating actual query-conditioned
amplification/counteraction and supplying its next testable local decomposition,
not inventing the chain rule or the variation-of-constants identity.

See [results](../reports/grain_vector_mechanisms/QUERY_TRANSPORT_FORMATION_REPORT_V1.md)
and [NEXT46 MLP24 mechanism](GRAIN_QUERY_COUNTERACTION_PLAN_V1.md).
