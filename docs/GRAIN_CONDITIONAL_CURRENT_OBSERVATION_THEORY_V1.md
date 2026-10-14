# Conditional currents and target observation

2026-10-10. HM95 finite experiment, not a personality scale, a new general
theorem, or global completion. Predecessor code and outcomes remain frozen.

## 1. The mathematical object

Let J_u(c,i,j) be an author's empirical adjacent-state mass indexed by condition
and edge position c. Let Q_u(c)=sum_ij J_u(c,i,j), and let R0(c,i,j) be a
source-common probability reference. Keep both terms in

    J_u(c) = Q_u(c) R0(c) + D_u(c).

D_u is a signed conditional-reference difference, not a probability, pure noise,
or automatically a psychological response parameter. Its definition depends on
the declared condition map and source reference. Removing Q would discard choice
and opportunity organization. Opportunity and D need not be orthogonal.

For any square mass M, project its signed edge part onto complete-graph cycles:

    A_ij = (M_ij - M_ji)/2;
    p_i = sum_j A_ij / n_states;
    C_ij = A_ij - (p_i - p_j).

This is ordinary unweighted graph Hodge projection. It does not identify a
physical irreversible drift or require a stationary psychological process.

For m condition-position cells, let Phi_u(c)=C(D_u(c)). Define

    Phi_visible = sum_c Phi_u(c)/sqrt(m);
    Phi_hidden(c) = Phi_u(c) - sum_d Phi_u(d)/m.

Then the Euclidean direct-sum identity is

    sum_c ||Phi_u(c)||^2 = ||Phi_visible||^2 + sum_c ||Phi_hidden(c)||^2.

The same identity holds for centered cross-view covariance. "Hidden" means
invisible to summation over these cells, not latent personality or causal origin.
This condition-axis orthogonality must not be confused with the nonorthogonal
opportunity/response decomposition above.

## 2. Why a pooled-invisible component can still be observed

A fixed condition-indexed target kernel F_k(c) acts as

    s_uk = sum_c <J_u(c), F_k(c)>.

Let J_bar=sum_c J_u(c)/m and F_bar=sum_c F_k(c)/m. Exactly,

    s_uk = m <J_bar,F_bar>
           + sum_c <J_u(c)-J_bar, F_k(c)-F_bar>.

The second term is a matched condition-by-response interaction. Increasing the
precision of the first term cannot reconstruct it. For two authors with
sum_c Delta J(c)=0,

    Delta s_k = sum_c <Delta J(c), F_k(c)-F_bar>.

No new general mathematics is asserted. The scientific issue is whether the
retained interaction is empirically present, author-linked, and acts on a
specified observer. A globally constant kernel cannot read it; an appropriately
varying kernel can. An unchanged marginal does not imply an unchanged joint law.

## 3. Finite information loss witness

On four contexts and three states, use P0(c,i,j)=1/36 and three balanced Walsh
patterns across c, multiplied by one signed three-state cycle of amplitude1/15.
The derivative of each joint law is the product divided by4. On legal small
parameter neighborhoods, direct finite sums give

    I_ab = sum_cij d_a P(c,i,j) d_b P(c,i,j)/P0(c,i,j) = .24 delta_ab.

Forgetting c sums each derivative to zero, giving pooled Fisher matrix0.
This quantifies information about the designed cycle parameters, not human
personality, all author information, or a measurement reliability coefficient.

## 4. What HM95 actually established

The frozen HM93 kernels were obtained from actual earlier neural prefix reads;
HM95 made zero new neural calls. Its72 new numerical Euler ledgers have identical
pooled pair law, occupancy, and symmetric traffic. The conditional readout has
nonzero paired contrasts, maximum absolute.264951; pooled and current-state-only
reset contrasts stay below1.34e-15. All324 target contrasts remain, no winner
selection. Local query, bound query, and uniform value have different units and
are not competing accuracy models.

Natural numerical evidence uses184 application authors/6107 comments from HM94,
four GRAIN lexical slice states and six anonymous conditions. The primary hidden
joint-reference response covariance is3.615685e-5. Against a source-common Markov
null preserving each author's actual opportunity shares, p=.20 and the fixed
reference author delta interval[-4.074764e-5,1.109323e-4] includes0. This packet
does not establish a stable natural conditional-cycle component. It also does
not exclude one beyond this restricted observation space. Three views are
technical splits, not longitudinal retests. No external trait labels were read.

Natural four-state GRAIN laws and the designed three-state behavior kernels do
not share a validated state map. Their coefficients must not be directly joined.
The interface is proved in the designed space; the natural bridge is still open.

## 5. Prior art and next scientific decision

- Ay/Jost/Le/Schwachhofer, *Information geometry and sufficient statistics*,
  Theorem3.10-3.11/Remark3.13: sufficient reduction and Fisher monotonicity.
  <https://arxiv.org/html/1207.6736v3>. This is established theory, not recent novelty.
- OpenAI/math, pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a, three-state
  reconstruction foundations16-48: removing order is justified only by the
  stated target-independent conditional law. Design inspiration only, no tree
  threshold or manuscript correctness guarantee is imported.
- Nartallo-Kaluarachchi et al., graph-flow decomposition:
  <https://arxiv.org/html/2409.07479v2>. Graph Hodge/trajectory analysis is prior art;
  its physical transition-rate field is not this empirical pair-mass operator.
- Existing GRAIN M04/HM82/HM93 already cover conditional responses, information
  reduction and currents. HM95 is an empirical interface test, not their rediscovery.

Next research should establish a common observation unit on the SAME text for
GRAIN context-indexed author structure and frozen target-vector action. First
inventory the old joint-observation results. A new packet requires a missing
mechanism that those results do not answer; do not start another K/rank/head,
bootstrap-count, classifier, or teacher-fidelity search to rescue this result.
