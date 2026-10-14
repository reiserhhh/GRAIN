# Source-defined configuration operators and observation units

2026-10-11, HM144. This is a finite, numerical author-material construction,
not a new kernel theorem, psychological factor discovery, or prediction model.
It connects HM143's joint-configuration question to existing PANDORA vectors.

## What changed in the theory

A scalar description of how well two author cohorts share a configuration can
increase while the nonconstant configuration component decreases. HM144 makes
this distinction executable with a signed decomposition, rather than treating
every improvement in a total statistic as preservation of author information.
The useful object is the joint configuration together with its observation
units and level/departure couplings. No component is deleted as noise.

## Observations and source construction

For author u, partition the same36 distinct publication events into three
disjoint12-event parcels. Each parcel supplies a mean vector h_uv, v=1,2,3.
Chronological thirds and venue-balanced parcels are two observation designs,
not three identical psychological situations or regular time measurements.

Two channels are retained: normalized E5 vectors and15 anonymous GRAIN
coordinates from historical source-learned directions. Coordinates are scaled
using source403 authors only. Validation101 and application120 authors remain
disjoint. These are historical exploratory cohorts, not untouched lockboxes.
No raw text, questionnaire item, personality label, teacher answer, or new
neural call enters this packet. Author IDs only align/group observations.

For fixed Gaussian kernels k_v and cohort law P, define the Lancaster tensor:

    T(P) = E_P [ (phi_1(h_1)-mu_1)
                tensor (phi_2(h_2)-mu_2)
                tensor (phi_3(h_3)-mu_3) ].

Its signed probability measure is

    Delta_L(P) = P_123 - P_12 P_3 - P_13 P_2 - P_23 P_1
                 + 2 P_1 P_2 P_3.

For disjoint source S and application A, the signed contraction is

    I(S,A) = <T(S),T(A)>
           = mean_ij product_v [ H_S K_v(S,A) H_A ]_ij.

The normalized tensor cosine I/(||T(S)|| ||T(A)||) is not Pearson r with a
personality scale. Both empirical laws are centered for this cohort comparison.
Cold per-author witness readings instead use only the source marginal means:

    f_S(x_1,x_2,x_3) = <T(S),
        tensor_v (phi_v(x_v)-mu_v,S)>.

This function is application-batch invariant, but it is not a calibrated
psychological author score. Source directions/units remain upstream assumptions.
Pair kernels and dHSIC are separate diagnostics, not interchangeable targets.
Nonzero Lancaster does not establish information irreducible to all pair laws;
zero Lancaster is not equivalent to mutual independence.

## Exact level/departure accounting

Let m_u = (h_u1+h_u2+h_u3)/3, which is identical under both equal-size partitions.
In each kernel space write

    A_uv = phi_v(m_u),
    D_uv = phi_v(h_uv) - phi_v(m_u).

A_uv is a level-clone representation, not an additional observed event. The
nonlinear phi_v(m_u) need not equal the mean of phi_v(h_uv). Consequently this
decomposition is tied to the declared observation map, not an intrinsic split
of an author's true personality into static and dynamic traits.

After cohort centering, multilinearity gives eight tensors. Let T_0 be the
all-A tensor and T_D the sum of the other seven tensors. Then exactly

    T = T_0 + T_D,
    I = C_00 + C_01 + C_10 + C_11,
    C_ab = <T_a,S, T_b,A>,  a,b in {0,D}.

All64 component contractions and their four grouped sums are retained.
T_D includes mixed level/departure configurations; it is not pure temporal
movement. C_01 and C_10 can be negative. They are signed couplings, not negative
information percentages, error components, or justification for deletion.

For the GRAIN15 chronological source/application120 comparison:

    C_00 = +0.002944780402
    C_01 = -0.000664806134
    C_10 = -0.000593433514
    C_11 = +0.000158685819
    sum  = +0.001845226572.

The nonconstant contraction has conditional bootstrap95% interval
[0.000102266366,0.000209846573]. The author-average contribution is larger,
but the joint law does not reduce to the level clone in these observations.

## Comparing observation designs without changing the ruler

The primary used source median-distance bandwidths separately for each
partition. That supports within-design comparisons, not an isolated claim
about which partition preserves more configuration. The companion fixes the
three chronological source bandwidths for all four source/application design
pairs. No bandwidth choice is optimized on application outcomes.

At those common units, C_00 must be invariant because m_u is invariant.
The observed maximum pooled/replay discrepancy is8.88e-16. For application120:

| Source design | Application design | Departure cosine | C_11 |
| --- | --- | ---: | ---: |
| chronological | chronological | .485082 | .000158686 |
| chronological | venue-balanced | .196783 | .000038549 |
| venue-balanced | chronological | .277865 | .000052887 |
| venue-balanced | venue-balanced | .168177 | .000019168 |

Changing both designs raises total I by34.02% but lowers C_11 by87.92%; the
negative mixed contractions become smaller. This is a property of these
materials and kernels, not evidence that the balanced design is psychologically
better or that chronological departures are causal traits. Validation101 has
the same qualitative pattern. The second design redistributes temporal,
venue and other content structure; these are not separately identified causes.

## Mathematical ancestry and next scientific interface

The signed measure and RKHS contraction are established mathematics:
Sejdinovic/Gretton/Bergsma2013, Sections2-4 and AppendixC:
https://arxiv.org/pdf/1306.2281 . Joint independence is separately addressed by
Pfister/Buehlmann/Schoelkopf/Peters2018:
https://arxiv.org/abs/1603.00285 . Whole-author bootstrap intervals here condition
on frozen directions/kernel units; they do not include direction-learning
uncertainty. Coarse venue/span relinking is a finite sensitivity diagnostic,
not full confound control or a validated conditional-exchangeability test.

OpenAI/math pinned adc7f124's Universal-Tensor-Squares Introduction suggests
tensor/dual contractions and representation-invariant questions. Its claims
about symmetric-group representations do not apply to author psychology.
The contribution here is a project-level observation interface and empirical
decomposition, not invention of tensor products, Lancaster interaction, or
proof of a universal psychological law.

The missing mainline map is a target-qualified diagnostic functional on this
joint evidence account, linked to the event-local operations in HM141 and
earlier vector interventions. T(S) itself is not that target functional. First
inventory the existing event/target/source interfaces. A new packet must show
which target consequence depends on qualified evidence combinations rather
than author-average geometry, not improve kernel alignment or fit LLM scores.
Keep condition choice, personal response, state, and unaccounted components;
all original18/100/44 requirements remain open.
