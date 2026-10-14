# GRAIN: contextual selection is a hierarchical observation measure

2026-10-10. HM88 executed and independently re-summed by the same agent from
frozen HM87 caches. No new neural calls/fits. Canonical:
`results/grain_math_response_v1/hierarchical_query_measure_r1`.

## 1. What changed in the theoretical object

The observation is not just a trait direction dotted with a mean vector.
For one encoded body e and target t, it is a probability measure over evidence
tokens, with both an event allocation and a conditional token law:

    phi_t(e) = sum_G pi_t(G|e) m_t(G,e)
    m_t(G,e) = sum_[i in G] p_t(i|G,e) <d_t,V_i(e)>
    Theta_t(P_u) = integral phi_t(e) dP_u(e).

Head sums are understood, with the source direction d_t fixed in the same
head/value coordinates. Pi is computational attention within the body, not
the author's choice distribution P_u and not psychological importance.
Keeping this distinction preserves GRAIN's group-referenced measurement spine:
the common observer and the author's evidence composition are different objects.

## 2. Exact hierarchical change, rather than mean-zero denoising

Let p0 be cold token weights. A contextual query induces a logit potential b:

    p1(i) = p0(i) exp(b_i) / Z.
    u_G = log E_[p0(i|G)] exp(b_i)
    v_i = b_i-u_G.

Then E_[p0(i|G)] exp(v_i)=1 and

    pi1(G) = pi0(G) exp(u_G)/Z
    p1(i|G) = p0(i|G) exp(v_i).

event_only=pi1*p0(.|G), within_only=pi0*p1(.|G). Each imposes its
named constraint exactly. A potential centered by E[b|G] does not: its
conditional log partition varies between events. In these actual caches,
naive centering changed event probability by up to .0190216 (absolute mass).
This is an empirical consequence of an established Jensen/normalizer effect,
not a newly discovered inequality.

KL(p1||p0)=KL(pi1||pi0)+sum_G pi1(G) KL(p1(.|G)||p0(.|G)).
The measured mean parts .00212065 and .00324916 nats quantify token-measure
change, not trait information, between-person variance or psychological entropy.

Readout changes need not add even though the laws factor. At a fixed body/head,

    r1-r_event-r_within+r0 = sum_G (pi1-pi0)(m1-m0).

This is the elementary product identity. Removing between-event allocation
can alter the effect of within-event evidence; neither component is automatically
noise. The signed interaction is retained in finite_components.csv.

## 3. Shared fields are not a universal persona axis

For the six fixed target queries, form b_t=l1_t-l0_t, token-uniform gauge-center,
and split b_t=bbar+(b_t-bbar). The equal-weight mean is invariant under target
permutations. It is the trivial sector of this particular target-index action;
deviations have zero target mean. No rank fitting, semantic grouping or winner
selection was used. Changing the target battery could change this shared field.
It is a contextual computational decomposition, not a common psychological trait.

shared_only and specific_only act through the same exponential normalization.
Their output interaction is not zero and cannot be ignored just because their
potentials add. Both are obtained from already contextualized native queries;
they do not yet provide the independent body-to-field construction.

## 4. Finite mechanism outcomes

Matched query selectivity means its own-target behaviour action minus the other
five queries' mean action. All values use calibrated vector units, not r/AUC.
Six cyclic companion designs, all16 x/y/g/h corners, all12heads, forward order:

| Target | Cold | Shared Only | Specific Only | Native | Event Only | Within Only |
|---|---:|---:|---:|---:|---:|---:|
| E | .002285 | .000746 | .009503 | .002504 | .006546 | .005105 |
| N | .016268 | .021166 | .029201 | .034322 | .027584 | .023461 |
| A | -.015379 | -.012910 | .043154 | .052816 | .037059 | -.001776 |
| C | .002479 | .005529 | .002254 | .006675 | -.004189 | .012123 |
| O | -.023278 | -.023669 | .006189 | .009854 | .009754 | -.026134 |
| Anthropomorphism | .069723 | .050898 | .187767 | .139971 | .095181 | .120951 |

Symmetric components of native-minus-cold selectivity:

| Target | Shared | Specific | Event Allocation | Within Event |
|---|---:|---:|---:|---:|
| E | -.004269 | .004488 | .000830 | -.000611 |
| N | .005010 | .013044 | .011088 | .006966 |
| A | .006066 | .062130 | .053515 | .014681 |
| C | .003736 | .000460 | -.006058 | .010254 |
| O | .001637 | .031495 | .034510 | -.001378 |
| Anthropomorphism | -.033311 | .103558 | .022239 | .048009 |

These are exact accounts under the chosen intervention convention, not unique
neural causal routes or final-answer mediation. The E net change is tiny despite
opposing components. A/O selection needs target-specific field in this battery;
C uses mostly shared adaptation but within-event reading offsets an adverse
event-allocation effect. Keep different targets' laws instead of optimizing one
common scalar. No text-level psychological strength ordering is inferred.

## 5. Author comparison and target selectivity can move oppositely

The synthetic evidence laws P+/-=(1+/-0.5yg)/16 keep each single coordinate's
marginal identical and change the association of companion behaviour and actor.
All six matched contrast directions remain positive for all six arms, including
the h=-g equal-total-name-count subset. This is compound actor configuration,
not pure ownership, real narrator identity or human personality validity.

Anthropomorphism native selectivity rises .069723 -> .139971 while the synthetic
author-law contrast falls .031590 -> .020881. N's contrast falls under within_only
(.084975 -> .081398) but rises under event_only (.094268). Thus choose neither
an observer by selectivity alone nor a personality method by teacher fidelity.
Define the author evidence contrast before deciding what an observation must
preserve. This is the concrete GRAIN measurement consequence of HM88.

## 6. Evidence, ancestry and scope

- 96 old forward body carriers,576 observed contextual queries; retrospective
  numerical interventions,0 new forwards/generation/API/private/training/fits.
- Primary direct/group sums max6.273e-15; separate verifier reconstructed all
  six arms from normalized probabilities, max6.217e-15. All scalar actions and
  components max6.412e-15; HM87 cold/native replay max2.442e-15.
- 28 focused tests/Ruff/format passed. Same agent, not independent review.
- HM87 prefix gate FAIL remains; no rerun or threshold change. Neither the
  original18/100/44 nor full psychological framework is promoted by these checks.
- Historical full token-slice factor bank, natural event segmentation, new
  human traits/states/dynamics, and body-only construction of adaptation fields
  were not added in this packet. Body vectors still require the neural encoder.

Prior S12/NEXT50/NEXT73 and HM32/75/86/87 are reused, not relabeled as novelty.
Exponential-family normalization and KL conditional chain rule are standard.
[Wainwright/Jordan](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf)
supplies the statistical background; original Princeton URL403, further
section-targeted fetch timed out, so this turn is not a full paper reread.
[Feng/Steinhardt](https://arxiv.org/html/2310.17191v2) supplies binding precedent.
[OpenAI/math isotypic tools72-140 and206-244](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Signed-tensor-densities-and-diagram-budgets-for-coordinate-sweeps-September-26-2026/build/sections/03b-isotypic.tex)
inspires sector/operation distinction, not a transferred signed-density theorem.

Next mechanism: construct pi_t and p_t(.|G) from the once-encoded body and fixed
source observation family, rather than regress the complete native query.
Keep author's composition P_u and target-specific/contextual laws distinct.
Review earlier NEXT74/NEXT75 selector compilation successes and limitations before a new finite
plan. Stop accuracy grids and further microscopic factoring of this same cache.
