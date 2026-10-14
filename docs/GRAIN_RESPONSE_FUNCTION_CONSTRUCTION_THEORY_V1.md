# Source-observed response function construction

2026-10-11. HM166 mathematical construction companion, written while the new
application collection was running. Numerical conclusions belong to the sealed
results/report. This is not a new mathematical theorem or a personality scale.
Original18/100/44 remain OPEN.

## 1. What changes in the measurement chain

An assigned persona label L, an article X, its writing condition Z, a later
opportunity C, option-order protocol O, and observed response Y are different
objects. Controlled articles are not people. At neutral current persona policy,
the measured native response is

    q(x,z,c,o) = P_model(registered positive action | rendered history).

HM165 observes this channel on actual historical article bodies, rather than
letting a reader restate the persona label. HM166 builds functions from those
source responses. L is not used to orient the new functions. New-article
scoring does not invoke a model to reason about personality, although source
response observation uses the local frozen Qwen model.

GRAIN4 here is the historical TF-IDF/SVD/factor bundle, applied to four lossless
lexical slices of article bodies. It is not an activation from the future
choice prompt. The four standardized views are lifted by

    Phi(x) = tensor_product_j (1,z_j(x)), j=1..4,

giving625 fixed coordinates, including joint-view products. A function linear
in Phi can be nonlinear in the original views. This does not prove that every
text relation or a neural inference circuit has been represented.

## 2. The observable witness and what it does NOT equal

For a source reference of n actual aligned articles and responses, define

    C_(c,o) = mean_i (Y_i(c,o)-Ybar(c,o)) (Phi_i-Phibar)
    T_(c,o)(v) = C_(c,o) dot (v-Phibar).

This is a concrete, source-observed signed functional. It has no guarantee of
being in[0,1] and is not E[Y|Phi]. In the ideal linear world Y=A Phi+eta with
uncorrelated eta, the centered cross-moment is C_YX=A Sigma_X, NOT A itself.
Changing source geometry can change the witness even when A stays fixed.
No application probability calibration is used to hide that distinction.

Song et al.2009 Sections2-3/Theorems4-6 explicitly distinguish covariance from
conditional embeddings. A conditional-mean operator requires an inverse/range
condition and estimation choices. Those assumptions are not automatically
satisfied by these625 features or42 controlled source articles. HM166 tests a
constructible response witness, not an inverse-problem solution.
https://alex.smola.org/papers/2009/SonHuaSmoetal09.pdf

## 3. Condition composition is retained, not removed

With source task/style block Z=g, total covariance gives the finite identity

    C = E_g Cov(Y,Phi|g) + Cov_g(E[Y|g],E[Phi|g])
      = C_within + C_between.

Both components are scored in the same source coordinates and denominator.
They can reinforce OR oppose each other; squared magnitudes are not additive
psychological information fractions. Between-condition behavior may reflect
choice, opportunity, writer policy or context. Within-condition association is
not by itself stable author identity. Neither is automatically noise.

Holding an article fixed while changing its opportunity/protocol produces a
family of functions, not necessarily one universal scalar trait axis:

    T_(c,o) = T_even,c + (-1)^o T_odd,c.

The same identity is applied separately to probability and log odds. Order
susceptibility is retained as an observation property. Even is not automatically
a position-free psychological variable. Every opportunity cross-reading is
reported rather than selecting the best endpoint after observation.

## 4. Author comparison is a further construction, not a relabeling

If a qualified author-event law P_u exists, these functions define

    Theta_u(c,o)-Theta_ref(c,o)
      = integral T_(c,o)(Phi(x)) d(P_u-P_ref)(x).

Reference and observation protocol must stay fixed. For condition-qualified
functions and P_u(z,x)=pi_u(z)P_u(x|z), the aggregate is

    Theta_u = sum_z pi_u(z) integral T_z(Phi(x)) dP_u(x|z).

An exact comparison can retain both response differences and selection:

    Theta_u-Theta_v
      = sum_z pi_u(z)[E_u(T_z|z)-E_v(T_z|z)]
        + sum_z[pi_u(z)-pi_v(z)]E_v(T_z|z).

This is ordinary finite mixture algebra. It shows WHY removing condition means
or selection changes the measured object, not why a human trait is identified.
Current synthetic article pairs do not supply repeated human P_u or behavioral
validity. A demonstrated cold response witness is nevertheless a positive
measurement-construction capability, not merely a repaired activation probe.

## 5. What a transport failure would identify next

The native observation uses article X AND the prior system/user context Z;
the cold article map observes Phi(X) only. A body-only witness may fail because
the source geometry, the source response law, or unrepresented context differs.
These are not interchangeable reconstruction errors.

If source task/style association dominates, the next mechanism question is
whether Z acts through the realized article, directly through retained history,
or jointly. A same-body context intervention can discriminate these; raising
Ridge quality cannot. Such an intervention would be a new observation-law
experiment requiring its own finite plan, not a prompt rescue or deletion of
useful context. If the source-specific within function transports, it provides
an actual article-response component to extend, still without calling it a
human personality dimension.

## 6. Prior art and contribution boundary

Preference representations, biography edits and positional/task components are
already studied by Gilg et al.2026v3 (Sections2.1/2.3, AppendicesA.3/L.2):
https://arxiv.org/html/2605.13339v3

OpenAI/math pinned adc7f124 Thorp section05 separates a fixed observation array,
conditional reveal laws, repeated labels, and entropy costs. It inspired the
law-versus-representation distinction, but its independent permutations and
density bounds have not been established here. No new Thorp spectral or entropy
theorem is claimed for text.

The present candidate contribution is a source-response-oriented GRAIN
construction whose units, conditions, source composition and new application
responses are linked by the SAME article IDs. Mathematical novelty, neural
mechanism identification and human psychological meaning are separate claims.
No accuracy grid, source inverse fit, post-outcome reorientation, API/private
data/neural training/Git/Claude/automation/lockbox is performed.

## 7. Completed finite observation: response linkage is conditional

All864 new application inputs completed in344.830 model seconds. The source
program and108 numeric cold readings were sealed before these observations.
All54 article pairs are complete. This is a finite model response-law experiment,
not a human personality accuracy experiment. HM165 remains PARTIAL_BUDGET.

In the evaluation role, the full probability-even witness has negative signed
alignment in all8 target/opportunity cells. Its within component has positive
social alignment in both article families: E .412924/.408681 and N
.516504/.396737 for free/required opportunities. The corresponding between
component is negative: E -.322425/-.353287 and N -.260231/-.204304.
The log-odds-even within social readings are also positive in all4 cells.
The effects are cosines of18 paired differences per cell, not Pearson r.

Source geometry and source response composition therefore cannot be collapsed
into one universal score. There is an observable article-response linkage, but
the source aggregate can oppose it under different application conditions.
Within is not universally superior: development E social within is negative,
and evaluation N planning-known within is -.3208. No component is selected as
the final scorer and no condition component is relabeled as disposable noise.

The within social functions cross-read the other social opportunity positively,
but the planning opportunities negatively in the evaluation role. This favors
an opportunity-family response object over a single persona-axis interpretation.
The source has no N observations: cross-family response linkage does not mean
identifying an N trait or constructing a new psychological factor.

A separate POST_OUTCOME address description shows that source cyclic relinks
2/4 preserve all42 persona poles but change every article and generation seed.
The actual within social cosines exceed both of these comparators in all4
evaluation cells. Odd relinks change every pole and are not an exchangeable
generic null. These finite descriptive comparisons do not produce a p-value,
prove unique identity linkage, or multiply the number of independent subjects.

The next missing distinction is whether the retained earlier writing task acts
through the article, directly through the continuing history, or jointly. It
requires an actual same-body context intervention. Increasing witness accuracy
or fitting an inverse would not distinguish these mechanisms. A future plan
must keep article identity, response opportunity and order fixed while changing
only the prior task context, and retain the full/within/between programs frozen.

Primary: results/grain_math_response_v1/response_function_transport_r1.
Post-outcome description: link_description_serialization_recovery_r1 beneath
that directory. The first descriptor left complete CSVs and an incomplete JSON
after a NumPy-int64 serialization failure; its outputs remain preserved.
The recovery changes serialization only, with zero new neural observations.
