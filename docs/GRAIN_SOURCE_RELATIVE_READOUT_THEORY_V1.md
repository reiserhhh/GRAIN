# Source-qualified relative readout and nonlinear calibration

HM260, 2026-10-13. Sections1--5 written while the one CPU discriminator runs,
BEFORE inspecting its contact outputs. The mathematical statements are standard
finite probability/differential geometry. They do not invent a new theorem or
declare a human trait from a relative lexical field.

## 1. The unresolved mechanism is meaning, not decimal precision

HM259 observes SOURCE-linked route interactions on complete actual text.
Selected-logit minus partition is a valid likelihood account, but its two raw
terms change under an arbitrary common logit origin. A negative partition
contribution alone cannot identify a generic calibration mechanism.
HM153 already establishes the appropriate relative-field distinction; this
packet joins it to actual SOURCE/GRAIN events and joint computation contracts.

For saved prediction state h, output rows W_v and bias b_v, define

    z_v(h)=W_v.h+b_v,
    A(h)=log sum_v exp(z_v(h)),
    m_q(h)=sum_v q(v)z_v(h),
    s_q(h,b)=z_b(h)-m_q(h),
    a_q(h)=m_q(h)-A(h).

Then log p(b|h)=s_q+a_q. Both s and a are invariant to z_v->z_v+k at each
observation. q is explicitly declared, independent of route/pole/context;
it is NOT a latent personality reference or posterior. A uses the full
original vocabulary even if q has only finite observed-token support.

For another q', s_q'-s_q=-(m_q'-m_q) and a_q'-a_q=m_q'-m_q. Total evidence,
word-pair differences, and SOURCE contact of the total are unchanged.
Neither account is noise; their allocation is reference dependent.

## 2. A concrete measuring covector

On a declared q, the relative readout has covector

    omega_bq=W_b-sum_v q(v)W_v,
    s_q=omega_bq.h + b_b-sum_v q(v)b_v.

This is a condition-qualified observation of a prediction state, not a
personality word vector. b indexes an ACTUAL observed word. Given the same
exported states, basis and references, another numeric program can execute
this function without invoking the decoder. Producing the states is still
neural; this is not end-to-end LLM independence.

The SOURCE functional M applied to signed shape/calibration actions describes
how the declared readout corresponds to frozen author-relative SOURCE fields.
A high output norm or an arbitrary change in output origin does not supply
this correspondence. M[s]+M[a]=M[log p] does, with both components retained.
The equality is bookkeeping; whether a component has useful SOURCE meaning
is a separate data question, not guaranteed by naming it a measuring covector.

## 3. Two nonlinearities must not be conflated

For each fixed context/pole/token, let the four saved graph states be h00,
h10, h01 and h11. Put D=h01-h00, R=h10-h00 and

    J_h=h11-h10-h01+h00.

Because s_q is affine in h at fixed b/q,

    J_s=omega_bq.J_h,
    J_a=(sum_v q(v)W_v).J_h - J_A.

Thus relative-word interaction is an output-covector observation of graph
state nonadditivity, whereas calibration includes log-partition geometry too.
It is incorrect to attribute ALL observed score nonadditivity to softmax,
or to assume all calibration interaction is another hidden personality axis.

The standard derivatives are

    grad A = sum_v p_v W_v,
    Hess A = W^T(diag(p)-p p^T)W.

The Hessian is positive semidefinite, but D^T(Hess A)R may be negative.
Convexity therefore does not imply positive two-route interaction. More
importantly, a pure affine-parallelogram curvature formula omits J_h whenever
the graph states themselves are nonadditive. The exact finite bridge is the
bilinear numerical surface

    h(s,t)=h00+sD+tR+stJ_h,

for which the ordinary fundamental theorem of calculus gives

    J_A = integral_[0,1]^2 {
        grad A(h).J_h
        +(D+tJ_h)^T Hess A(h) (R+sJ_h)
    } ds dt.

This surface is an accounting path between observed endpoint vectors, NOT a
reachable neural or psychological intervention. The current packet does not
numerically integrate the full-vocabulary Hessian, and does not claim to have
localized curvature or mediation uniquely. It supplies the exact mechanism
types that subsequent constructions must distinguish, without another head
or precision scan.

## 4. Relative evidence is a measurable distinction, not automatic psychology

For two supplied alternative token rows,

    Gamma(v,w)=(W_v-W_w).Delta_h.

The reference-free observed-support range of Gamma rejects a strictly common
word field ONLY on that supplied support. It does not exclude grammar/style,
conditioning opportunities, token positions or generic task interactions.
The current references are actual-token frequency and uniform distinct IDs,
both fixed before outcomes. They are not a universal vocabulary measure.
Same-author use does not make every source-linked field a personality factor.

What this construction can add to GRAIN is an observation unit containing
relative word evidence AND scalar calibration under an explicit contract.
Source/reader meaning can be tested on that richer unit rather than inferred
from one arbitrary scalar or vector norm. Choice and state remain included;
no centering operation is used to remove topic or occasion information.

## 5. Ancestry and connection to the wider goal

[Wainwright and Jordan, section3.5.1, Proposition2](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf)
derives log-partition derivatives/covariance geometry. HM153 already uses the
categorical Fisher pullback, and HM222 already has route interaction.
[Activation patching2024, section2.2](https://arxiv.org/html/2404.15255v1)
separates mediation from direct amplification/calibration.
The new project question is their joint linkage to complete actual-text SOURCE
meaning, not priority for those tools or a new neural interaction theorem.

The [pinned OpenAI/math foundations](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
98--116/166--232 motivates an explicitly declared experiment before combining
observations. Its conditional independence and input-independent degradation
kernels have not been established here; no tree reconstruction theorem or
information fraction is claimed. The curvature formula above is mature
calculus, not a transplanted new OpenAI theorem.

Original18/100/44 remain open. Completing this discriminator does not identify
a psychological measurement function on arbitrary new people/conditions.
Only a mechanism that changes the source-supported construction decision
warrants another finite packet; no automatic reference/grid/cut/precision work.

## 6. Executed finite discriminator and construction consequence

HM260 completed once on ALL24 historical authors/48 directional uses and five
targets, with0 new model/API/private/training calls. Run wall time6.837532s;
analysis accounting time6.822086s. Canonical `source_route_readout_r1` retains
both predetermined references, all directions, choice/within and cross-fields.
This is a post-HM259 mechanism followup on the same observations, not another
independent sample or a psychological accuracy test.

The pooled FULL-source interaction contacts in E,N,A,C,O order are:

| Reference / account | E | N | A | C | O |
| --- | ---: | ---: | ---: | ---: | ---: |
| occurrence / relative shape | -.021247 | +.007444 | -.001546 | -.001998 | -.018850 |
| occurrence / calibration | +.005073 | -.010109 | +.009281 | -.006410 | +.021480 |
| unique / relative shape | -.019610 | +.007400 | -.000866 | -.001146 | -.019422 |
| unique / calibration | +.003436 | -.010065 | +.008600 | -.007263 | +.022053 |
| either / total | -.016174 | -.002665 | +.007735 | -.008408 | +.002631 |

These are covariance-like SOURCE contacts, NOT Pearson r, AUC, probabilities,
psychological information fractions or criterion validity. O has opposite
pooled relative/calibration signs under both references; their exploratory
marginal intervals exclude0 separately, while its total interval includes0.
Occurrence O shape[-.035612,-.002760], calibration[+.003813,+.040091]; unique
shape[-.037133,-.002984], calibration[+.004085,+.040496]. Its directional shape
contacts have opposite signs, so pooled cancellation is not a universal
author/direction rule. N calibration is negative in both directions under
both references and its pooled marginal intervals exclude0; N shape and total
intervals include0. C total is negative with marginal intervals excluding0,
but both component intervals include0. This does NOT uniquely locate C's
SOURCE-linked negative interaction in either mechanism.

Across18715 saved token positions, the observed-support interaction range has
mean.553799/max6.091126nats, compared with native mean.687206/max8.074234.
This establishes non-common relative-word action on the supplied support, not
that all such action is personality-specific. Shape being affine in h means
nonzero relative interaction exposes graph-state nonadditivity in these
readout directions; scalar log-partition curvature alone cannot explain that
relative component. No full-vocabulary Hessian integration was performed.

The new construction constraint is to retain a source-qualified observation
as `(relative_word_field, calibration, reference, joint_route_contract)`.
Near-zero total need not mean absent source correspondence, and selecting the
positive account or maximizing its amplitude changes the measuring rule.
Both signs and both references stay; no component is deleted as noise.
This advances a measuring-unit decision, not a new general theorem or a
completed measuring function on arbitrary new authors/conditions.

The1050 pooled FULL-contact scalar-loop checks have maxerror1.3878e-17 and
recovery6.2172e-15; head replay4.6494e-6, old total recovery4.1744e-14.
All parent/analyzed/verification seals PASS;18 focused tests/three Python
lint-format checks passed before freeze, with two old SWIG warnings. This is
same-agent alternative arithmetic, not blind replication. All run/verify/check
handles86029/38801/47669 are terminal0 and no experiment remains running.
Instrumentation is adequate for this distinction: do not continue a
precision, reference, head, layer or predictor grid. Next work must construct
a SOURCE-supported measurement rule with a declared meaning-preserving or
meaning-changing operation; inventory existing interfaces before proposing
that finite experiment. Original18/100/44 remain OPEN.
