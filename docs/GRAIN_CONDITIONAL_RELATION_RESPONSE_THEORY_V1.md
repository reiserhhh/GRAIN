# GRAIN: conditional behavior evidence, instrument response and author comparison

2026-10-10. HM114 finite construction and result-after direction diagnosis.
This is an addition to GRAIN, not a replacement for choice, expression, state,
event history or the original18 protocols/100cards/44 computational problems.

## 1. The two response objects must not be identified with each other

An author response function describes a conditional distribution of declared
behavior alternatives. An instrument response describes how the same evidence
changes a vector observation. Both depend on conditions, but neither is a
measurement of the other by definition. In the finite binary experiment,

    t_u(c) = E_u[b | c] = a_u + B_u*c,    c,b in {-1,+1}.

a_u is the level of a declared behavior-frequency contrast; B_u is its change
under a request versus its absence. These are known simulated parameters,
not Big5/MBTI or human psychological ground truth. Natural conditions need not
be binary, and a general response function need not be linear in c.

For supplied own-actor evidence and a fixed query q, the actual vector instrument
has endpoints K_q(c,b). Define

    m_cq = (K_q(c,+1)+K_q(c,-1))/2,
    g_cq = (K_q(c,+1)-K_q(c,-1))/2.

Then, on exactly these two behavior alternatives,

    y_ucq = m_cq + g_cq*t_u(c).

This identity comes from expectation on binary support, not from assuming that
the neural encoder is globally linear or that arbitrary interpolated vectors
are natural language. The endpoints were obtained from actual frozen prefixes.
Different action families are separate supports with their own m/g.

## 2. Raw conditional change contains three different terms

    y_u,+,q - y_u,-,q
      = (m_+,q-m_-,q)
        + a_u*(g_+,q-g_-,q)
        + B_u*(g_+,q+g_-,q).

The first term is the condition's effect on the instrument offset. The second
mixes the author's overall behavior level with condition-specific instrument
gain. Only the third directly represents the declared frequency-response B.
Thus B=0 does not imply equal raw conditional scores.

Conversely, the first two terms are not automatically noise: helping without
being asked can have a different social meaning from helping after a request.
Known frequency invariance does not establish psychological invariance. The
reader's contextual interpretation is an observation object to retain and
eventually interpret, not a nuisance to subtract indiscriminately.

For each family and unit, stack all six target observations under both
conditions, with rows [g_-, -g_-] and [g_+, g_+]. A frozen pseudoinverse of this
12x2 D identifies (a,B) on its range. Null directions stay explicitly unknown.
This is standard linear inverse/calibration mathematics. Local exact recovery
under supplied endpoints is an algebraic capacity control, not a discovery
of a psychological trait or a prediction-accuracy result.

## 3. Retain choice and a fixed reference, rather than deleting conditions

For actual opportunity shares pi_u, the contextual target observation is

    Theta_uq = sum_c pi_u(c) * [m_cq + g_cq*t_u(c)].

For a declared common reference pi_star, also retain

    Theta_star_uq = sum_c pi_star(c) * [m_cq + g_cq*t_u(c)],
    Choice_uq = sum_c [pi_u(c)-pi_star(c)] * [m_cq+g_cq*t_u(c)].

Then Theta=Theta_star+Choice. This is an accounting identity, not a causal
identification theorem. Different authors' opportunity supplies and choices
are not disentangled by a count ratio alone. The binary experiment supplies
shares and condition support explicitly; naturally occurring supply remains
a separate observation/design dependency.

Within one fixed conditional instrument, matched-author differences remove
the common offset m_c, but differing opportunity compositions do not. This is
why neither unconditional centering nor one overall calibration is sufficient.
GRAIN C1 choice and C2 conditional response remain distinct and retained.

## 4. Across wording, separate response directions from score origins

Actual pre-read endpoints have the exact finite expansion

    h_c,b = h_bar_c + b*Delta_h_c,
    Delta_h_c = (h_c,+ - h_c,-)/2.

Here h is K, V or token-averaged V in the same captured coordinates and token
positions. For the uniform-value observation, projecting mean Delta_V through
the frozen target covectors exactly gives g_c. The query-weighted observation
also depends on K and nonlinear attention weights; it is not the same linear
projection, nor a full native model judgment.

HM114 result-after diagnostics found matched direction cosine averages:

| Object | Mean cosine across12 family/condition matches |
| --- | ---: |
| Pre-read K behavior difference | .828604 |
| Pre-read V behavior difference | .829274 |
| Token-mean V behavior difference | .838542 |
| Six-target uniform-value effect vector | .940789 |
| Six-target query-weighted effect vector | .945518 |

For target-effect vectors the exact identity label matching ranked first among
all720 family permutations, with means .306182/.404236 over those permutations.
This is exploratory matching in six designed families, not a preregistered
human significance test. Behavior clauses and current carrier were identical;
only opportunity wording changed. It does not prove general paraphrase,
cross-language, new-topic or psychological construct transport.

Meanwhile source chart transport produced B-coordinate RMS errors
.313720/.341675. The affine diagnostic accounts for these exactly:

    [a_hat,B_hat] = d + T*[a,B],
    h_c = <g_source,c,g_target,c>/||g_source,c||^2,
    delta_c = <g_source,c,m_target,c-m_source,c>/||g_source,c||^2,
    T = [[mean(h), (h_+-h_-)/2],[(h_+-h_-)/2,mean(h)]],
    d = [mean(delta),(delta_+-delta_-)/2].

In the fixed symmetric numeric profiles, d_B accounts for96.91%/92.86% of the
B-transport MSE; this is an error decomposition, not a fraction of personality
information. No fitted transport correction was applied to primary results.

Therefore transport of a score origin, transport of a response direction and
transport of a psychological construct are distinct. This actual vector example
supports keeping an origin and a condition-indexed action field separately.
It does not authorize treating all origin shifts as unwanted noise, or treating
the complement of the six target covectors as meaningless: old GRAIN residual
and expression channels remain available.

## 5. Shared target responses are instrument geometry, not author correlation

Each behavior family has a six-target response vector under each condition.
With supplied equal family weights1/6, the two-condition mixed-evidence design
is [[G_-/6,-G_-/6],[G_+/6,G_+/6]]. All eight own/other, bank/unit matrices have
numerical rank12 under the frozen tolerance. Own query-weighted designs have
condition numbers89.899 and39.062. No multi-family population distribution,
author covariance or independent psychological factors were estimated.

Family response vectors overlap: the agreeableness-associated and openness-
associated behavior controls have cosine .984344/.935845 at the strongest
conditions in the two banks. Those are instrument effect cosines, not a
correlation between Agreeableness and Openness in people. Shared reading
channels and shared human dispositions must be established separately.

## 6. Prior theory, originality and the next research decision

[Mischel and Shoda1995](https://psychology.columbia.edu/sites/default/files/2016-11/246.pdf)
pp246-247 already distinguish overall behavior levels and characteristic
if-then conditional patterns. Their principle is prior art, not our discovery.
M01/M04/M11/M12 already supply condition scope, relational observations,
response sharing and conditional calibration in this project.

The [OpenAI fixed matrix-moment manuscript section](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex)
lines19-109 was re-read for complementary observations and a fixed family with
range/conditioning checks. Its harmonic-jet/PDE assumptions are not transferred
to text. Binary expectations, pseudoinverses, SVD, projections and permutation
comparisons are mature mathematics, not new theorems.

The incremental contribution here is a runnable scoped-condition vector
instrument connected to cold author level/response/choice observations, and the
empirical distinction between the transport of its origin and action direction.
The next question is whether the partly common conditional action field can
read new behavior expressions and GRAIN author relations without just retaining
the same behavior clauses. Inventory HM105 and old conditional-response worlds
before choosing a new finite test. Do not optimize B reconstruction, chart
condition number, target selection or teacher fidelity to rescue transport.
Do not repeat another known if-then calibration as a mainline breakthrough.

Natural evidence ownership/condition inference, native psychological judgment,
independent construct interpretation and original protocol scope remain open.
This finite construction needs no new human experiment to be investigated
further, but it is not human psychological validation or a universal scale.
