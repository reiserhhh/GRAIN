# Joint operational behavior from paired vector evidence

HM276, 2026-10-14 JST. A finite SOURCE-calibrated measurement constructor,
not a new covariance theorem or an accuracy promotion.

## What this adds

HM252 reads marginal operational behavior means from vector proxies. HM273
preserves actual paired vector laws. HM275 separates evidence updates from
observation-model changes. HM276 adds a bridge for a JOINT operational outcome
law, followed by an author-level coupling functional. Authors can have identical
single-condition means and different paired behavior relations. Naming a stable
linear factor axis is not a prerequisite for this specified functional.

This capability is verified in an exact simulator census. Its behavior endpoints
are two finite binary actions, not established human personality constructs.
Geometry supplies observable distinctions; the separately specified criterion
law supplies their behavioral meaning. Neither part alone supplies both.

## Qualified observation and identification assumptions

Two fixed offered opportunities A then B define the frame. Each observation
contains two distinct paired proxy bundles Z=(Z_A,Z_B), W=(W_A,W_B), and two
separately recorded operational outcomes Y=(Y_A,Y_B). Pairing, author and role
identity are part of the observation law. Distinct IDs are only an identity
guard, not proof of statistical independence.

For an unnamed simulator regime U assume W independent of Z conditional on U,
and E[1{Y=y}|U,Z]=P(Y=y|U). Require existence of a bridge and separation of
the relevant regime functions by Z. NO conditional independence of Y_A and Y_B
is assumed. These are substantive assumptions, not consequences of a small
equation residual or full rank in a finite sample.

Write the W emission matrix A, posterior U given Z matrix B, and joint outcome
matrix C, so the observed conditional matrices obey

```text
K[z,w] = P(W=w|Z=z) = (B A)[z,w]
m[z,y] = P(Y=y|Z=z) = (B C)[z,y]
K H = m.
```

If B separates the relevant columns, B(AH-C)=0 entails AH=C. For a compatible
new author's W law p_u=pi_u A, each joint outcome is

```text
q_u(y) = p_u H[:,y] = (pi_u C)[y]
theta_A = q(10)+q(11)
theta_B = q(01)+q(11)
chi_u = q(11)-theta_A*theta_B.
```

The four q values are identified linear functionals before chi is calculated.
If an equation admits several bridges, a reading must annihilate its nullspace;
the reused HM252 functional checker implements that restriction. Joint bridge
weights need not be probabilities. An inadmissible aggregate distribution is
returned explicitly, with unclipped cells; it is not normalized into success.

Changing source mixture alone preserves the readings in this design because
A and C remain fixed. This is NOT transport under an arbitrary new scenario,
language, criterion definition or emission law.

## Exact model and the three different products

The two regime bits have sign s_j=2u_j-1. W accuracy is .8, Z accuracy .75.
Behavior means are b_j=.3+.4u_j. Joint11 is b_A*b_B+gamma*s_A*s_B.
Both declared criterion protocols have positive outcome cells: gamma=.04 or
-.08. Three author laws pi_k(U)=(1+kappa*s_A*s_B)/4 use kappa=-.6,0,.6.
All author bit marginals and marginal behavior means are identical.

Let r=2*.8-1=.6. Then

```text
E_u[s_WA*s_WB] = r^2*kappa = .36*kappa
chi_u = (.04+gamma)*kappa
      = ((.04+gamma)/.36)*E_u[s_WA*s_WB].
```

The single-outcome calibrated weights are h_j(w_j)=1/6+(2/3)w_j. The joint
indicator bridge includes the conditional outcome dependence:

```text
h_11(w_A,w_B) = h_A(w_A)*h_B(w_B)
                + (gamma/.36)*s_WA*s_WB.
```

Thus these operations must not be conflated:

1. Product of author means reconstructs independent outcomes, giving chi=0.
2. Averaging products of single-event calibrated weights gives .04*kappa.
3. Reading the calibrated joint indicator gives (.04+gamma)*kappa.

Protocol B reverses the sign of the third relative to the second. Vector
association alone cannot decide which criterion meaning applies.

## Executed findings

All12 readings (two criterion protocols x two source mixtures x three author
laws) match the independently retained outcome law. Every author has means
.5/.5. Couplings are [-.048,0,+.048] for A and [+.024,0,-.024] for B, in
both source mixtures. Single-weight products return [-.024,0,+.024] instead.
Replacing each paired vector law by the product of its own marginals gives
zero coupling, preserving every marginal. Independent-process cold scoring
of the positive-kappa author under A returns relative coupling .048.

One CPU computation .108981s, no neural forward/API/private data/training.
452 Fraction-based census/equation/functional checks pass, maximum alternate
error1.263e-15. Focused tests33 and five Python lint/format checks pass. This
is same-agent alternate arithmetic, not blind replication. The four historical
GRAIN15 vectors are an EXACT finite alphabet, not a natural embedding decoder.
Complete weighted laws were supplied; no short-text sampling CI is claimed.

## Reproduction and ancestry

Canonical output: `results/grain_math_response_v1/behavior_coupling_r1`.
Frozen plan: [HM276](GRAIN_BEHAVIOR_COUPLING_PLAN_HM276.md). Runtime interface:
`grain_research/construction/math_behavior_coupling.py`; cold CLI:
`scripts/score_grain_behavior_coupling_cold.py`. Reproduce only into a new
output directory using the runner's `--help`; do not overwrite the sealed run.
The archived `check` mode is read-only. The verifier uses rational enumeration
and Gauss-Jordan rather than the production SVD solver.

[Zhang et al.2023, proposition2.3](https://arxiv.org/html/2303.10134v4)
already gives functional identification despite nonunique bridge solutions.
We use a finite noncausal operator argument, not its treatment-effect theorem.
[Fu/Green2026, section6.2](https://arxiv.org/html/2505.21909v3) discusses
reference measurement bridges and sketches nonparametric extensions, whose
full results are deferred. [Sejdinovic/Gretton/Bergsma2013](https://arxiv.org/abs/1306.2281)
already studies joint interaction beyond marginal associations; no Lancaster
test is implemented here. [OpenAI/math pinned foundations166-232](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
informs conditional-product design only; no reconstruction threshold or new
psychometric theorem is transferred.

Local addition is an operational interface beyond marginal means. The simulator
verifies the specified construction, not discovery of a human mechanism. Next
inventory existing qualified public paired vectors and independently observed
paired actions on the SAME events. Do not launch another precision grid or
known-identity toy variant. Original18/100/44OPEN; Goalactive/nojob.
