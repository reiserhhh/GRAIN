# A common-law measuring rule: evidence refinement is not model replacement

HM275,2026-10-14 JST. This is a logical correction to interpreting GRAIN's
own SOURCE-reader contacts and a consolidation of its cold rule. It is not
a novel Bayesian theorem, empirical personality accuracy or a new factor.

## 1. Mainline result

A negative SOURCE-coupling/reader-departure contact can occur with an EXACT
common-target conditional reader. A positive contact can occur with the
opposite target. Consequently, moving a contact toward a positive sign is
not itself progress toward correct target measurement.

This does not invalidate HM271-274's actual interventions, source structure
or arithmetic. Their contacts remain useful descriptions of how two chosen
functions change together. It corrects any stronger reading of their signs
as a necessary or sufficient criterion for target agreement.

"Exact reader" below means the true conditional expectation of the declared
target in a fully specified world, NOT zero event-level prediction error.

## 2. The consolidated candidate measuring rule

Fix an evidence qualification Q,encoder phi,reference population pi,
conditional observation law L_g and declared target key b_q(g). For an
addressed pair H=(X,Z):

    p_L(g|H) = pi(g) L_g(H) / sum_h pi(h)L_h(H)
    eta_L,q(H) = sum_g b_q(g)p_L(g|H)
    relative_L,q(H) = eta_L,q(H) - sum_g pi(g)b_q(g).

The relative position needs a fixed reference; it must not depend on the
other authors presented in the same application batch. Choice,condition,
state and event relations are kept in L,not declared removable noise.

For existing HM273,phi is the frozen E5/GRAIN15 map,L=F/J/I/R are the
positive SOURCE-observed or resampling laws,pi is the empirical32-profile
prior,and b comprises the old authorized five binary SOURCE-label keys.
This is a label-associated conditional reading. The Gaussian working law
is not thereby calibrated to human trait probabilities or behavior.

The parallel neural channel is different:

    ell_j(H) = log p_model(complete IPIP sentence_j | H,neutral writer request)
    B_q(H) = mean_high ell_j / 2 - mean_low ell_j / 2.

The50 complete sentences are not exhaustive outcomes of an observed human
questionnaire. Their balanced log-likelihood contrasts are not eta_L,q.
No softmax over the50 sentences or logit of B creates that missing equality.
Both channels can be useful,but their numerical units and meanings differ.

This distinction is retained rather than fitting native answers or erasing
context to manufacture agreement. Existing HM252 supplies a different
operational-behavior bridge when independently qualified (C,Z,W,Y) data and
its proxy conditions are available; that finite construction remains intact.

## 3. Three different increments

Within one law F,let etaX=E_F[Y|X],etaZ=E_F[Y|Z],etaF=E_F[Y|X,Z]. Define

    D_evidence,left = etaF - etaX
    D_model = etaF - etaJ
    D_mean = etaF - (etaX+etaZ)/2.

The first adds an observation within the same probability space. The second
changes the observation model on the SAME evidence. The third changes the
aggregation reference. Neither J nor the singleton average is generally
the posterior of a coarser observation in F. There is no theorem requiring
E_F[D_model*D_mean] to be positive,even for the correct target reader.

Within the common law the genuine refinement identity is

    E_F[D_evidence,left | X] = 0
    Risk_F(etaX)-Risk_F(etaF) = E_F[D_evidence,left^2] >= 0.

This is conditional expectation orthogonality under squared loss. It is an
expected statement under the specified law,not improvement for every
realized author. It does not apply automatically to an LLM's phrase
likelihood,to the change F-minus-J,or to a changed evidence-sampling regime.

## 4. Exact sign counterexample and analytic family

Let Y~Bernoulli(.5),X and Z be distinct binary observations with identical
single-observation accuracy a>.5. Given Y define

    v=a(1-a), k=rho*v
    A=a^2+k, B=v-k, C=(1-a)^2+k, D=A+C
    F(correct,correct)=A; F(correct,incorrect)=F(incorrect,correct)=B
    F(incorrect,incorrect)=C.

These are conditional probabilities,not human response parameters. Require
A,B,C>0. J is their conditionally independent product,with k=0 and the SAME
singleton laws. Put d=a^2+(1-a)^2. On the concordant positive observation,

    etaF=A/D; etaJ=a^2/d; etaX=etaZ=a
    etaF-etaJ=-(2a-1)k/(Dd)
    etaF-a=(2a-1)B/D.

The negative concordant cell has reversed increments; discordant cells
have etaF=etaJ=(etaX+etaZ)/2=.5. Therefore the entire population contact is

    E_F[D_model*D_mean] = -(2a-1)^2 k B/(D d).

It is negative for positive rho,zero for conditional independence,and
positive for negative rho. Nonetheless the exact same-target reader has

    Risk_F((etaX+etaZ)/2)-Risk_F(etaF) = (2a-1)^2 B^2/D > 0.

This is redundant-evidence correction,not deletion of personality signal.
The wrong-key reader 1-etaF reverses the contact sign and has positive
regret against the unchanged target Y. Thus contact sign is neither a
necessary nor a sufficient test of common-target meaning.

## 5. Executed finite result

All three predeclared worlds,a=.7,pi=.5,rho in [0,.5,-.25],were enumerated:

| Conditional dependence | Exact same-target contact | Wrong-key contact | Exact reader gain over singleton mean |
| --- | ---: | ---: | ---: |
| Independent rho=0 | 0 | 0 | .012165517 |
| Redundant rho=.5 | -.003849847 | +.003849847 | .002232911 |
| Complementary rho=-.25 | +.008003630 | -.008003630 | .023210526 |

In the redundant world the exact reader risk is .199367089,not zero. Its
regret relative to the true conditional expectation is zero; the wrong-key
regret is .202531646. Adding Z after X has innovation energy/risk gain
.010632911. These are population census quantities,not fitted accuracy,
sampling significance,Pearson correlations or psychological information
shares. There were no simulated subjects/seeds chosen for acceptance.

## 6. Actual cached-vector integration

The new SOURCE rule was compiled for all528 old addressed adjacent pairs,
both frozen spaces and bank0/bank1/pooled references. Full32-profile/5-target
readings,conditional log-BF updates and all reference types are retained.

Example pooled GRAIN15 N: F-minus-J RMS .000249221,adding the second
observation after the first RMS .003395244,difference RMS .003373366.
They are numerically different objects,not measures of better/worse accuracy.
The packet does not attribute the natural HM274 negative contact to rho=.5
or prove its native reader is calibrated. That requires common-criterion
evidence not supplied by the counterexample.

One2.638682-second compute stage,zero neural/API/private/train/application
label calls. Existing F/J values reproduce HM273 exactly. A separate cold
CLI reads one pair from the numerical source bank without an LLM. Upstream
E5 encoding remains neural. Overlapping pairs are not multiplied as
independent evidence. Historical representations and panels are exploratory.

27 Fraction scalar checks and500544 literal SOURCE values: maximum error
1.67e-16/8.88e-16;old F/J error0,telescope0,cold CLI8.88e-16.35 focused tests
and5Python lint/format PASS. Same-agent verification,not independent blind
replication. Before freeze: missing test shebang/import order/format fixed.
No scientific retry or postfreeze code/science edit occurred.

## 7. What changes next

Stop using "make the diagonal contact positive" as the route to target
meaning. Keep signed contacts as observational diagnostics. Future claims
of target agreement must declare one common operational criterion and its
source observation law; future refinement claims must say what new
observation was added rather than silently replacing a model or reference.

The priority is a SOURCE-supported target rule and its usable behavioral
meaning,not another head/kernel/precision or contact-rescue grid. Known
label-linked measuring functions remain valid at that level. Unknown
behavior links are not set to zero or inferred from raw contact signs.
Original18/100/44 remain OPEN;no new packet or job is queued.

## 8. Prior art, scope and reproduction

[Reid/Williamson2011](https://jmlr.org/papers/volume12/reid11a/reid11a.pdf),
Sections4.1-4.6,places target probabilities,risk and statistical information
in a common joint law. The paper's more general divergence equivalences are
not newly proved or fully implemented here. Bayes/tower/orthogonality and
the conditional-dependence correction are established mathematics.

[OpenAI/math pinned foundations](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex),
98-116,166-232,motivates a common observation law and explicit independence
before composition. No tree/spin/reconstruction threshold is transplanted.
HM174/HM211/HM252/HM273 are prior project components,not failed versions.
The new delta is the interpretive counterexample and typed integration.

Canonical: results/grain_math_response_v1/measuring_rule_r1.

```sh
python scripts/score_grain_measuring_rule_cold.py \
  --bundle results/grain_math_response_v1/source_event_coupling_r1/source_pair_bank.npz \
  --record results/grain_math_response_v1/source_event_coupling_r1/example_pair.json
python scripts/run_grain_measuring_rule_hm275.py check
python -m pytest -q tests/test_grain_measuring_rule.py \
  tests/test_grain_source_event_coupling.py tests/test_grain_same_regime_reader.py
```

For fresh reproduction,run prepare,compute and the separate verifier against
a NEW --out directory. Never overwrite canonical results or repeat its
once-only verifier into the same directory. No original lockbox was opened.
