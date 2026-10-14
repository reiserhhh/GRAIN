# Context-dependent target evidence composition: HM174

## 1. What changes in the measurement formula

HM171/173 used a fixed SOURCE event field psi_q(a) and an author event mean.
HM174 adds an explicit joint observation law. A one-event target field does
not determine how several same-author observations compose. Two source laws
can agree exactly on every psi_q(a), yet disagree on contextual increments.
The new capability is an executable SAME-event multi-target composition
interface, not a new Bayes, mixture, de Finetti or Deep Sets theorem.

The SOURCE catalogue has386 empirical8-cell author distributions P_u and
their authorized old binary Big5 outcomes Y_uq. All24 D2 panel authors are
excluded from SOURCE label calibration. Historical directions/cutoffs remain
reused, so this is not a fresh representation holdout.

    R(a)=mean_u P_u(a)
    pi_q=mean_u Y_uq
    psi_q(a)=mean_u[P_u(a)Y_uq]/R(a).

Keep three distinct rules:

    M_mean,q(E)=mean_{a in E} psi_q(a)
    M_iid,q(E)=pi_q product_{a in E}rho_q1(a) /
               [pi_q product rho_q1 +(1-pi_q) product rho_q0]
    w_u(E)=product_{a in E}P_u(a) / sum_v product_{a in E}P_v(a)
    M_shared,q(E)=sum_u w_u(E)Y_uq.

rho_qg is the equal-author mean of P_u within SOURCE target group g. Both
posterior rules and the event mean have the SAME singleton value psi_q(a).
The independence variable differs: target label versus source author index.
This is a declared empirical-resampling model. Real comments need not be
independent given an index, and source indices are not named psychological
types. Conditions/venue choices remain observations, not erased noise.

## 2. The explicit contextual evidence operator

For a supported new event a, the shared-author rule gives

    w_u(E+a)=w_u(E)P_u(a)/sum_v w_v(E)P_v(a)
    M_shared,q(E+a)-M_shared,q(E)
      = Cov_{w(E)}[Y_q,P_u(a)] / E_{w(E)}[P_u(a)].

Derivation: subtract E_w[Y] from E_w[YP]/E_w[P]. The numerator is precisely
E_w[YP]-E_w[Y]E_w[P]. It is standard finite Bayes/covariance algebra, not a
new identity. It gives the GRAIN interface a useful interpretation: the target
effect of the SAME observed cell can change because earlier evidence has
changed the weighting of source configurations.

All five targets use one common w(E), but different Y_q. Shared target updates
therefore have a joint author reference; their overlap is not automatically a
shared psychological circuit. For a sequence of distinct events, these
increments telescope to M_q(E)-pi_q. The working likelihood is order-invariant:
intermediate increments depend on history, whereas the final posterior does
not depend on event ordering. This is not an attention path or a statement
that real text/order is exchangeable.

## 3. An additive-mean and an additive-log-odds discriminator

In each old12-comment view, S and C are the SAME disjoint4-comment subsets;
R is the remaining4. For the five available natural conditions define

    J_mean(F)=F(R+S)-F(R+C)-.5[F(S)-F(C)]
    J_sum(F)=F(R+S)-F(R+C)-   [F(S)-F(C)].

The event mean has J_mean=0. Target-conditioned iid BINARY posterior log odds
have J_sum=0. Probability curvature alone does not violate that second null.
Shared-author posterior log odds need not satisfy it. For SOURCE one-cell
context b and new cell a, all64 combinations/five targets gave85 sign reversals
of a's log Bayes increment; increment-change RMS.187804 nats. These320 cases
are constructed SOURCE-law questions, not320 participants or a significance
test. Their mechanism exists in this law, not yet identified in human behavior.

Native D2 output is an ordinal expectation over digit0..9, not a binary
personality posterior. Do NOT logit-transform that expectation. Instead use
ALL9 saved categorical log odds z_k-z_0. Under a fixed additive event increment
model, even allowing arbitrary common count-dependent baseline b(n), J_sum
must be0 on these logits. Softmax/gauge normalizers cancel identically.

Actual48 views,24 authors:

- Native ordinal J_mean RMS.025573, all48 nonzero.
- Native9-axis pre-softmax J_sum RMS.563938 nats, maximum1.874708.
- All48 native vector contrasts nonzero; minimum vector norm.322591.
- At24 author means, native9-axis RMS.392514 nats.
- Mean-model identity max1.11e-16; iid binary-log-odds identity max8.88e-15.
- Shared-author binary-log-odds J_sum RMS.571628, maximum2.863593.

The native observation contradicts that simple additive increment model.
This is not only probability-readout curvature. It does not uniquely identify
the shared-author model: positions, formatting, text length, event order,
context reencoding and the question may contribute. Two views are not48
independent people; no new neural forwards or causal interventions ran.

## 4. What the comparison did not collapse

Matching RMS values does not establish identical mechanisms. On cue-mask,
shared-author N probability J_mean relates to the native ordinal J_mean only
r=.171121; target-iid N is-.340881. Raw secondary values are .168947/-.343302.
All five targets and both arms remain. No model, signs or source smoothing
were selected from these descriptive correlations. No human trait accuracy
was evaluated in this packet, and no high-r reader was promoted.

The frozen descriptive_links.csv also contains correlations of the mean
model's1e-16 roundoff. Its J_mean is analytically zero: those10 correlations
are undefined scientifically and MUST NOT be interpreted as associations.
The original CSV/science seal is preserved; post_outcome_description.json
explicitly records this diagnostic limitation rather than hiding/recomputing
the result. No mean-model correlation is a valid positive or negative control.

## 5. Construction, provenance and next mechanism

All480 cold families scored, no missing joint support or nonfinite posterior
log odds. SOURCE bundle, outcome-free event JSONL and a separate cold CLI now
implement three lawful rules without querying an LLM. The upstream E5 encoder
remains neural. A SOURCE catalogue posterior is not natural human truth.
25new/52focused tests and5Python lint/format pass; alternate product audit
12000 scalar values/432 native axes, max8.22e-15/0. Same-agent, not blind.
No raw/private questionnaire/application label/API/Git/training/Claude access.

Prior art: Deep Sets Sections2.2-2.3 explicitly connects nonlinear set
aggregation and exchangeable mixture laws:
https://arxiv.org/html/1703.06114 . LDA2003 abstract gives hierarchical text
mixtures: https://www.jmlr.org/papers/v3/blei03a.html . Existing GRAIN C2/HM142
already had likelihood integration. OpenAI/math adc7f124 three-state
foundations.tex120-255 motivates declaring the common latent variable before
normalized product composition. Its spin/tree/threshold theorem is not
transferred here. Newness is the SAME-natural-event construction/discriminator,
not inventing hierarchical Bayesian inference.

Next inventory should distinguish target-associated dependence from residual
source-author fingerprint dependence. A proposed comparator pools sources by
the FULL five-target binary profile while preserving its joint label prior
and singleton fields. Its within-profile pool removes author-specific mixing
without erasing observed topic/choice/state. This is a standard hierarchy
comparator, not automatically a new theorem or an authorized precision grid.
Check existing C1-C4/HM142 joint laws first; then freeze a finite mechanism
question only if this specific natural observation family is not covered.
Do not chase the .171 correlation or add posterior temperature/rank/priors.

Canonical results/grain_math_response_v1/joint_evidence_law_r1.
Original18/100/44OPEN. GoalACTIVE; all finite commands terminal; no next job.
