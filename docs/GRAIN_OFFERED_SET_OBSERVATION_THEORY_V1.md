# Offered-set relations in a conditional text observation law

2026-10-11. HM135 theory companion, outside the sealed experiment inputs.
This extends the evidence interface, not a personality-accuracy objective.
The empirical section uses complete collection and read-only numerical replay.
The original18 protocols/100 cards/44 computational questions remain OPEN.

## The construction problem

GRAIN must distinguish a performed action from an unperformed alternative and
distinguish an actual opportunity from a narrator's comparison. These are
relations among observations, not additional interchangeable keyword features.

Consider the same person helping briefly. Helping briefly rather than refusing
and helping briefly rather than devoting the entire session need not have the
same evidential value. However, an unavailable option is not a rejected choice.
Which alternatives were available cannot be recovered from the performed act
alone. A missing opportunity record is not evidence of a closed opportunity.

Keep the interface explicit:

    event e = (actor, performed action, condition, modality, time, source)
    offered-set record M = (option identities, availability, evidential source)
    report relation r = (performed, available-not-performed, unavailable, unknown)
    observer y_q = F_q(e, M, r)
    psychological likelihood ell(e | theta, M, observation design)

The observer's action, the real person's choice law and the reporter's account
are three different objects. Their numerical equality is not assumed.
Unknown M must remain unknown; a model-proposed local comparison set can be
stored as proposed, not converted into a historical fact about the person.

## Standard choice geometry: a known source of conditional evidence

For a declared finite option set M and action features f(j), use the familiar
exponential-family example:

    A_M(theta) = log sum_{j in M} exp(theta^T f(j))
    log p(a | theta, M) = theta^T f(a) - A_M(theta)
    L(a,M) = (theta_high-theta_low)^T f(a)
             - A_M(theta_high) + A_M(theta_low).

The numerator is fixed when the performed action is fixed, but the normalizer
depends on alternatives. This is why a behavior-only map can omit diagnostic
information even when it predicts that behavior precisely. It is not a new
identity or a claim that a pretrained model implements this particular law.

Two standard differential identities make the informational limitation clear:

    grad_theta log p(a | theta,M) = f(a) - E_theta,M[f]
    -Hessian_theta log p(a | theta,M) = Cov_theta,M[f].

Inference is relative to the feasible set's expected action, not an absolute
score attached to the performed action. If M has only one feasible action,
the likelihood is constant and this choice supplies no parameter information
under this model. The words used to report it may still carry other information.
Unobserved M, unknown features and utility-scale ambiguities prevent an observer
score from identifying psychological parameters without further assumptions.

The executed illustrative calculation fixes f(a)=0 and theta_high/low=+1/-1.
A feasible alternative f=-1 gives L=+1; f=+1 gives L=-1. Removing either
alternative from M gives L=0. These values are declared-model consequences,
not measurements of a person or predictions fitted to HM135 outcomes.

## What the new experiment can distinguish

HM135 keeps the moderate performed act and its voluntariness fixed. Two
alternative descriptions and two availability descriptions form four corners:

    D_closed = F(high alternative, closed) - F(low alternative, closed)
    D_open   = F(high alternative, open) - F(low alternative, open)
    I        = D_open - D_closed.

An additive option-word response alone gives I=0, even when both D values are
large. Nonzero I therefore rejects that additive description on the registered
support. It does not distinguish every generic semantic interaction from a
relative-choice likelihood. Actor/non-actor and unchanged-action readings are
retained to expose the narrower alternative explanations, not tuned away.

The frozen observer uses H-only suffix access. A 19-token availability clause
is swapped separately in K, V and both across all28 blocks. The complete
intervention field is decomposed into K-only, V-only and their joint remainder.
These signed components describe this operation. They are not psychological
information percentages or a universal K=person/V=trait division of labor.

The decision after collection is whether the account must preserve offered-set
relations and availability-dependent observation. There is no accuracy target,
teacher-score fitting, prompt rescue or head/rank grid in this packet.

## Empirical result and changed decision

All196 batches/2744 endpoints collected in900.325241 modelseconds under the
original1100-second cap. Nine execution guards are exact; alternate arithmetic
error is3.55e-15. All672 effect rows/8064 component rows replay exactly.
The33 focused tests and3 Python Ruff checks pass; no model fit or rescue was used.

The actor's option-by-availability score interaction RMS is .013720/.007970;
the non-actor's is .035010/.014507. The unchanged action fact also changes by
up to .335110/.311630 for A. No matching actor diagonal is negative in both
wording banks, as the declared simple relative-choice example would predict.
Thus the additive option-cue description is insufficient, but a clean
actor-specific choice likelihood is not supported by these controls.

Actor K/V/joint signed projections are [.716137,.861672,-.577809] and
[.197978,.558128,.243894]. The first includes cancellation; neither is an
information percentage. Availability has an internal operation on this frozen
reading support, not a proven Bayesian algorithm or a new human-personality law.

Preserve offered-set relations in the account, but stop accuracy/prompt/wording
rescue of this reader. Before another packet, distinguish a meaningful
diagnostic contrast from generic semantics and check M03/F03/HM119/132-135 for
already executed mechanisms. Do not automatically repeat this factorial with
a better score. Parent report: GRAIN_UNCHOSEN_OPTION_MECHANISM_REPORT_20261011_ZH.md.

## Prior work and the actual increment sought

- Baker, Saxe and Tenenbaum2009, *Action understanding as inverse planning*,
  formalize goal inference conditional on action and environment. This supports
  the generative-inference framing, not a new Big5 estimator here.
  https://doi.org/10.1016/j.cognition.2009.07.005
- Chen et al2026 preprint, *Inverse planning of social interactions in relationships*,
  uses LM-elicited local alternatives/features and a utility-based Bayesian
  model (Methods Eq3-7). Its comparison set is observer-constructed, not the
  actor's complete historical menu. Its primary correlations concern
  scenario-averaged judgments; post-hoc comparison reweighting is disclosed.
  Our proposed increment is a frozen observer's internal availability operation
  and a provenance-preserving GRAIN interface, not inventing alternatives or
  Bayesian social inference. Found during collection, not a pre-freeze source.
  https://arxiv.org/html/2609.01639v1
- Hu et al2023, *Expectations over Unspoken Alternatives Predict Pragmatic
  Inferences*, links context-dependent conceptual alternatives to pragmatic
  interpretation. Its scalar-language estimand differs from personality
  judgment, but makes an alternatives-as-novel-idea claim inappropriate.
  https://aclanthology.org/2023.tacl-1.50/
- Cao, Cohen and Szpruch2021, *Identifiability in inverse reinforcement learning*,
  gives a formal warning that a policy need not uniquely identify its rewards.
  Its MDP theorems are not applied to our language histories.
  https://proceedings.neurips.cc/paper/2021/file/671f0311e2754fcdd37f70a8550379bc-Paper.pdf
- OpenAI/math `adc7f1241b42e322a6451854ab7e4b4c146bf78a`, Hellinger-contraction
  induction/compression: preserving a mean while changing a conditional profile
  motivates a discriminating intervention. This is design inspiration only;
  no Boolean-kernel inequality or new high-math theorem is transferred to LLMs.

## Consequence for the research loop

M03/F03 already represent opportunity supply and selection. HM134 already
tested chosen versus imposed acts. HM135 is not a rebranding of those results:
it tests availability of an unperformed alternative with the act held fixed.
Conversely, this packet does not supersede those author-level measurement
models. It supplies an observation-interface operation they can retain.

The useful next step depends on the complete field. A native conditional effect
would justify preserving the offered-set coordinate. Actor-specific diagnostic
interpretation requires the relevant actor/fact distinctions to survive, not a
higher correlation with this observer. A broad semantic effect remains an
operation result but must not be labeled a clean psychological choice mechanism.
