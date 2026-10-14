# Evidence Response, Intervention and Author Measurement

2026-10-09. HM54 finite results; theory update, not a new scale or global closure.

## Distinguish the objects

For event e, author u and target q, distinguish:

    E_{u,e}: evidence with actor, event, modality and condition identity
    H_{u,e,q}: query-conditioned hidden state produced by that evidence
    Y_{u,e,q}=F_q(H_{u,e,q}): actual observer readout
    d_q: a source-derived intervention used to diagnose the observer

The measurement observation is not automatically F_q(H+d_q). The source
intervention can inject a source judgement or change downstream response.
For author aggregation, qualify the observation, retain its condition and
unit, and represent missing target evidence as unobserved rather than zero.
Another person's action is not the reporter's own action. The reporter's
choice, framing and context remain available as distinct evidence.

## Exact finite response law

For two fixed evidence states and the same signed source vector:

    A_sigma(E)=F_q(H(E)+sigma*d)-F_q(H(E))
    M_sigma=A_sigma(E1)-A_sigma(E0)
    D_sigma=D_native+M_sigma

An output-only constant log-odds offset has M_sigma=0. HM54 actual M is
nonzero on both new own-behaviour families and controls. Actual log-odds
evidence-contrast RMS shrinks from 6.891/7.713 to 2.271/2.750 under +d;
-d also shrinks it to 2.512/2.420. Both signs shrink 23/24 own contrasts.
This retains a nontrivial conditional action, but makes contrast compression
part of the observation law, not an author trait or an accuracy improvement.

Result-after decomposition on the already acquired data:

    M_even=(M_plus+M_minus)/2
    M_odd=(M_plus-M_minus)/2
    D_plus/minus=D_native+M_even +/- M_odd

Own-family even RMS 5.666/6.180 exceeds odd RMS .908/.966. This decomposition
is standard finite C2 arithmetic, not a learned personality axis or proof
that all hidden states form an equivariant group representation.

## Relation to curvature, with explicit mathematical assumptions

In a fixed-dimensional frame, let u=H1-H0 and v be the full-state padded
intervention. If the same downstream scalar map F is twice continuously
differentiable on the rectangle, the multivariate fundamental theorem gives:

    M = F(H0+u+v)-F(H0+u)-F(H0+v)+F(H0)
      = integral_0^1 integral_0^1 u^T Hessian_F(H0+s*u+t*v) v ds dt.

This is an ordinary calculus identity. The experiment has matching token
counts for all48 evidence pairs and explicit suffix slot maps; it has not
numerically integrated those full-state Hessians. It does not assume that
the fitted/queried coordinates are a person's latent psychological truth.

A nonzero mixed response can arise from downstream curvature with a fixed
additive payload. For example F(h)=h^2 yields M=2*u*v, without a reading
procedure encoded in v. Thus nonzero M excludes the specified constant
output-offset null but does not uniquely identify an evidence-reading
program. Under small v, the even intervention begins with curvature and
the odd intervention with a directional derivative; HM54 v is finite, so
these local asymptotics are not asserted to explain its measured magnitude.

## Mainline consequence

Keep intervention strength, fidelity to the native observer and measurement
of author differences as separate quantities. Better fidelity may help
inspect a mechanism; it can also reproduce contrast compression better.
It is not a sufficient theory criterion.

The next developmental object is passive evidence contribution versus
source-payload contribution under qualified actor/event binding. GRAIN can
organize repeated events and conditions, while the vector analysis locates
how the observer reads them. Do not automatically add an intervention to
every author score or remove context/choice information to make it stable.
Do not restart an alpha/head/layer/RMSE grid as this theory update.

This chapter does not close natural B_u, psychological/behavioural meaning,
LLM-free operation or original18/100/44 requirements. It specifies a useful
new question rather than assigning their work to humans or declaring it done.

## Prior Art and Inspiration

- Function Vectors Appendix A distinguishes procedures and output offsets:
  https://arxiv.org/html/2310.15213v1.
- Activation patching and its interpretation:
  https://arxiv.org/html/2404.15255v1.
- OpenAI/math signed-isotypic sector arguments inspire separating components;
  their density/positivity/group assumptions and angle bounds are not proved
  for these neural states and are not used as personality theorems.

New empirical response patterns and measurement interfaces are contribution
candidates; ordinary calculus, factorial contrasts and patching are not new.

## Reproduction

Canonical results: results/grain_math_response_v1/evidence_response_operation_r1.
Frozen programs: run_grain_math_evidence_response_operation.py and verifier.
Result-after companion: analyze_grain_math_evidence_response_parity.py;
04_response_parity explicitly records its status and zero model calls/fits.
113 focused tests and alternate scalar/patch reconstruction are complete.
No model training, API, private corpus, publication or new lockbox access.
