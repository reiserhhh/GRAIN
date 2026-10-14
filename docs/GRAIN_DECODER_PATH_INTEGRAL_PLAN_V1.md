# HM25: Finite Decoder Paths and Observation Covectors

Frozen before execution. Extend all48 HM24 addressed references, not all2172
old interventions. Source/validation/application roles remain24/12/12. This is
a mechanism study of a fixed Qwen reader, not a human personality test.

## Scientific Question

HM24 separates two failures: a summary can discard readout-active token directions,
and a reference-point derivative can miss a finite property edit even when the
property summary is injective. Test the second failure without adding predictor
hyperparameters. Write Fq(H)=sq(zq(H)) and gamma(t)=H0+t Delta, where Delta
changes only the original A_property positions with the original donor states.

The scalar differential gives the identity

    Fq(H1)-Fq(H0) = integral_0^1 <dFq(gamma(t)), Delta> dt.

This is standard path attribution / Integrated Gradients, not a new theorem.
The averaged covector is path/reference dependent, not a universal personality
axis. The total exact differential is endpoint determined; component allocations
can be path dependent. No claim of nonzero scalar holonomy or a unique neural path.

Define the frozen candidate-logit covector g0=dsq(zq(H0)). Separate

    B = <g0,zq(H1)-zq(H0)>
    D = B-<dFq(H0),Delta>       [decoder nonlinear increment]
    C = Fq(H1)-Fq(H0)-B        [candidate-readout nonlinear increment].

Then endpoint-minus-start-tangent is D+C. Integrate native and frozen-readout
derivatives separately. A large D with small C locates missing variation upstream
of the output readout, not simply in the final probability transformation.

## Execution and Acceptance

- Use9 fixed uniform nodes and all48 previous reference/query addresses.
- Each node: one fixed-model forward, two backward covectors, no parameter update.
  Store six span gradients, actual patched property state, candidate logits and
  full readout compliance. Baseline calls additionally measure the new frozen-
  readout derivative; they are not a replay of the previous six-direction packet.
- Check original live body/query states exactly before patch; node0 and node1
  candidate logits must reproduce original baseline and full-property patch.
- Primary Simpson8; Simpson4 and trapezoid8 share nodes. Preserve quadrature
  residuals and gradient/replay failures, never select node count from outcomes.
- Report error in score units, sign changes of the path derivative, decoder/readout
  increments, six-span covector rotation and property integration. Gradient energy
  is not psychological variance and off-manifold patch interiors are not natural
  text observations.
- Verify with separate explicit Simpson sums and extended-precision contractions;
  independent arithmetic by the same agent is not independent human replication.
- Zero API/private reads/new neural training/commit/push/lockbox/Claude/old queue.
  Stop at432forwards/864backwards or30minutes. Preserve resumable per-node records.

## Reading Basis

[Sundararajan et al.2017](https://proceedings.mlr.press/v70/sundararajan17a.html),
sections2-4: point gradients may miss finite effects; path integration and component
allocation must be distinguished. [AtP*2024](https://arxiv.org/html/2403.00745v1),
sections2-3.1: gradient approximations require actual intervention checks.

OpenAI/math fixed commit adc7f1241b42e322a6451854ab7e4b4c146bf78a, metric/unitary
connection manuscript, section07-global lines237-272: borrow the discipline of
transporting observations and units together. Its PDE uniqueness/gauge theorem
does not apply to this transformer, and this experiment does not assert it does.

## Next Decision

If integration recovers finite effects but the start tangent does not, maintain a
context-dependent differential field / finite action, not a fixed global axis.
If it fails, use the retained quadrature, precision, and endpoint logs to separate
bad numerical integration from the actual observation law; do not rescue by grids.
After that, source-learned compression must preserve the action of this field;
the experiment itself remains model dependent and does not finish LLM-free scoring.
