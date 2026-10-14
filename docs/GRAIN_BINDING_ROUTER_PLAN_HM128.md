# HM128: a neutral binding operation versus personality evidence routing

2026-10-10. Original18/100/44 remains OPEN. This is not an accuracy search.

## Unanswered question and overlap inventory

HM04/05 used event-local semantic role programs. HM10B3 patched whole owner
spans. HM116 used a source factorial actor difference with design-supplied
signs; HM117/119 constructed focus interactions. None identified a binding
subspace by differentiating a neutral factual retrieval function. HM127
repeated earlier82/109 observability ideas; repeating a null witness is excluded.

The question here is whether a neutral entity--attribute binding operation
also routes behavioural evidence into person-specific trait judgements.
This separates a reusable attribution stage from target-specific interpretation.
It does not assume that either judgement is a true human trait score.

## Source construction before behavioural observations

Eight balanced public authored worlds contain two names and two countries.
Both names and country assignments are crossed. Query each person's country.
Use the already frozen local Qwen observer, eager float32, decoder block16 only.
Average slot differences Delta_E and Delta_A over source contexts. At the fixed
block, move both slot pairs towards their estimated midpoints and inject +x/-x
at name slots, +y/-y at country slots. Record, rather than assume, F(0,0)=.5.

F(x,y) is the mean conditional probability of the factual answers. Compute
the mixed derivative H=d_x d_y F at zero by automatic differentiation. Model
weights never receive updates. A fixed twelve-column Gaussian range sketch
and H^T Q products construct an approximate rank-four SVD; no rank selection.
This identifies only a local, one-block, source-dependent operator.

## Frozen intervention and new materials

For two person-slot activations z0,z1, use the source entity basis U:

    delta = U U^T (z1-z0)
    z0' = z0+delta; z1' = z1-delta

Four fixed arms: baseline, binding projection, a rank-four random projection
with matched edit norm, and full-slot swap. The latter also swaps content and
is a diagnostic, not a pure binding control. No application target/family label
or answer enters the intervention. Slot addresses come from controlled text
construction, not autonomous parsing of unknown natural text.

Four held-out factual worlds contain three new people/countries, with the third
person untouched. Twenty-four behavioural worlds contain six paired behaviour
families, two surface forms and both assignments to the first two people. A
third person has a constant neutral action. Query all three people for all six
fixed targets. Retain all candidate mass, entropy, signs and failed cells.

## Discriminating outcomes

1. Factual swaps and neutral-person preservation assess instrument capability.
   If these fail, a behavioural null is inconclusive about general binding.
2. Behavioural changes towards the natural assignment-swapped judgement,
   beyond matched random edits, support reuse of the attribution operation.
3. A mostly common/third-person effect or target-specific failure instead
   favors a context-dependent joint interpreter at this site. It does not
   reject binding at other sites or in other models.

The main output is the signed person-by-target causal response, not AUC tuning.
Keep per-event responses for GRAIN's qualified author ledger; no extra pooled
score or new psychological factor names. Existing external correlations are
not reassigned to this new operator.

## Mathematics and prior art

Feng/Russell/Steinhardt's mixed-Hessian identification is an existing method:
https://arxiv.org/html/2406.19501v1 sections4.1--4.2 and appendixC.
This packet restricts it to one block and a randomized approximation, rather
than replicating their all-layer13B experiment. Halko/Martinsson/Tropp's
randomized range finder is also established: https://arxiv.org/abs/0909.4061.
Dai/Heinzerling/Inui's cell-based relational binding motivates retaining
relation/context indices: https://aclanthology.org/2026.acl-long.2194/.
OpenAI/math's pinned selected-jets argument motivates complementary derivative
observations, not a transported PDE theorem or a claim of mathematical novelty:
https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex

Potential contribution: factual binding to conditional personality-reading
operation and its limits, demonstrated on identical numerical coordinates.
It is not the first discovery of binding, Hessians or low-rank geometry.

## Execution boundary

Canonical results/grain_math_response_v1/binding_router_r1. Prepare, run,
analyze; immutable inputs and outputs, atomic mutable status, fixed1200-second
cumulative cap and130 forwards including failures/load. No terminal restart,
silent device fallback, layer/rank/strength search or budget renewal. Public
authored data only; no API/private/train/Claude/Git/automation/lockbox/old queue.
