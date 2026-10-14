# HM101: history writing through direct and relay connections

2026-10-10. Goal ACTIVE; previous discussion-only turn was no progress.
The next decision is whether an observation law can address an action span
without carrying the intervening context that changes its vector.

## Question And Prior Inventory

HM92/HM100 hold the current tokens fixed and account for carried value,
key selection and subject law. NEXT45/49 propagate local GRAIN tangents;
HM44 partitions text/query Jacobian routes. HM89/90 compute boundary laws
from an already captured attention graph. None of those experiments performs
the following complete, body-only prefix edge interventions. This is a new
intervention at a known mathematical interface, not a newly invented theorem.

## Design Frozen Before Outcome

Two public action banks: HM93 workshop source wording and HM98 new everyday
wording. All six action families, three nominal states per family, all 3x3
past/current pairs and two neutral bridge wordings are retained. This produces
216 new token streams, 864 intervention forwards and two no-op-control forwards,
not human participants.
The source query/direction bundle is inherited, without new fitting.

Partition each token stream into H (the past record), B (all other prefix
tokens including the neutral bridge and current-record header), and C (the
current action). H is right-padded to one fixed width with ignored keys; all
current positions, token IDs and B tokens are identical across the three
past states. This deliberate token-space construction is not unmodified
natural prose. Padding queries cannot transmit because their keys are blocked.

At every completed block 0--16, retain the original causal mask and:

| arm | H -> C | H -> B |
| --- | --- | --- |
| neither | blocked | blocked |
| direct_only | allowed | blocked |
| relay_only | blocked | allowed |
| both | allowed | allowed |

B -> C and C -> C remain allowed. The relay-only arm therefore permits
H -> B -> C across successive layers. Within-token normalization/MLP and all
weights are unchanged; only pre-softmax permissions change. These interventions
also change normalization, so a one-arm response is not the same as subtracting
a saved attention contribution. No head/layer is selected by outcome.

Capture only block17 current K/V before attention/output/MLP/answer. Evaluate
uniform-value and the frozen source-query local observable for all six targets.
Keep current-vector differences as well as target-projected differences.

## Mathematical Account

For any fixed observable, let F_dr be the score with the two routes enabled
by d,r in {0,1}. Finite Mobius inversion gives exactly

    F11 - F00 = (F10 - F00) + (F01 - F00)
                    + (F11 - F10 - F01 + F00).

This is a signed intervention account, not a unique path attribution or
psychological information fraction. Define two positive toy author laws

    P+/- = (1 +/- .5 D)/9,
    D = [[0,1,-1],[-1,0,1],[1,-1,0]].

They have identical event occupancy and symmetric traffic. Their contrast
is <D,F>/9. Apply the same Mobius law to that author contrast, retaining
all families, targets, bridges and both banks. No accuracy or correlation
threshold chooses a route. A nonzero relay response changes the event-unit
interface; a material interaction rejects additive route assembly.

## Checks And Stop Rules

- Token identity/position and complete partitions must hold before calls.
- Each capture must complete exactly blocks0--16; no answer or upper block.
- The finite causal dependency graph with both routes blocked has no H -> C
  path. Actual current K/V should then be invariant across past states within
  fixed float32 tolerance 1e-4; this control tolerance is not a success target.
- No-op hook has the same causal mask as the stock prefix on an unpadded
  public control. Its current K/V are checked at the same float32 tolerance.
- Mobius reconstruction, positive author laws and independent cycle sums
  are checked algebraically; all signed/negative/null results are retained.
- A failure stops interpretation; do not tune until a favored mechanism wins.
- Resumable per-unit outputs, input/code/source seals, bounded 300 model
  seconds and one writer lock; never overwrite earlier or partial packets.

This does not train a personality predictor, establish human psychological
validity, remove context as noise, or close original18/100/44 requirements.
No API, private data, new neural training, Git or old queue is authorized.

## Literature And Inspiration

Elhage et al.2021's transformer-circuit framework already analyzes composition
of attention heads and virtual connections. Nonlinear path intervention and
finite factorial/Mobius accounts are established tools. OpenAI/math pinned
adc7f1241b42e322a6451854ab7e4b4c146bf78a, 03b-isotypic.tex lines106--194,
motivates recording an operator's placement/support before composing it.
Its signed-density theorem is not assumed for a transformer. Our finite
no-transmission proof is instead an induction over the actual masked DAG.

The possible GRAIN contribution is the empirical route dependence of the
same-author evidence observation, and the resulting rule for its construction.
