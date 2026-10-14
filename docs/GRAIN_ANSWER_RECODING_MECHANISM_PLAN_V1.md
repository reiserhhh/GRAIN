# HM53: trait-direction versus answer-code operation

2026-10-09. Execution implementation prepared; no model calls or result at this
pre-acquisition revision. Decision follows verified HM52, not an accuracy optimization.
This plan is frozen with the implementation before acquisition; subsequent
status and findings belong in the result/report, not edits to this source plan.

## Scientific question

Does the source operation transport a concept-oriented change, a raw digit
direction, a code-conditioned judgement, or a mixture? HM52 control effects
make this more valuable than decreasing teacher-approximation error.
This experiment cannot alone distinguish conceptual evidence injection from
a genuine reusable reading procedure; keep that next logical question open.

## Fixed design and source barrier

1. Preserve all frozen HM46--52 code, materials and outputs. Use the same
   local frozen observer and block16, system, six target definitions, actors,
   bodies and ten single-digit candidates. Do not generate or train.
2. Reuse24 standard-code source captures from verified HM50/HM51. New inverse
   source captures24: six targets x two actors x two directions. In the exact
   person query change only `Use 0 for very low, 9 for very high` to
   `Use 9 for very low, 0 for very high`; keep the final allowed alphabet0..9.
   This is a controlled instruction change, not a result-tuned prompt.
3. Before any forward freeze all material strings, actual input IDs, token
   offsets, actor/source IDs and paired address maps. Token counts and complete
   suffix lengths must agree across codes; every differing suffix token must
   be one of the two declared digit-cue tokens. Stop on mismatch, do not edit
   materials after native answers are observed.
4. Acquire and seal24 inverse source captures before new application calls.
   Construct per-target/person/code ownership0-to1 full-suffix source deltas.
   Source vectors and norms are sealed before loading application outputs.
   No averaging over target scores, source/app gradients or fit.
5. Application uses HM52 own and roster texts, six targets/two actors/two
   directions under both codes:96 receivers. Reuse48 verified standard-code
   captures; acquire48 inverse-code captures. They are controlled recodings
   of known material, not a new behaviour family or independent authors.
6. All96 receivers get six fixed arms: exact same-code matched suffix donor;
   forward-code source delta; inverse-code source delta; wrong-person source
   delta under receiver code; token-norm-matched random under receiver code;
   noop. Source and receiver code cross fully. Add source delta with the
   original1-2*direction sign. No strength rescaling or model selection.
7. Same-code maps require identical suffix IDs. Cross-code maps require
   identical ordered slots except the declared two digit cues. Save complete
   pre/post states and all unaffected positions. This is slot-controlled
   code-changing intervention, not assumed neural group equivariance.

## Calls and safe stopping

24 new source +48 new application capture +576 patch =648 maximum new forwards;
72 original captures reused and verified, not counted as fresh replication.
600 cumulative model seconds including load/failure/IO. Commit attempts before
each call; unknown in-flight/terminal failure cannot automatically retry.
Source barrier precedes application; preserve outputs on budget/error stop.
0 API/private/generation/training or old queue, no new Claude/automation.

## Outputs and exact equations

Store p_k, candidate logits, candidate mass and full-vocabulary normalizer,
raw conditional score and L_raw=logsumexp(logits5..9)-logsumexp(logits0..4).
kappa=+1 standard/-1 reversed; y_sem=(1-kappa)/2+kappa*y_raw;
L_sem=kappa*L_raw. Keep all raw and decoded quantities.

For a fixed source code s and material/person/direction, report both receiver
code actions. With eta=(1-2*direction)*(-1 forA,+1 forB):

    R_{s,kappa} = eta*(L_raw_patched - L_raw_native)
    M_{s,kappa} = kappa*R_{s,kappa}
    R_even=(R_{s,+}+R_{s,-})/2
    R_odd =(R_{s,+}-R_{s,-})/2

These are finite C2 character contrasts at the output coordinate, not assumed
hidden-state isotypes. Raw-even suggests a raw-direction component; raw-odd
is compatible with concept-direction action if native code use is correct.
Matching/opposite source-code arms identify whether source operation itself
depends on format. Interaction with own versus roster remains a separate axis.
Do not call even/odd terms psychological factors or information percentages.

Report native direction/recoding compliance for every target and actor,
same-code donor control, all raw/decoded intervention actions and full factorial
contrasts. If inverse native reading fails, retain it and mark the operation
contrast ambiguous for that cell; do not drop it from output or rescue prompt.
No score-improvement gate or p-values over dependent constructed observations.

## Deliverables and acceptance

New module, runner, verifier, configuration, tests; result directory
results/grain_math_response_v1/answer_recoding_mechanism_r1, without overwriting
HM52. Save48 total source captures by verified reuse/new provenance,96 native
receiver rows,576 effect rows, source/receiver-code x evidence-class x target x
person summaries, even/odd contrasts and report. All96 inputs/648 attempts
and unexpected/failed calls must be accounted for.

Minimal tests: correct/inverted unit identities; exact log-odds sign change;
raw-digit steering and semantic-steering synthetic counterexamples; mixed
case; source-code and actor pairing; only two cue tokens may differ; source
barrier/no application truth in vectors; zero/noop/cap/uncertain stopping.
Alternate verification uses no new neural calls and reconstructs every source
delta, patch, raw/decoded score and contrast. Arithmetic1e-10, prepatch/noop1e-6,
random norm relative1e-5 are implementation tolerances, not psychology thresholds.
The alternate norm formula may change float32 patch rounding by one ULP;
assignment tolerance1e-6 is fixed before calls. All unselected positions must
remain byte-identical. Source seal records48 capture rows/24 delta operators,
actual completed-source count and timestamps before48 application captures.

## Next decisions, not automatic success

Concept-compatible action: test whether it carries injected source evidence
or actually changes the mapping applied to new evidence. Raw-code action:
assign it to observation-unit/output control, not author trait estimation.
Code-conditioned action: transport the observation rule with the declared
code, rather than assume one universal vector. Mixed action: keep the joint
code/context law; do not tune alpha until the mixture disappears.

Connect qualified evidence and GRAIN author aggregation only at those actual
units. This closes a mechanism question, not the entire18/100/44 scope,
natural-author psychological links or independent mathematical personality
inference. Goal remains ACTIVE; HM45 approximation queue stays deferred.

## Reading basis

Function Vectors AppendixA differentiates procedures and output offsets:
https://arxiv.org/html/2310.15213v1.
Causality != Invariance (ICLR2026):
https://proceedings.iclr.cc/paper_files/paper/2026/hash/86b3697c4eb7792c951831636bfdacd5-Abstract-Conference.html.
OpenAI/math03b-isotypic motivates keeping sectors/domains and assumptions
explicit, but its signed-density/angle bounds are not imported. Exact readout
recoding and finite character decomposition are standard mathematics.
