# HM115: behavior expression, actor assignment and shared action directions

2026-10-10. Frozen finite design before new model calls; not a predictor search.
Parent goal and original18 protocols/100cards/44 computational questions stay open.

## Scientific decision and inventory

HM114 retained behavior clauses while changing request wording. HM105 changed
behavior clauses and the carrier together; raw transport was weak and visible
response transport depended on the carrier. Neither isolates lexical reuse from
behavior-instance correspondence. M01/M04 already have actor/relational evidence;
Feng et al. already establish generic binding, so this is not its rediscovery.

Compare three predeclared behavior supports: the HM114 behavior clauses, a
rewording of the same event, and a different event intended to express a related
behavior tendency. The last correspondence is a design hypothesis, not a
psychological truth or proof that both events measure the same Big5 trait.
No private texts, human labels or external scales are used.

Both A and B receive the SAME opportunity condition. HM114 assigned opposite
conditions; its owner contrast therefore also changed the actor's opportunity.
This new matched design removes that specific ambiguity. It is not a rewrite
or direct same-number replication of HM114. The current A carrier and its token
positions are fixed across all variants. Owner swaps preserve actual token bags
at each condition/polarity; condition swaps need not preserve token bags.

Three expression variants x two request wordings x six fixed families x
two conditions x two polarities x two owners =288 actual local prefixes.
Frozen Qwen2.5-1.5B layer17 pre-attention K/V, all0--16 causal paths open,
padding excluded. No new suffix, logits, generation, training or query selection.
All old six query/covectors are reused unchanged. Budget120 model seconds;
pending failures or cap crossing are retained, never restarted with a new budget.

## Fixed mathematical objects

For native K, native V, token-mean V, uniform readout and query-weighted readout:

    F_f(c) = [h_f(c,+,own)-h_f(c,-,own)]/2
    O_f(c) = [h_f(c,+,other)-h_f(c,-,other)]/2
    S_f(c) = [F_f(c)+O_f(c)]/2
    D_f(c) = [F_f(c)-O_f(c)]/2
    F=S+D, O=S-D.

S is the part even under actor assignment; D is the odd interaction between
behavior polarity and actor assignment. Neither is automatically a semantic or
psychological factor. Nonzero D is an actual matched text contrast, not a unique
internal binding mechanism; name POSITION and recorder roles also move.
Keep all eight factorial characters, including conditional interactions.

Predeclared comparisons: same behavior/two opportunity wordings; original vs
reworded at each wording; original vs new instance at each wording. All own,
even and odd fields, all five observations, all six families and two conditions
are reported. Compare identity matching with all720 family permutations; no
rotation, whitening, fitted metric, rank selection, champion family or p rescue.
Zero directions return unavailable, not zero cosine. Permutations describe this
six-family finite design, not human-population significance or personality r.

Also compute a role-blind token-count difference from behavior clauses alone.
This is a lexical control, not a learned predictor. The vocabulary is merely the
union of frozen design tokens. It cannot supply an odd ownership component.
Equal-condition reference author contrasts use declared amplitude .4 and family
mass1/6. They are exact numeric integrals, not new humans or measurement accuracy.

## Reading and decision boundaries

If rewording corresponds but new instances do not, describe the narrower
behavior-support limit rather than tuning it into a broad personality direction.
If S transports better than D, common content does not establish portable author
attribution. If target readouts correspond more than native states, describe a
reader-dependent stable observation, not a universally stable raw vector axis.
Even a weak/negative result changes which object a future GRAIN measure should
carry. Do not discard condition origins, choice, state or vector complements.

Numerical acceptance means all288 authenticated prefixes, fixed token positions,
matched owner bags, full signed decomposition, actual readout replay, explicit
author-integral replay and all output hashes checked. It does NOT require a
positive scientific result. No local inverse/precision grid is added.

## Literature and transfer of ideas

- Park, Choe & Veitch, ICML2024: https://arxiv.org/abs/2311.03658
  distinguishes probing/steering and a causal inner product. Native Euclidean
  angle here is descriptive; we do not satisfy their causal assumptions or call
  the existing Qwen covectors a recovered causal metric. Official code:
  https://github.com/KihoPark/linear_rep_geometry .
- Feng, Russell & Steinhardt,2024/ICLR2025:
  https://arxiv.org/html/2406.19501v1 sections3--4.2. Lexical content and binding
  must be separated; their Hessian/intervention identification is not performed
  here and cannot be claimed from D alone.
- OpenAI/math pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a:
  preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex lines71--109.
  Fixed complementary observations/range checks inspire the design. Harmonic
  jets, PDE and smooth-bundle assumptions are not transferred to language.
- Symmetric/antisymmetric character decomposition and conditional expectations
  are standard mathematics. New evidence, not new notation, is the contribution.

## Execution

Use a fresh results/grain_math_response_v1/action_expression_r1 directory:
run_grain_action_expression_hm115.py prepare, collect, analyze, verify.
The analyze CLI loads only NumPy/frozen numeric captures, no model runtime;
upstream encoding still depends on pretrained neural weights. Failures remain.
