# HM74: encoded ownership sectors versus recomputed observation

2026-10-09. Finite design fixed before model acquisition. Codex executes locally.
This is a mechanism experiment, not an accuracy improvement or new human scale.
Original 18 protocols / 100 cards / 44 computational questions remain open.

## Decision changed by this experiment

HM73 found content-matched ownership actions, but descriptive hidden-state
interactions do not establish their causal participation. HM51/52 already showed
that strong injected vectors can also move judgments without relevant evidence.
Do not optimize their strength. Ask instead whether subtracting the explicit
behavior-by-ownership component of the complete contextualized query changes the
corresponding observation, keeping other components and the body intact.

Two alternatives: an explicit stored sector participates in the observation;
or upper layers can reconstruct the action from remaining body/main-effect
states. Mixed outcomes across targets and contexts are retained. No absence
of an ablation effect proves that ownership is irrelevant or that recomputation
is the uniquely identified alternative; redundant routes remain possible.

## Material and fixed scope

One new manually written public synthetic wording bank, all six targets:
extraversion, neuroticism, agreeableness, conscientiousness, openness,
anthropomorphism. No model-generated text, API, questionnaire or private data.
Thirty directed focal/companion pairs, eight x/y/g corners. Each pair has exactly
two predeclared queries: focal and companion. Six focal/neutral pairs have only
the focal query. Therefore 66 query cubes / 528 native captures / 288 bodies.
The four other query cross-cells per pair are not measured here; HM73 retained
the full six-query descriptive scope. Do not call two-query selectivity global.

x changes an always-A focal behavior. y changes a companion behavior. g assigns
the companion to A versus B while assigning a neutral anchor to the opposite
person. Events K1/K2/K3 retain order. The g edit swaps two actual name tokens with
the same token bag; it is a joint ownership/context edit, not a pure instrument.
Within each cube, complete query suffix token identities must match in order.
Natural body lengths and absolute positions are retained, not interpolated.

Frozen Qwen2.5-1.5B, eager float32 MPS, zero-based block16 output, full input,
unchanged HM68 system/queries/ten digit candidates. No head/layer/rank/strength
search. Every native capture precedes every patch. Native states and all numeric
payloads are sealed before reading patched outputs. Coefficients use the whole
designed cube; this is a counterfactual experiment, not a standalone new-author
reader or a source-only portable learned operator.

## Exact finite mathematics

For z=(x,y,g) in {-1,+1}^3 and complete suffix state S(z),

    c_J = (1/8) sum_z chi_J(z) S(z)
    S(z) = sum_J chi_J(z) c_J

Character order is constant,g,y,yg,x,xg,xy,xyg. Seven fixed arms per corner:

1. remove_yg: S - yg*c_yg.
2. remove_xyg: S - xyg*c_xyg.
3. remove_both: S - yg*c_yg - xyg*c_xyg.
4. remove_xy: S - xy*c_xy (ordinary content interaction control).
5. random_both: subtract a cube-wise random suffix-token direction with the
   same per-token norm as the joint subtraction and the same yg parity.
   Directions are fixed by seed and cube, independent of all output scores.
6. exact_owner: replace suffix with the g-flipped donor (same x/y/query).
7. noop: replace suffix with itself.

The constant, all main effects and untargeted coefficients are retained in the
assigned states. No hard deletion of contextual means or author choice.
After the block16 assignment, all upper layers freely recompute. The body is
not frozen to donor states or erased. These are propagating interventions, not
path patching or storage-only binding-ID experiments. Altered states may be off
the natural manifold. Controls qualify interpretation, not remove that fact.

Y(z) denotes each arm's actual candidate logits, score, and high/low log odds.
Use both the marginal 4*c_yg and conditional 4*(c_yg+x*c_xyg) ownership actions.
Keep focal main effects, all eight output coefficients and signed changes.
Joint ablation interaction:

    I(z) = Y_native - Y_remove_yg - Y_remove_xyg + Y_remove_both

The joint loss equals individual losses minus I. Deleting a coefficient in
the input state does not algebraically require deleting it in a nonlinear
readout. Synthetic linear-storage and nonlinear-recomputation worlds must
demonstrate this before any model run.

Connect to four fixed synthetic author compositions, not human identity:

    P_{J,+/-}(z) = (1 +/- 0.5*chi_J(z))/8, J in {yg,xyg}
    Theta_{J,+/-} = sum_z P_{J,+/-}(z) Y(z)
    Theta_{J,+} - Theta_{J,-} = c_J

This compares hypothetical evidence distributions and links the intervention
to a measurement consequence. For xyg profiles all first/pairwise marginals
match. For yg profiles only first-order marginals match. Do not equate observed
composition with opportunity-corrected preference or common reader with B_u.

## Budget, outputs and interpretation

528 native + 3696 patch = 4224 actual attempts maximum; 2400 seconds maximum
collector wall time including I/O/model loading. No uncertain-call retry.
Pre/post token arrays or sufficient exact-array records, logits, actual patch
suffixes, inputs, code, payload and output hashes are retained. Save checkpoint
before each attempt. Any error leaves STOPPED and all completed units intact.
Do not restart solely on a timeout or stale status observation.

Output root: results/grain_math_response_v1/causal_ownership_sectors_r1.
Inputs/code in 00; native/patch captures and frozen payloads in 01/02;
coefficients, actions, paired-query contrasts, joint-ablation cells, composition
consequences and summaries in 03. Numerically replay from captures with another
formula; no new neural calls for verification. Same-agent arithmetic verification
is not third-party semantic replication. Tests do not prove a psychological claim.

Interpret effects with their sign, target and context. A change specific to the
companion query relative to the focal query, unlike the xy/random/neutral controls,
supports functional participation in this finite design. Generic disruption
supports a less selective account. An unchanged action despite exact removal
directs the next question toward recomputation/redundancy, not a fidelity grid.
No adjusted prompt, stronger patch or extra layer search to rescue a failure.

## Prior art and mathematical inspiration

[Heimersheim and Nanda](https://arxiv.org/html/2404.15255v1), sections 1--2,
distinguish necessity/sufficiency and propagating versus path interventions.
[Feng and Steinhardt](https://arxiv.org/html/2310.17191v1), sections 2--4,
test entity--attribute binding and freeze other context states in storage tests;
our experiment is deliberately a different propagation estimand.
Boolean character projection is standard finite harmonic analysis, not ours.
OpenAI/math pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a, the boundary
metric/connection manuscript section 06-matrix, motivates fixed observation
families and preserving joint information. Its PDE/harmonic-jet assumptions are
not established for LLMs; no uniqueness/reconstruction theorem is imported.
Candidate contribution is the event-qualified evidence-to-target operation and
its measurement-composition consequence, not activation patching itself.
