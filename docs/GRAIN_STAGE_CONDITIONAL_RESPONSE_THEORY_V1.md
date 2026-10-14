# GRAIN stage-conditioned response, not an independent-axis dictionary

2026-10-10. Interpretation of HM123, not a replacement of its sealed plan.
Canonical child: results/grain_math_response_v1/stage_mass_action_r1/shape_adapter_r1.
HM122 target-selectivity failure and the HM123 primary input-guard failure remain.

## 1. Explicit measurement maps

For a qualified history H, focus f, cue q and reader r, let m0 be its baseline
history measure and mq,s its frozen positive/negative GRAIN tilt (s=-1,+1).
Only the access measure changes. Historical K/V, token positions, current tokens
and all pretrained maps remain fixed. Write

    F_r(a,b;H,f) = R_r(Psi_late[m_b](Psi_early[m_a](seed_f;H);H)).

Here a,b in {0,1} choose baseline or the specified tilted measure on each stage.
Psi_early contains blocks0--7 and Psi_late blocks8--16 plus current projection.
History states already encode context. This is not a claim about an isolated
natural psychological pathway or an author's longitudinal early/late stages.

The inherited R_r is a vector instrument: uniform value projection, or a
query-key softmax weighted value projection. It is not an LLM-generated trait
answer. Frozen GRAIN4 coordinates are anonymous observation marks, not six
psychological control axes.

## 2. Exact finite composition

    E = F(1,0)-F(0,0)
    L = F(0,1)-F(0,0)
    I = F(1,1)-F(1,0)-F(0,1)+F(0,0)
    T = E+L+I.

All sign, behavior, owner and focus contrasts are linear operations, so the
identity survives those contrasts. The effect of early access given altered
late access is E+I, not E; the analogous late effect is L+I.

For any C2 two-parameter supported-measure extension, the fundamental theorem
of calculus gives I = integral_0^1 integral_0^1 d_a d_b F(a,b) da db.
This is standard mixed-derivative mathematics. We measured the finite corners,
not the Hessian, a unique intrinsic interpolation, or a natural indirect effect.

HM123's new-instance weighted owner-odd I/T RMS ratios are .217736/.222090;
E/L cosines .352064/.453747. Additive stage actions do not describe the complete
observed response. The operational .1 trigger is not psychological significance.

## 3. Nonadditivity is not gain or independent control

Define alpha_I=<I,T>/<T,T> and I_perp=I-alpha_I T in the fixed source-scaled units.
RMS(I)/RMS(T) measures magnitude, not the percentage of variance or prediction
accuracy contributed. Here alpha_I=.039359/-.003762 while the result-after
RMS(I_perp)/RMS(T)=.214149/.222058. Most interaction is transverse configuration,
not a uniform amplification. Do not explain the full selectivity failure as
simple stage cancellation.

Neither stage passes both-wording correct>0 and correct-minus-misaligned>0.
Late wording1's gain+.035555 does not pass: its correct selectivity is-.083331.
Report both stages, all six inputs/outputs, roles, conditions and reading modes.

If two score functions are exactly equal on the same support, their tilted
measures are identical. Any fixed early or late schedule then produces identical
responses. Opposite score functions swap the signed tilts, giving opposite
signed response contrasts. No downstream nonlinear calculation can separate
literally identical admitted interventions without an additional input.

Actual cues are approximate, not exact equivalents. Their stage responses still
couple strongly: E/A early/late=-.984564/-.973939 and O/anthro=.940156/.979225.
These instrument correlations do not prove shared neuronal paths or human
trait correlations. A finite nonlinear response rank is not bounded by the
four linear mark dimensions; do not invent a dimensional impossibility claim.

## 4. Author measurement must retain its distinct objects

For author evidence law P_u and specified instrument policy nu,

    S_u,r^nu = sum_e P_u(e) sum_f nu(f|e,r) F_r(0,0;H_e,f).

The staged action tensor describes how this instrument changes under supplied
access interventions; it is not itself the author's trait or response parameter.
If policy/focus is chosen by the author, keep it in the evidence law. If supplied
by the instrument, type and standardize it as an instrument choice. Preserve
condition selection, state and all off-diagonal responses; do not residualize
away these objects merely because they interact.

The next construction should identify additional qualified relation information
and its conditional response, not increase the fidelity of the existing cue
dictionary. Existing HM105/109/112/119 sources must be inventoried before a new
packet; a new mechanism needs a new distinguishable scientific prediction.

## 5. Established theory and actual contribution

- Vaidyanathan et al.(2026), Sections3--4, discusses interaction in activation
  interventions: https://arxiv.org/html/2606.27510v1 . Our defined mass switches
  are not automatically their PIE/NIE estimands.
- Geiger et al.(2025) formalizes causal abstraction and intervention:
  https://jmlr.org/papers/v26/23-0058.html . Factorial/Mobius algebra is mature.
- OpenAI/math fixed commit adc7f1241b42e322a6451854ab7e4b4c146bf78a,
  matrix-jets71--108 motivates complementary admissible observations:
  https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex .
  Its PDE assumptions and spanning theorem do not establish a text theorem.

The contribution here is a finite staged test and a resulting constraint on the
GRAIN-to-vector measurement interface, not a new theorem or accuracy breakthrough.
All288 public histories/28224 explicit pretrained current-state evaluations were
completed; original failure+adapter254.348907sec within the original600sec budget.
Numeric baseline/account errors0; offline full-cell audit passes separately.
No new model capture, answer generation, fit, gradients, private data/API/training,
Claude/Git/automation/lockbox or old queue. Original18/100/44 remains OPEN.
