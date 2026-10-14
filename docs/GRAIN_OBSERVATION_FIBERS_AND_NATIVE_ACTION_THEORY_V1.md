# Observation fibers, actor relations and native vector action

## 1. Scientific object

HM124 asks whether a frozen GRAIN observation retains a relation that a native
model judgment uses. This is not a prediction-accuracy improvement experiment.
The observation is the inherited four-coordinate lexical operator, either on
the whole text or on three lossless slices. It is not every GRAIN representation.
The native judgment is a frozen Qwen digit-logit instrument, not human ground
truth. Its four units remain separate.

Two texts can differ in who performed a behavior while sharing the same words.
Matching content or lexical coordinates alone need not preserve that relation.
The required question is not whether a stronger downstream fit can interpolate
more labels, but whether the observation can distinguish the required inputs.

## 2. Exact observation fibers

Let a text be x, its admitted observation be G(x), and a fixed-query native
readout be R_q(x). Define the observation fiber

    [x]_G = {x': G(x') = G(x)}.

There exists a deterministic readout f_q with R_q(x)=f_q(G(x)) on a specified
domain if and only if R_q is constant on every fiber on that domain.

Necessity follows because identical inputs to f_q have identical outputs.
For sufficiency, define f_q(g) to be the common R_q value on G^{-1}(g).
No linearity, continuity, particular optimizer or factor rotation is required.
This is elementary factorization through an observation, not a new theorem.
It concerns the specified R_q, not every conceivable psychological target.

On a finite ledger with uniform weights, the minimum squared reproduction loss
over ALL functions of G is

    min_f E_ledger[(R_q-f(G))^2]
      = E_ledger[Var_ledger(R_q | G)].

The minimizing value on each class is its within-class mean. For a class of two
rows with readings r_1 and r_2, its minimum MSE is (r_1-r_2)^2/4. This is a finite
information-loss certificate, not a generalization estimate or a clinical
error floor. If all observations are distinct, this certificate is zero and
says nothing about prediction on new texts.

The HM124 supplement uses exact floating-point coordinate equality, not a
tolerance selected to make collision classes. It starts after source actor
invariance was found and is explicitly supplemental. It never selects edits
or models using the native application responses.

## 3. Actor exchange and sample-space projection

Let T exchange the two actor rows in each matched source pair. T is an
orthogonal involution, T^2=I. The even and odd sample-space components are

    Q_+ = (I+T)/2,  Q_- = (I-T)/2.

Let A=[1,G] be the source affine-design matrix and P=A A^+ its orthogonal
sample-space projector. If TA=A, then TP=P and Q_-P=0. Thus, for any matrix X
of contextual vector values,

    X_G = P X,  X_R = (I-P)X,
    Q_- X_G = 0,  Q_- X = Q_- X_R.

This follows from the invariance of range(A) under actor exchange. All
actor-odd source variation must lie in the remainder in this setup. A near-zero
represented actor action is therefore a consequence of the observed invariant
design, not a new empirical discovery of a unique psychological subspace.

The group calculation is the two-character decomposition of an involution.
It does not justify transferring a physical group symmetry to human traits.
P operates on source rows, NOT neural-feature directions. X_R is orthogonal to
A on the source ledger; it is not independent, pure noise, or a personality
coordinate. A general nonlinear GRAIN readout also cannot distinguish exact
same-G actor pairs, by Section 2.

## 4. From missing observation to an actual native action

A nonzero contextual remainder does not by itself show that the model uses it.
HM124 takes the original source actor-by-behavior contrasts in X_G, X_R and X,
without application answers, and applies each at the fixed current-token V
site to new behavioral instances. Keys, positions, padding and text stay fixed.
The full model then answers twelve actual target/person questions per batch.
A norm-matched random action is retained as a finite control.

The full 6-by-6 response matrix is kept, including off-target changes. Effects
are fixed finite contrasts, not uniquely attributable natural causal fractions.
Source construction uses control-design family, polarity and actor labels to
select application edits. These labels are NOT inferred from unknown text.
Thus this experiment cannot deliver an autonomous personality scorer.

Because Q_-X_G=0 in this source design, whole and remainder actions differ only
by numerical rounding. Their native arms are still recorded as registered;
they are not two independent replications or evidence for additivity.
Once this degeneracy is known, repeated whole-versus-remainder experiments
have little additional scientific value. No dose/layer/rank grid is warranted.

## 5. Implication for the GRAIN construction interface

An author measurement must retain evidence qualification, not just local
lexical position. A candidate observation object is

    O_e = (content state, actor relation, condition/opportunity,
           time/event identity, reporting modality, observation uncertainty).

These entries cannot be silently collapsed into one averaged text score.
Their contextual vector implementation is a research task; naming entries is
not its completion. Existing event/binding work already provides part of this
interface and must be reused rather than rediscovered.

The decision from HM124 is scoped: use its fiber test to reject an insufficient
observation before fitting a stronger predictor. Next development must preserve
qualified relations and organize them across authors and conditions. It must
not replace the personality-interpreter goal with a generic role classifier.
Choice composition, conditional response and state remain separate retained
objects. Their psychological interpretation needs its own evidence.

## 6. Prior art and the actual mathematical contribution

- [Amnesic Probing](https://aclanthology.org/2021.tacl-1.10/) separates encoded
  information from information used by model behavior. HM124 follows this
  scientific distinction, but does not implement INLP concept erasure.
- [LEACE](https://arxiv.org/abs/2306.03819) has specified linear-erasure
  guarantees. Our sample-space residual has no such guarantee.
- [Park et al.](https://proceedings.mlr.press/v235/park24c.html) distinguishes
  context/word representations, measurement and steering. We do not estimate
  their causal inner product or claim an identified independent concept axis.
- The pinned [OpenAI matrix-jets manuscript section](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex)
  motivates testing whether selected observations cover complementary useful
  directions. Its harmonic-PDE, constant-rank and bounded-inverse assumptions
  are not established for GRAIN or transferred by analogy.

Projection, involution decomposition, fiber factorization and conditional
variance are established mathematics. The finite empirical contribution is
the explicit interface test between this frozen GRAIN operator and native
model use on controlled new expressions. It is not a new mathematical theorem,
a completed text-personality theory, or proof of human personality validity.

## 7. Reproduction and evidence

Verified finite HM124 results (2026-10-10):

- All48 fresh texts have exactly24 distinct whole observations and24 distinct
  three-slice observations. Every actor pair has coordinate difference zero.
- Native actor-swap RMS is .221427661 in conditional digit score and
  1.430652773 in grouped digit log-odds. The finite best-function reproduction
  RMSE floors are .110713831 and .715326386, respectively, for either G view.
- Source represented actor-action norm is6.5605e-14; remainder/whole5.617819433.
  This near-zero represented part is forced by the invariant source design.
- Remainder native-effect/baseline norm is .002677918/.003258966 for score and
  .002045429/.002608593 for log-odds in the two paraphrases. Its RMS is larger
  than the one frozen random control, but the absolute native action is small.
  Off-target effects are retained; this is not a selective personality axis.
- All240 native batches/2880 endpoints completed in642.080135 model seconds
  under the original900-second cap. Twelve focused tests passed. Alternate
  source/logit/field readback max error4.84e-14; exact-fiber readback6.67e-16.
- Supplemental Python Ruff checks pass. The main sealed code retains16 style
  lint findings (EXE002/C408/F841), so there is no full Ruff-pass claim.

These results decide against a stronger predictor on these same observations;
they do not establish a strong transferable native action or a human measure.

The immutable main plan is `docs/GRAIN_NATIVE_REMAINDER_PLAN_HM124.md`.
Canonical outputs are under
`results/grain_math_response_v1/grain_native_remainder_r1`.
The result-after supplement is `collision_supplement_r1` under that root.
The offline readback does not call the model. Source and application are kept
separate; native units can be recomputed from all saved candidate logits.
Do not rerun terminal collection or mutate frozen inputs to improve a result.
The original 18-protocol/100-card/44-question scope remains open.
