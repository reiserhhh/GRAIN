# Controlled-response overlap is not a personality/noise split

HM149, 2026-10-11. Executed finite diagnostic, using mature projection/kernel
mathematics and historical synthetic-text vectors. This changes the next source
design, not the prediction model. Original 18/100/44 scope remains OPEN.

## 1. The connection to the mainline

HM146 distinguishes a person-sensitive carrier from diagnostic meaning.
HM147 supplies a source-assignment-anchored joint function without an LLM answer.
HM148 shows its new-task reversal involves both expression change and location.
HM149 asks whether that change is simply a response shared with instructions
that do not explicitly name personality. It does not fit a better reader.

The proposed HM149 Bayes/conditional-shift simulator was NOT executed: declaring
a likelihood and recovering it would mostly instantiate known mathematics.
Instead this packet compares the existing empirical persona and nonpersona
response geometry at fixed units, using no new model inference or labels.

## 2. Retained finite objects

Source:135 vectors from diary/email/opinion. Cold:135 vectors from advice,
review/neighbor. Each contains four GRAIN4 slices. Persona seeds11/23/37 are
generation replicates, not people. Source and cold each have18 matched
newsletter-minus-neutral/spelling-minus-neutral pairs. The54 cold persona
high/low pairs provide27/target, with18/target in evaluation and9 in development.
This corpus is historical and has been inspected before, not an untouched test.

For the parent's unchanged finite tensor map Phi and target covector w, stack
source control differences as rows of B. Its empirical row span C has rank18
under the fixed1e-10 numerical tolerance. No factor count was discovered.
An orthonormal row basis Q defines P_C. Then

    w = w_C + w_perp;
    w_C = (w Q^T) Q; w_perp = w - w_C;
    f(x) = f_C(x) + f_perp(x).

All three original arms, both targets and both parts remain. No coefficient is
renormalized, no text component deleted and no application result selected.
The cold scorer reads only numeric views/program. Metadata joins afterward.
The control null of w_perp on SOURCE rows is a construction identity, not a
new validation result. It is not guaranteed on another task's control response.

This is a finite secant span, NOT a proven group orbit, universal nuisance
tangent space or stochastic noise law. Newsletter/spelling transformations can
also evoke personality-relevant behavior. Their overlap must not be called
irrelevant merely because the source instruction was nonpersona.

## 3. Actual task-level results

Full-arm source overlap squared-norm fractions: E .185779, N .173034. Those
are fractions of the constructed function's coefficient norm, not personality
information or causal explanation. E/N coefficient inner product is -.079866:
overlap contributes -.023761, complement -.056105. Both are signed, neither
is a human trait correlation. The two parts' normalized cosines cannot be added.

All values below are high-minus-low FUNCTION UNITS, not r, AUC or probabilities:

| Role/target | Original | Overlap | Complement |
| --- | ---: | ---: | ---: |
| Evaluation E,18 pairs | -.017085 | -.020955 | +.003869 |
| Evaluation N,18 pairs | -.113990 | -.097105 | -.016885 |
| Advice E,9 pairs | -.187065 | -.069881 | -.117184 |
| Advice N,9 pairs | -.201795 | -.165882 | -.035913 |
| All application E,27 pairs | -.073745 | -.037263 | -.036482 |
| All application N,27 pairs | -.143258 | -.120031 | -.023227 |

N's pooled negative response is largely in the control-overlap part, but the
complement also reverses. E's tiny positive evaluation complement cannot be
promoted into a successful predictor; its all-application complement is negative.
No percentage of psychological information is inferred from these signed sums.

Local possibilities are retained, not used to select a score:

- E review/anchored: +.595845 = +.090097 +.505748.
- E review/described: +.279368 = -.135832 +.415200.
- E neighbor/described: -.584426 = -.034290 -.550136.
- N review/named: +.555010 = +.053152 +.501857.
- N neighbor/named: -1.102322 = -.672441 -.429881.

Each local cell has three generated pairs, not three human participants.
Off-target and both remaining parent arms are preserved in the output tables.

## 4. The source control span is not a universal style invariant

Holding one source task out of the span gives average held-control squared-norm
coverage .074819. The source unit/origin is still the frozen full-parent source
unit, so this is not a completely independent normalization experiment.
It nevertheless shows that the two-task control secants span little of the
third task's actual control response at that metric.

Cold controls also act in the complement. Full-arm control-effect RMS:

| Task/target | Full RMS | Complement RMS |
| --- | ---: | ---: |
| advice/E | .641879 | .608898 |
| advice/N | .239217 | .231277 |
| review/E | .320843 | .338465 |
| review/N | .690725 | .610227 |
| neighbor/E | .725248 | .587097 |
| neighbor/N | .613348 | .618157 |

Each is calculated from all six stored control pairs in that task, after the
primary output, without choosing new coefficients. RMS components do not add;
on actual observations they can reinforce or cancel even though coefficient
subspaces are orthogonal. Therefore a complement is not automatically pure
persona information or a stable clean representation.

## 5. The new missing observation

Existing persona and style manipulations are separate, NOT a complete crossed
persona-by-style experiment. They locate overlap but do not identify whether
style moderates a persona response, changes its observation law, or merely
occupies a coincident direction. Constructing artificial sums of their vector
differences cannot supply that missing observation.

The next source design must jointly cross target assignment, style instruction
and task/opportunity. Its informative object is the mixed difference

    I_(k,s,c) = [Phi(x_k+,s,c) - Phi(x_k-,s,c)]
                - [Phi(x_k+,neutral,c) - Phi(x_k-,neutral,c)].

Also retain the individual effects and levels. I=0 permits additive response
on that finite support; nonzero I locates target/style interaction in the
DECLARED representation. It is not automatically a psychological interaction
or a causal natural-language path. Actual jointly generated/observed corners
are required, not pasted embeddings or an oracle response function.

Before collecting, inventory the original persona generator and earlier
HM10A/10B/HM105-119 factorial designs. A new collector needs a mechanism-specific
contrast those designs lack, declared fixed compute bounds, role splits and
no target-answer imitation. Do not tune this projection's rank/kernel/controls,
flip task signs, restart old generation, or substitute another Bayes simulator.
Local frozen-model forwards may be considered only through a new bounded plan;
no API, private material, neural training or publication is authorized here.

This advances the conditional-realization question while preserving GRAIN's
author/condition/choice/state accounts. Psychological grounding and natural
author transport remain separate links, not discarded because this is synthetic.

## 6. Ancestry, verification and implementation

Projection geometry is established, including Scharf/Friedlander1994 SectionIII
https://www.mssc.mu.edu/~daniel/mscs6960/spring2010/ScharfFriedlanderTSP94.pdf .
Their Gaussian, known-subspace and optimal-detector hypotheses are not asserted.
Kernel mean witnesses are mature (Gretton etal2012):
https://www.jmlr.org/papers/v13/gretton12a.html .
OpenAI/math adc7f124 Universal-Tensor-Squares preliminaries21-100 inspired
component-sensitive dual tests only; no Specht theorem transfers to psychology.
The increment is empirical attribution of this function's task dependence and
the joint-observation requirement, not invention of subspace projection.

prepare/fit/score/analyze/readonly verify all completed.48 focused tests passed,
including17 new tests;3 new Python files passed Ruff lint/format. Before sealing,
inspection caught a wrong expected parent-seal filename; corrected to actual
source_seal.json and an integration contract test added. No scientific run
failed, no old artifacts edited and no new inference/API/training/private call.
The source decomposition took .012130 seconds. All handles are terminal.

Full score replay1.78e-15; kernel overlap1.33e-15; source control null2.54e-15;
Gram identity6.66e-16; dual coefficient replay2.36e-16. All324 persona effect
rows,108 control rows and108 cells replay. Same-agent alternate arithmetic,
not independent blind replication. New program/input/output hashes are sealed.

Results: results/grain_math_response_v1/control_overlap_r1/.
Runner: scripts/run_grain_control_overlap_hm149.py verify (read-only).
Do not rerun prepare/fit/score on this directory or edit its four sealed files.
