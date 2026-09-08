# Narrower-target vector functions: M30B/M31

2026-09-09. COMPLETE. The same reference-based construction now supplies25
additional functions:22 relational/session queries and3 adapted-state queries.
Together with M28's nine Big5/MBTI functions, there are34 frozen query equations.

## What was found

Functions constructed on two independent PANDORA authors retain real questionnaire
associations in Japanese MEPS/free-chat text, without training on those46 people.
After the frozen neural prefix, scoring is an affine matrix calculation: no
neural suffix, generated explanation, human-label fit or sign adjustment.

| Free-chat target | Native external r | Frozen function external r [author-bootstrap95% CI] | Function/native fidelity r |
|---|---:|---:|---:|
| Adapted dejected feelings5 | .536 | .564 [.409,.705] | .931 |
| Adapted negative affect8 | .427 | .422 [.236,.609] | .907 |
| Combined negative feelings13 | .518 | .555 [.387,.705] | .961 |
| Disclosure level2 | .284 | .357 [.079,.581] | .915 |
| Overall session evaluation | .319 | .349 [.083,.564] | .949 |
| Pre AI-directed attachment anxiety | .308 | .303 [-.044,.579] | .899 |
| Pre positive AI trust | .324 | .217 [-.014,.435] | .909 |

All three state-function endpoint associations retain family BH q<=.0036.
Relational/session functions remain exploratory: minimum phase-family q=.122.
The native/function external differences above all have paired intervals crossing
zero, including the trust decrease. The achievement is **retention of external
signal in an independently constructed vector function**, not proof of superior
questionnaire accuracy. The three state composites share items and are not three
independent replications.

Across all25 targets, native/function fidelity is median .899 (range .754-.961)
in free chat and .895 (.591-.968) in direct answers. Full tables retain every
target, not just these favorable examples. Direct-answer external state r stays
weak (.109/.166/.126): faithful model readout does not create missing evidence.

## What the formula carries

The unchanged common form is

```
f_q(P,Q) = b_q + <G_q,P-P_Rq>_F + g_q' (Q-Q_Rq)
         = b_q + M_q + C_q + Q_q.
```

Each q has its own token-coordinate matrix and final-state vector. Human-label
correlation decomposes exactly into signed covariance contributions, all divided
by the same sd(f_q)sd(Y_q):

| Target | Mean term | Token-configuration term | Final-state term | Total external r |
|---|---:|---:|---:|---:|
| Adapted dejected5 | .057 | .329 | .179 | .564 |
| Adapted NA8 | .051 | .217 | .153 | .422 |
| Combined negative13 | .033 | .375 | .147 | .555 |
| Disclosure level2 | .054 | .193 | .110 | .357 |
| Session evaluation | .058 | .141 | .150 | .349 |
| Pre AI attachment anxiety | .050 | .076 | .177 | .303 |

Maximum accounting error2.92e-16. Thus ordered query-token configuration carries
much of the state-linked variation; the final state matters more for the
attachment target. A generic pooled mean is not the complete mathematical object.
These are contributions to observed association, not causal shares of human mood.

## Shared paths are signed, not a single common trait

M31 additionally computes1700 query-by-human-criterion links on the same46
participants:25 queries x34 actual criterion occasions x2 views. It also records
600 matched-query pairs and9600 signed cross-component terms (max error3.47e-18).
Big5/MBTI source populations are NOT mixed into this same-subject table.

Examples from the complete free-chat ledger:

- Perceived empathy/session evaluation: native query r=.950, function r=.885,
  human questionnaire r=.668. Final x final contributes .208 in native-r units;
  configuration x configuration contributes .060. All other signed terms remain.
- AI mind attribution/AI attachment anxiety: native .729, function .658, human
  .349. Configuration x configuration (.136) and final x final (.133) both share.
- Dejected feelings/AI attachment anxiety: native -.129, function .025, human
  .003. A positive configuration x configuration term (.117) is opposed by other
  terms. Sharing one component does not require high total trait correlation.

These are descriptive matched-author score relations, not uniquely identified
causal neural paths or evidence that every shared response is personality.
AI-directed attachment anxiety is a relational target, not generic negative mood.
Repeated occasions and overlapping item composites remain explicitly labeled.

## How this was built

1. Reuse the same two already fixed M7 source authors and their first views.
   Their posts are truthfully described as online posts, not AI conversations.
2. Use all22 previously frozen M30 questions and all3 M30C questions. Their
   question-region token identities exactly match the MEPS scoring prompts.
3. For each reference/query, capture full and E1 states; compute the full-state
   partial Q derivative, and six P-derivative nodes on the two fixed E-context
   orientations. Average Simpson paths and the two reference authors as in M28.
   E1 is a legacy common excerpt removal, not a narrow-target-optimized edit.
4. Freeze arrays before any MEPS evaluation.50 reference units,450 calls,
   467.01seconds. No MEPS text or labels enter construction. The source/context
   mismatch is deliberately tested through transport rather than corrected by
   fitting a46-person predictor.
5. Apply the matrices to2300 cached P/Q states across the two MEPS views.
   Freeze function values before joining questionnaire outcomes. New MEPS
   forward calls for this step:0. Native fidelity, external fidelity and signed
   contributions are reported separately, with no clipping or calibration.

This is finite cross-corpus/language function transport in a fixed observer.
It does not establish general cross-language measurement invariance. The neural
prefix is still required: the result removes the downstream neural readout, not
all model-based representation computation.

## Reproduction and decision

Construction: `scripts/run_grain_narrow_functions.py`.
Transport: `scripts/analyze_grain_narrow_functions.py`.
Full cross-target ledger: `scripts/analyze_grain_narrow_covariance.py`.
Plans: `docs/GRAIN_NARROW_FUNCTION_TRANSPORT_M30B.md` and
`docs/GRAIN_ALL_FACTOR_MATHEMATICAL_ATLAS_M31.md`.

Local, ignored outputs:
- `results/grain_vector_mechanisms/narrow_functions_m30b_v1/` (25 coefficient sets).
- `results/grain_vector_mechanisms/narrow_function_transport_m30b_v1/` (50 intended
  native/function comparisons,1040 within-target-family links and signed terms).
- `results/grain_vector_mechanisms/narrow_covariance_atlas_m31_v1/` (expanded1700
  links, complete query matrices and9600 cross-path terms).

GO: common construction plus target-specific matrices, with an explicit occasion
argument and both shared/specific terms. Next finish M29's all-nine human-source
connection and the34-function equation index. Do not restart a accuracy grid or
turn the observed coefficient decomposition into a universal causal claim.
