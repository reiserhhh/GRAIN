# Query/Context Action Localization

Evidence: frozen HM32 query_context_cube_r1, alternate verification_r2 with an
explicit CSV-bool serialization adapter, and separate result-after diagonal
alias account_r1. This document supplements HM31 rather than overwriting it.

## Observation Object

    F_q(H,R,Q), alpha_q(H,R,Q)=d_HF_q,
    A_q(H,R,Q;delta)=[F_q(H+delta,R,Q)-F_q(H-delta,R,Q)]/2.

H is the four property-token states after layer16; Q is the endpoint state at
that layer; R comprises every other state and the receiver layout. The fixed
query suffix is token-identical across each paired context. Interventions
preserve actual donor arrays and float32 half-deltas; no lexical editing occurs.
Q is not the entire query representation and R is not pure topical context.

HM31 established that an H-only action law loses context-dependent action.
HM32 separates two carriers previously varying together, before compiling a
conditional law. The scientific object is the action of a vector in context,
not a trait label attached intrinsically to that vector.

## Finite Index Group and Aliasing

The experimental indices form Z2^3, not a physical group acting on hidden
states. The standard characters yield eight coefficients and Parseval energy.
Restricting to the old diagonal Q=R aliases masks m and m XOR011, so that

    d_mean=c_mean+c_RQ, d_H=c_H+c_HRQ,
    d_c=c_R+c_Q, d_Hc=c_HR+c_HQ.

The diagonal has rank4 while the cube has rank8. More repetitions on the
diagonal cannot recover this missing distinction. New off-diagonal observations
do; missing observations must not be imputed from the desired decomposition.

Reference: Filmus, Boolean Function Analysis notes, section1 (2016):
https://yuvalfilmus.cs.technion.ac.il/Courses/BooleanFunctionAnalysis/2015/LectureNotes.pdf
Reading scope is introductory character/Parseval material, not the entire text.
OpenAI/math noncommutative rational-formula manuscript introduction at fixed
commit adc7f1241b42e322a6451854ab7e4b4c146bf78a inspires changing evaluation
support. Its division-algebra/complexity theorem is not a decoder theorem.

## Conditional-Variance Compilation Obstructions

On the equally weighted experimental cube, for scalar actions or Euclidean
vector observations:

    min_f E||A-f(H,Q)||^2 = E||A-E[A|H,Q]||^2
      = E_R+E_RQ+E_HR+E_HRQ.
    min_f E||A-f(H,R)||^2 = E_Q+E_RQ+E_HQ+E_HRQ.

The per-cell oracle is more permissive than a single learned compiler; these
are optimistic finite lower bounds, not held-trained prediction results. The
two fixed intervention directions define a task seminorm, not all psychological
information. Units must remain distinct for gradient, covector and actions.

Adding Q removes only2.97-37.03% of application raw-action H-only MSE bound;
adding R removes62.87-96.67%. These are information accounting ratios, not
accuracy gains. Raw and unit-displacement comparisons agree in direction.

## The Actual Carrier and Natural Co-Movement

R's main gradient energy exceeds Q's in all9role/query groups. Application
R-only gradient relative difference medians are17.71/11.89/22.92%; Q-only
medians5.24/3.87/6.96%. R is the next localization target, not a nuisance to
subtract. Q still matters in specific task directions; total norm cannot decide
whether a carrier is irrelevant to a selected query.

The old diagonal action bound includes signed cross terms:

    V_diagonal=E_R+E_Q+E_HR+E_HQ
      +2<c_R,c_Q>+2<c_HR,c_HQ>.

The full-cube bound additionally includesRQ/HRQ energies and does not contain
these diagonal cross terms. Actual low-mood source-complement actions cancel
on the diagonal, while trust actions amplify. Different design measures must
not be compared as successive model improvements. Co-moving carriers can hide
each other's action without being uninformative themselves.

## What Is Enabled Next

Split R into earlier query states P and the remaining text/system states plus
layout B. Token-identical P donors can be mapped explicitly; heterogeneous B
layouts remain attached to the receiver. Test H/B/P/Q recombinations before
choosing a source-only conditional compiler. If P carries most of R modulation,
compile explicit target-conditioned action; if B persists, localize text-side
addressed carriers. Preserve mixed channels and signed interactions.

This supplies a mechanism-level connection to GRAIN's conditioned observation
principle: conditions change how evidence is read, not just its mean content.
It does not yet identify a human author's response operator B_u, name a new
personality factor or remove the encoder dependency. Old psychometric scope
and old positive/negative findings remain intact.

## Reproducibility

Run stages are prepare, source_calls, freeze_source, held_calls, evaluate.
The frozen original verifier has a preserved CSV bool parsing failure; run
scripts/run_grain_math_query_cube_verification_companion.py for the explicit
r2 adapter. It changes two serialization columns only, no score/role/formula.
scripts/run_grain_math_query_alias_account.py reproduces every HM31 diagonal
from the new cube with zero model calls/fits; do not overwrite existing outputs.
960new forwards/192backwards,314.05modelseconds; no API/private text/training.
