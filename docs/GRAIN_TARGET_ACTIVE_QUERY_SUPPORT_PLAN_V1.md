# HM36: Source-Only Target-Active Observation Family

Status: planned after verified HM35; not executed. No new neural calls or
kernel/rank search. Preserve the frozen HM35 primary, companion and all arrays.

## Decision and Mechanism

HM35 shows both genuinely active source-unsupported directions and changing
local reading. In application AI-trust arms, unsupported signed-integral RMS
is99.1%/101.8% of full RMS, with a negative cross term in the mean arm.
Holding the initial mixed map fixed gives15.3%-60.6% relative endpoint error
across all application groups; integrating its actual changes reduces that
to.0072%-.0250%. A source support failure is not solved by assuming every
large residual is noise, nor by assuming a constant linear response law.

The next question is whether source-observed TARGET-ACTIVE directions cover
cold action better than source-natural displacements alone. This is a support
and observation-family experiment, not a claim of a new cold predictor.

## Fixed Source Construction

Use only144 HM35 source nodes (pair0, wordings0/2, all H/B/Q fibers) and the
unchanged HM34 source models. Keep each query's own ordered token coordinates.
Per query: eight D columns,96 actual P covectors from two fixed H arms and
48 source nodes. Never concatenate different query token spaces.

In the original physical Euclidean metric, define three spaces:

    S_D=span(D);
    S_V=span(source P covectors);
    S_DV=S_D+S_V.

Construct each source basis with normalized NONZERO columns solely to make
numerical rank invariant to column units. Freeze the rank criterion:
Gram eigenvalue >100*float64_eps*max(Gram.shape)*largest_eigenvalue.
Keep raw columns and their physical scales. Reject exact-zero columns
explicitly; do not choose a rank or threshold by application agreement.
At most9 learned CPU basis constructions (three spaces x three queries),
plus18 dimension-matched random control bases,0 model calls. Use thin matrices
and small Gram/QR/SVD calculations, never a dense ambient-dimension projector.
Total CPU analysis limit900seconds; retain partial results without extending it.
Freeze all source bases/hashes before loading application numerical covectors.

## Cold Evaluation and Controls

Use all48 application paths,144 nodes and288 node/arm actions. Source and
application together retain
the original96 paths; do not double-count them as new text samples.

For each frozen source projector Pi, report:

    a=<v_application,DeltaP>;
    a_supported=<v_application,Pi DeltaP>;
    a_missing=<v_application,(I-Pi)DeltaP>.

These use a cold ACTUAL covector as an evaluation reference. They do not
predict that covector without the decoder. Also report norm fractions,
covector projections, full signed sums/cross terms, and both original fixed
quadrature rules. Retain all groups, especially cases with cancellation.

Add source-only dimension-matched random bases with three predeclared seeds
1701/1702/1703 for S_V and S_DV, using the same physical dimension and rank.
No best-random selection. This checks whether more dimensions alone explain
coverage; it is not a null about human personality. Source S_D should replay
HM35's decomposition within1e-8. Alternate QR/SVD or direct dual contractions
must verify projector/action identities with the fixed tolerance.

## Deliverables and Next Branch

Save source basis/raw-scale arrays and rank spectra, source barrier, cold node
and integrated tables, all random controls, numerical verification and report.
Test rank deficiency, nonorthogonal metric transport, zero columns, signed
cancellation, source isolation and dimension-matched controls on synthetic data.
Before real execution freeze code/spec/inputs; no applications in source fitting.

If S_V or S_DV captures substantially more actual cold action than S_D and
dimension-matched random spaces, the new reusable object is a source-learned
target-active observation family. Then address its state-dependent reading
law using the already recorded node variation, not an arbitrary bigger kernel.
If coverage fails, preserve the failure and locate conditional change of the
active family; do not extend the matrix with application covectors.

No API/private/Claude/automation/neural training/publish/lockbox/old queue.
Natural-author response and psychological interpretation, and the original
18 protocols/100cards/44 computational requirements remain open.
