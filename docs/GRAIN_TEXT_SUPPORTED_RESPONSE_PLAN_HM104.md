# HM104: actual text contrasts before response geometry

Frozen before capture, 2026-10-10. No personality-label fitting or accuracy search.
Prior HM103 controls occupy arbitrary K/V directions; this packet asks whether
actual public text changes supply response directions and, crucially, whether
content-by-owner changes add carrier directions absent from the marginal changes.

## Design and decision

Two fixed public lexicons, two bridges and six fixed current-action families
give 24 carrier contexts. Within each, all current token IDs and positions are
identical. Earlier history contains six action entries with an initially unknown
owner. Each content slot has low/mid/high wording from the existing material.
The owner clause names A, names B or leaves identity unspecified. No personality
term, question suffix, answer or outcome appears in the newly encoded body.

All six entries are mid at the origin. Capture the origin, twelve single-content
contrasts, two single-owner contrasts, and 24 content-by-owner corners: 39 actual
histories per context, 936 frozen local forwards. Ignore right-side history padding
at blocks0--16, using the existing scoped mask implementation. Capture only
current K/V before block17 attention/MLP/output. The source six query/covectors
are unchanged and registered before opening application carriers.

For content axis i and owner o in {A,B}, retain the actual finite response

    dF_i(o) = F(x_high,i,o) - F(x_low,i,o)
    I_i = dF_i(A) - dF_i(B).

Compare the corresponding actual carrier difference

    W_i = x_high,i,A - x_low,i,A - x_high,i,B + x_low,i,B

with the span S7 of six unknown-owner content contrasts and the single owner
contrast. Report ||(I-P_S7)W_i||/||W_i|| without fitting target outcomes. Nonzero
innovation changes the measurement design: an ownership flag added after
encoding is not equivalent to jointly encoding qualified content. No claim that
ownership is the only changing semantic variable; the explicit identity clause
and name exposure change together.

## Empirical secant charts, not a natural continuous text manifold

Use orthonormal SVD bases in original Euclidean K/V units. S6 contains all six
content contrasts; S7 adds the owner contrast; S13 adds all six W_i. These three
are theory-derived marginal/condition/interaction constructions, not a rank grid.
No favorable column or target is selected. All singular values and ranks retained.
Actual encoded vectors are finite endpoints; their affine span interior need not
be the encoding of any text. Call it a text-derived secant chart, not reachability.

On z coordinates, F_B(z)=F(x0+Bz), J_B=J_x B. Its declared minimum-norm fields
are U_B=J_B^T(J_B J_B^T)^-1. Calculate all 15 brackets at the origin for S6/S7/S13.
Rank failures remain explicit; do not ridge, drop targets or retune the cut layer.
S6, when square/full rank, is an integrability control; K-fixed/value-only with
S7 is the linear control. These are known mathematical consequences, not discoveries.
S7/S13 response geometry is secondary to the actual finite text interactions.

For S7 joint K/V only, execute all 15 score rectangles at two fixed relative
scales and reversed orientation. The base dose sets maximum first-order state
motion to 1% of the smallest observed marginal secant norm; it is not selected
using score improvement. Two RK4 steps per leg. Read the fixed seventh probe
formed from the equal means of the six source queries/covectors. No new probe fit.
Report drift, closure, area scaling and probe values in model units, not psychology.

## Verification, provenance, stop

Check identical current IDs/positions, actual mask traces and zero forbidden
suffix/output calls. Seal inputs/code before collect and raw/results afterwards.
Verify analytic chart derivatives with central differences; independently rebuild
four-corner contrasts, orthogonal innovation via least squares and every CSV.
Test with synthetic fixtures including absent interaction and out-of-span mixed
interaction. Preserve incomplete/failed collection; no resampling or rescue grid.

Freeze the next decision from data: if mixed text responses/innovations exist,
next construct a jointly qualified evidence operator; if they vanish, simplify
this material's observation to marginal content plus owner. If S7 cannot identify
six controls, report the insufficient declared text support, not absence of traits.

## Literature and what is borrowed

Makelov et al., arXiv:2311.17030v2, abstract/introduction read: an effective
subspace intervention can fabricate rather than localize a mechanism. This
motivates starting from actual text endpoints; affine interpolation still does
not remove that limitation. No DAS/CL training is implemented.

OpenAI/math fixed adc7f1241b42e322a6451854ab7e4b4c146bf78a,
Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch,
build/sections/06-matrix.tex lines1--150: finite first/second-order jet families
must span the declared observation constraint; a smooth right inverse requires
rank. Borrow the design distinction between marginal directions and mixed
innovations, not its elliptic-PDE theorem or exact derivative assumptions.

Classical secants, mixed finite differences, pullbacks and Lie brackets are not
new mathematics. Candidate contribution: an executable qualified-text response
unit, measured differences between marginal and jointly encoded evidence, and
its consequences for GRAIN author measurement. No human validity, API/private
data/training/Claude/Git/automation/lockbox/old queue; original scope stays OPEN.
