# Result-triggered remainder intervention

2026-09-15, after conditional_operator_v1/r2 completed and was audited. Not part
of an untouched preregistration. No source model, rank or split is changed.

The application E-query full-donor action correlates .4643 with full contrasts,
but compiled action correlates -.2603 and its RMSE .09693 exceeds zero .09285.
Test whether missing native action remains in the donor-minus-compiled remainder,
not whether another parameter choice improves prediction.

For each of the same 72 application pairs, both existing queries, let a be the
compiled head change, d the actual donor change, and r=d-a. Existing records give
f(0), f(a), f(d). Add ONE local forward per pair/query for f(r), 144 total.
All rows used; no cherry-picked axis, task or seed. No API or new generation.

Exact four-corner quantities:

    A = f(a)-f(0)
    R = f(r)-f(0)
    I = f(d)-f(a)-f(r)+f(0)
    donor_effect = A + R + I
    symmetric_A = A + I/2; symmetric_R = R + I/2

This is finite action accounting, not a psychological variance decomposition.
Report signed and RMS quantities (means may cancel). If R retains the donor
effect, a remainder matters for measurement, but this does not distinguish loss
in the four coordinates from failure of the fitted linear map/rank constraint.
Do not call r pure noise or interpret its magnitude as human personality content.
