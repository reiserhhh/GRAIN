# HM241 development preflight, before data calculation

The first synthetic-control pytest run gave12 pass/1 failure. Adding a common
log-density offset to each toy author's two sources produced roundoff roots
for targets whose complete score differences were within1e-12 of zero.
The comparison status already treated them as numerical ties, but its root
list mistakenly exposed meaningless roots as if they were order crossings.

Before freezing scientific code or reading experiment outcomes, the output
contract was repaired: preserve the polynomial coefficients and numeric
range, but a within-tolerance tie has no reported order-crossing weights.
The fixed1e-12 tolerance was NOT changed. No precision refinement, science
rerun or actual-data selection occurred. One RuffC401 style issue was also
corrected with a set comprehension.
