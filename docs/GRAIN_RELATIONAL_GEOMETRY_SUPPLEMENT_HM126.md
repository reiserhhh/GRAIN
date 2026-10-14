# HM126 result-after geometry supplement

The primary seven cases are terminal and numerically verified. New-expression
Gram alignments remain .85--.96, yet different-instance correct-family ranks
are135/720 and187/720. Best orthogonal residuals are nonzero. Before changing
the algorithm, account for two possible sources of misleading global similarity.
This supplement is explicitly exploratory, planned AFTER those results.

For all seven cases and both calibration/held-out Gram matrices, retain the
whole Gram and decompose with u=ones(k)/sqrt(k), P=I-u u^T:

    G=(u u^T)G(u u^T)+PGP+(u u^T)GP+PG(u u^T).

All four blocks stay. The first is a common design-family component, not a
named psychological general factor. Report common trace share and full and
family-contrast Gram alignments, with all720 fixed family permutations again.
Do not use centering as denoising, optimize the retained block, or discard origins.

Using source Gx>0, compute the pulled-back target relation metric

    L=Gx^(-1/2) Gy Gx^(-1/2).

Its eigenvalues bound ||Y theta||^2/||X theta||^2 over nonzero declared relation
directions. All eigenvalues1 would mean a pure isometry. A common scalar would
mean uniform scaling; dispersed values mean direction-dependent deformation.
This is classic generalized-eigenvalue/Rayleigh-quotient theory, not a new
theorem. No new metric is deployed, no target fit, and no winning directions
are selected. All anonymous directions and outcomes remain.

Second-formula checks: block reconstruction and trace split; each eigenvector
is mapped back to its original coefficient direction and its direct norm ratio
is recomputed. No neural/API/private/Claude/Git/automation/lockbox calls.
Use remaining original600CPU-second allowance; exclusive child directory,
sealed new code and source inputs. Parent scientific outcomes remain unchanged.
