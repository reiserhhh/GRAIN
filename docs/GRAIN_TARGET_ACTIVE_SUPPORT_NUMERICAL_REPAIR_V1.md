# HM36 Numerical Basis Companion

The primary target_active_support_r1 is preserved. Its complete CPU battery
ran in29.9269633750082seconds with0modelcalls. Alternate verification found
Gram-derived basis geometry errors beyond the frozen1e-8 tolerance:
subspace1.7968103427357107e-7, geometry1.8860288586441953e-7.
Signed node actions differed by1.227705278747642e-9 and integral values by
5.354059157644146e-10. Primary geometry is not accepted. This failure is not
fixed by loosening tolerance or selecting a new application rank.

The source rank criterion is unchanged: normalized nonzero columns,
eigenvalue >100*float64eps*max(Gram.shape)*largest_eigenvalue. All9 learned
ranks must match their primary counterparts. The companion uses thin QR
then SVD of R to construct that rank, avoiding inverse-square-root scaling
of inaccurate Gram eigenvectors. The alternate checker instead uses a
direct tall-matrix SVD. No ambient dense projector is constructed.

Preserve original code/plan/spec/results. Separate companion helpers,
entrypoints and tests are hash-frozen before executing target_active_support_r2.
Only the in-process source_basis function and alternate basis function are
substituted, with frozen original module hashes checked first. The same
original96paths,144source/144applicationnodes,9spaces,18randomcontrols,
seeds,physicalmetric,sourcebarrier and1e-8 threshold remain.

No new neural calls/API/training or application fitting. The original900second
CPU bound is not extended: combined primary analysis/failed verification plus
companion analysis/verification must remain below it. A second numerical
failure stays a failure, not a different scientific protocol.
