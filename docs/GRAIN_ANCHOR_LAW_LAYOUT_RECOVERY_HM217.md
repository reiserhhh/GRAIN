# HM217 cache-layout compatibility, frozen after a pre-statistic failure

Primary `anchor_law_disintegration_r1` prepare completed. Its first compute
failed at contacts before any permutation or statistic was produced. NumPy
advanced indexing `[:, :, [0,6], 2, :, 2]` placed the reader-mode axis FIRST:
stored response shape (2,24,2,5), not (24,2,2,5). The primary science/config,
inputs and prepared seal are retained unchanged. This is an interface error,
not a negative scientific outcome or a rationale for retuning.

Only permitted correction: copy the sealed primary observations and transpose
native_response axes (1,2,0,3), restore the registered author/view/reader/target
layout, validate the shape and compare all values to the SAME parent response
using sequential basic indexing. No new response, choice, data, parameters,
permutation or scientific code edit. New result directory
`anchor_law_layout_recovery_r1` has its own manifest and stage seals.
Run the unchanged HM217 compute/verify/cold through this compatibility CLI
once. The shared module/primary runner remain sealed. Record this failure in
the final theory/report; do not call the primary run successful.
