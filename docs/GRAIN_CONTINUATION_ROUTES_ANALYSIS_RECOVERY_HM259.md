# HM259 offline reader recovery

Written after collector17639/PID45070 completed864batches once, but before any
analyzed results were written. Analyze23162 failed at family.items(): the
sealed observations() function reused the outer list name `values` in two
local dictionary loops. It returned the wrong container. This is a software
reader failure, not a failed neural observation, changed scientific criterion
or reason to repeat the model.

The original code/inputs/collected seal are NOT edited. A separate reader
uses distinct collection/chunk variable names and validates every addressed
capture. Native arrays still come from HM258; closed/direct/relay arrays still
come from the SAME complete HM259 collector. No values are imputed or refit.

The child analysis_recovery_r1 binds all original method hashes and every
collected capture. The original analyze/verify implementations run with only
their observations and capture-path functions scoped to this corrected loader.
These scoped functions are restored in finally. Output algebra, source fields,
bootstrap seed/draws, metrics, signs, support and budget remain unchanged.

Tests use synthetic saved arrays to check complete address/shape/order and
mutation rejection. Original analysis exception and no-output state are saved
as provenance. Child prepare/analyze/verify/check are CPU/numeric operations;
none can collect, encode, generate or renew the original1800second cap.
The scientific collection remains945.292644modelseconds/3168new sequences,
8guards/2guard-prefix-encodes;no automatic retry or replacement.
