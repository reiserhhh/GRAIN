# HM261 offline registered-address reader recovery

The primary collector completed all672 batches/4800 candidates once, using
209.89823785517365 model seconds. Primary analyze9717 failed before any analysis
output because `captures.glob('*.npz')` visited macOS AppleDouble `._*.npz`
metadata. Their observed header is 00051607; they are not NumPy observations.
Do not enable pickle, delete those files, alter primary science, or rerun models.

The separate recovery uses the EXACT registered addresses 0000--0671, whose
hashes are already in the primary collected seal. It leaves primary equations,
keys, source fields, bootstrap, target lengths, native observations and all
thresholds unchanged. Its Path adapter enumerates registered addresses instead
of wildcard discovery. The original analyze and verify functions otherwise run
unchanged. Relative links point to the primary captures and basis; new analysis
outputs and their provenance live under `analysis_recovery_r1`.

Prepare must confirm all672 capture names in the seal, complete runtime, absence
of primary analyzed outputs, and metadata headers before freezing the recovery.
The adapter is tested against both metadata and an unregistered numerical file.
No native collect entry point exists in this recovery. Analysis, verification
and check are each bounded CPU operations, not a renewed scientific experiment.

