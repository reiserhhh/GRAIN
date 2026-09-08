# M19 execution r2: preserve the observer dtype

2026-09-08. Original run `configuration_parity_v1` stopped before any suffix call
because NumPy float64 reference coordinates were sent to MPS without an explicit
float32 conversion. Its FAILED_RETAINED status and zero-call error are preserved.

r2 writes to `configuration_parity_v1_r2`, with fresh implementation hashes and
the same original design/148-call limit. `anchor_coordinates` explicitly uses
the cached native tensor dtype/device. This matches M16's reference handling;
there is no changed endpoint, sample, mirror amplitude or scientific selection.
Test CPU and actual MPS when available, then rerun the same bounded experiment.
