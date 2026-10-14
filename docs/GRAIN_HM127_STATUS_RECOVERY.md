# HM127 status-bookkeeping recovery

The first `prepare` completed its immutable inputs and capture receipt, then
failed on exit because the imported exclusive-create writer was used for the
already existing mutable `status.json`. Terminal session26671 exited1.
No history/constructor/model calls were attempted. The original code is sealed
and retained, not rewritten. This is an implementation defect, not a scientific
negative result.

The separate adapter authenticates all prepared/source/code hashes and the
zero-call receipt. It archives the initial statuses and failure, charges the
preparation file-mtime span plus one second against the original600 CPU-second
allowance, then atomically updates only root `status.json` and
`01/capture_status.json`. Every scientific file retains exclusive-create writes.
The repair itself and tests are separately sealed before capture. There is no
budget renewal, source/encoder replay, criterion change or new result directory.

Run recovery once, then collect/analyze/readback through this adapter. Any later
scientific/compute failure remains terminal under the original budget rules.
The cold numeric scorer is unchanged. Tests verify the exact mutable boundary,
nonfinite-state rejection and restoration of the scoped proxy after failure.
