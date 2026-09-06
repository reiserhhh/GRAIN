# History queue status I/O recovery

2026-09-07. Operational repair, not a changed experiment or recovered result.

## Failure and preserved evidence

The original history invocation65840 failed after30 accepted shards/1,680 authors.
`Path.replace` of queue_status.json.tmp raised Windows PermissionError/WinError5.
The supervisor killed its owned child and removed its lease as designed. Its
failed status was not a timeout, model error, natural budget stop or completion.
The dependent direct-semantic waiter failed closed and loaded no model.

Shard030 had42/56 authors recorded but no completion summary. Once both original
processes were verified dead, its directory/log and remaining temporary status
were moved, not deleted, to `vm05_history_io_recovery_r1`. All89 file hashes and
sizes were verified after preservation. The failed queue status and inventory are
kept there. This partial shard is NOT accepted, merged or used for fitting.
Recollection must use all56 original registered authors, with unchanged material,
sampling, tokenizer, model, precision, prompts and collector code.

## Narrow repair

`scripts/run_grain_history_queue_io_guard.py` delegates to the unchanged original
queue and collector. Only the queue's own status-file write is wrapped. Windows
access/sharing/lock error codes5/32/33 retry at most five times, with waits
.05/.1/.2/.4/.8 seconds, total1.55 seconds per affected write. The payload, target,
original atomic writer and final exception remain unchanged. Other paths and
error codes fail immediately. There is no unbounded retry, exception suppression,
metric rewriting or automatic deletion/restart of partial shards.

Each explicit resume creates an append-only invocation journal under
`io_guard_invocations/`, with original manifest/code hashes, wrapper hashes/source
archive, previous status and actual retry/exit events. The wrapper is operational
provenance IN ADDITION TO the unchanged frozen collector manifest; do not hide it
behind the older source hash or claim the original unwrapped supervisor resumed.
Completed source/cache qualifications remain tied to their original code.

Seven synthetic tests cover exact-value preservation, recognized errors, other
paths/errors and permanent-failure bounds. Together with history source and
collection tests:19 passed. A real Windows file-handle test reproduced WinError5
and succeeded after three retries in.359s, with identical JSON payload and0
forwards. This supports the contention remedy; it does not identify which process
or antivirus held the original file. Artifact: vm05_status_io_real_lock_test_r1.

## Verified continuation and sequencing

After preserving the failure, an identical `--resume-completed` invocation with
the new wrapper and `--max-seconds 60` reverified all30 complete shards. Since the
original2,400-second shard budget could not fit, the original queue itself wrote
PAUSED_BUDGET_BETWEEN_SHARDS at shard030. No child/forward was launched. This was
an explicit verification-only recovery, not relabeling FAILED_STOP as success.
Both new owner exit and absent lease were verified. The frozen direct-semantic
gate can now run without accepting a failed or active history job.

The next full history invocation must keep the previous flags/material and use
the guarded entry point, `--resume-completed`, `--shard-seconds 2400`,
`--max-seconds 28800`, `--shard-cache-bytes 500000000` and
`--max-cache-bytes 24000000000`. Keep its SSH connection open; do not detach into
the Windows session-termination failure already observed earlier. First finish
the separate180-forward semantic panel so there is only one owned GPU model.

The incomplete history family still requires128 shards before source fits. This
repair closes an operational incident, not history mechanism/bridge acceptance.
