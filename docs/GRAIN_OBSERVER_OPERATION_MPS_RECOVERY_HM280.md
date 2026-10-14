# HM280 one zero-outcome mechanical recovery

2026-10-14 JST, before companion collection. Primary observer_operation_r1 is
frozen and FAILED;0 saved families,1.084523875033483 charged model seconds.
The reused HM255 native_items called full.double() on MPS, which cannot hold
float64 tensors. No scientific observation was saved; preserve primary code,
head, manifests, runtime and every scientific seal unchanged.

Reuse the ALREADY implemented HM255 math_item_bridge_mps.native_items. It moves
the same float32 FULL vocabulary logits to CPU before float64 logsumexp. Model
weights, evidence tokens, both operations, all50 items, alphabet, output states,
batchsize and tolerance remain unchanged. This is an export compatibility
repair, not observation tuning. The primary failed first partial call is NOT
a scientific replication.

Companion mps_cpu64_recovery_r1 receives the same frozen inputs, units and
scientific protocol. Remaining cap1498.9154761249665 seconds; combined original
plus companion cannot renew1500. One companion only, no outcome retry or relaxed
guard. Reuse the existing MPS mechanical tests. No new API/private/train/Git/
Claude/automation/lockbox/oldqueue. The primary stays visibly FAILED.
