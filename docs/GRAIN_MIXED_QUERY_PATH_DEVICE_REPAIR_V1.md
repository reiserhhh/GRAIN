# HM35: Device-label Companion, Not a Backend Change

The frozen r1 execution stopped before any forward or derivative call. Its
device check compared str(torch.device) to "mps". The live fixed observer
reports "mps:0", type "mps", eager attention and all weights frozen.
Preserve the r1 code, manifest and zero-call status. No Hessian was acquired.

The separately frozen r2 companion hash-checks the original source and adapts
exactly one expression: compare device.type rather than its indexed string.
It does not switch CPU/MPS, attention backend, inputs, model, numerical
tolerances, paths, nodes, stages or budgets. The actual first source node is
still the counted double-backward feasibility observation. All other logic
comes from the frozen original source. No automatic retry of an uncertain
model call is authorized.

Record primary and companion paths separately. Run the original alternate
verifier with explicit --out pointing to r2, once actual acquisition completes.
