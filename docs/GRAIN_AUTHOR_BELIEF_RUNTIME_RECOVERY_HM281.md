# HM281: one status-only I/O companion

Primary prepare succeeded and is frozen. Compute exited1 before the scientific
try/loop because HM170 R.write uses exclusive creation for runtime.json, which
already existed. No source bundle, prediction or numerical scientific outcome
was produced. The failure and PREPARED-to-failure administrative status remain.
No primary scientific code or input hash changes.

One companion reuses ALL prepared inputs and the unchanged compiler. Only
writes to THIS companion's runtime.json use existing HM218 atomic status();
all scientific writes remain exclusive. The adapter does not read private
materials, invoke HM218 experiments or change scores, priors, modes or checks.
Charge the failed process's.261422s conservatively and reduce the180-second
integer CPU cap to179; no renewed budget, scientific retry or parameter rescue.
All later stages reuse the primary functions. This fixes an execution defect,
not a missing human measurement or a numerical-precision problem.
