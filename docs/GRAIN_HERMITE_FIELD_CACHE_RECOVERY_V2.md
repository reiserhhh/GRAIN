# HM26 Cache Recovery R2

Original scientific protocol: `GRAIN_HERMITE_ACTION_FIELD_PLAN_V1.md`, unchanged.

R1 was frozen/prepared, with0 fits or model calls. A pre-run schema inspection
found that its consumer expected `native.score`, while the actual cache uses
`native.conditional_score`. No numerical experiment failed or was selectively
rerun. Preserve the prepared r1 and its frozen code/manifest as the defect record.

R2 adds an in-memory compatibility alias, checked for ambiguity. It does not
alter original JSON, numerical scores, source membership, representation,
kernel, alpha or evaluation. Both original implementation and recovery wrappers
are frozen as inputs. Twelve source CPU fits remain the full declared budget.
The independent verifier performs the same schema compatibility and uses its
own explicit Gaussian-operator arithmetic, not the production kernel.

The standalone pytest entrypoint first lacked the repository import path;
`python -m pytest` successfully collected and passed the tests. This environment
error is not an experiment result. Initial Ruff import/shebang findings were
fixed before r1 prepare; no frozen code was edited afterward.
