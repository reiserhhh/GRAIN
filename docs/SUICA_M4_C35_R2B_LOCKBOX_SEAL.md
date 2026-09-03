# GRAIN M4-C.3.5-R2B Lockbox Seal

Sealed before opening any R2B confirmation response or oracle endpoint.

```text
estimand
SUICA_M4_C35_R2B_CONDITIONAL_RESPONSE_SAFE_CHART_CONFIRMATION_V1

config
382f0aa5de306d716927e2da4dc21158822c54145f502c3b6c7133539e72b2fd

protocol
eaa3c35dd14895e24329172ba8da1b2af39eecc5369e0d1619aa00778b725bd6

R1 decision
32f5991d26049d79594eabba650f1478efd97f57b8e67b1e4124b2d4c8648b68

Stage-A manifest
dea7e92661f71099e52d902ab31e41266811a7f1d6c8c367bd34c7316566b23b
```

Stage A contains repetitions `0..7`, 160 main/null chart cells, eight
support-shift controls, eight latent-alias cells, and 199 source-file hashes.
It contains no ecology response, mechanism endpoint, author mechanism
parameter, oracle basis, or geodesic truth.

Frozen numerical runtime:

```text
Python         3.14.3
NumPy          2.4.4
pandas         3.0.2
SciPy          1.17.1
scikit-learn   1.8.0
```

Every Phase-B shard must provide the exact Stage-A manifest hash above.
Changing the config, protocol, R1 decision, source tree, runtime, condition
digest, or serialized basis invalidates the confirmation.
