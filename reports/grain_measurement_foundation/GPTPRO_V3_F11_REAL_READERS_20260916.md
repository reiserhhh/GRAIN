# F11 actual reader interface

## Finding

The approved F08 caches contain two actual readers of the same 96 controlled
synthetic materials: lexical and pinned local NLI. GRAIN has no same-material
output here and is recorded missing rather than filled with simulated scores.
This run performs no new neural inference or fitting.

Application role, 32 materials:

| Target | Between-reader r | Lexical MSE | NLI MSE |
|---|---:|---:|---:|
| Content pole | .893896 | .062308 | .032325 |
| Evidence qualification | .221727 | .029439 | .353283 |
| Signed qualified evidence q(2p-1) | .714650 | .121037 | .353367 |

Both readers can largely agree on the content direction yet disagree on whether
the content qualifies as the author's actual behavior. Error correlations are
.420323/.640248/.497578 respectively: agreement can coexist with shared errors.
These are correlations against designed material semantics, not personality labels.

## Wording halves and variance

Within each role, the four wording variants are divided even/odd and averaged
within eight designed pole/status cells. NLI application half correlations are
.997811 for pole, .803331 for qualification, .935636 for signed evidence.
Qualification half RMSE is .317072. Thus even substantial wording-half stability
can coexist with incorrect eligibility. Lexical qualification half r=.999876.
The eight designed cells deliberately differ; these numbers are not author
test-retest reliability, independent occasions or population psychometric validity.

A balanced fixed-effects identity separates material variance, fixed reader offset
variance and material-by-reader interaction. It is descriptive, not a latent-trait
variance estimate from two random readers. Its orthogonal components sum exactly
to total variance. The real-reader interface therefore does not pretend to fit the
three-trait/three-reader MTMM simulation to these two-reader material scores.

## Matched effects

432 signatures retain actual-to-quoted, actual-to-hypothetical and
actual-to-negated-correction differences for both readers, all roles, poles and
wording variants. Each carries its designed target difference. In this bank,
`negated` denies the opposite claim and affirms the actual pole; it must not be
treated as unqualified merely because it contains negation. These are finite
semantic edits, not continuous attention Jacobians or human causal effects.

## Verification and next work

576 scalar scores, all 432 edits, nine agreement rows, eighteen half rows and
nine fixed-method decompositions replay through alternate arithmetic with maximum
error 4.44e-16. Eight related tests and Ruff pass. This is same-agent numerical
verification, not independent external replication. Source/validation/application
roles and exact text identity are checked before joining; source is not pooled
with application in headline results.

Results: `results/grain_gptpro_foundation_v3/F11/reader_interface_r1`.
Audit: `reader_audit_r2` (r1 partial audit retained). Input SHA256s are in summary.
Code: `scripts/run_grain_gptpro_f11_reader_interface.py` and
`scripts/audit_grain_gptpro_f11_readers.py`.

This closes the two-available-reader descriptive interface, not all F11. Remaining:
constant-author method simulation with independent-anchor estimation; MTMM source
uncertainty/optimizer and aggregate audit; standard card/hash/DAG packaging.
Do not start another prompt sweep to increase reader agreement. The mechanistic
requirement is to preserve content, evidence qualification and method error as
separate parts of the construction procedure.
