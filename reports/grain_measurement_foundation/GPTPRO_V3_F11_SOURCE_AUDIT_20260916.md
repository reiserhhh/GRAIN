# F11 source and optimizer follow-up

## Evidence

The six finite source anchor fits were reconstructed from paired source observations
using a separately specified scalar covariance basis and QR solve. Maximum raw
coefficient difference: 3.20e-15. The reconstruction uses no application targets.
All 162 original MTMM aggregate metric rows (MSE, coverage, mean posterior variance)
were replayed from saved prediction matrices and separate truth files, maximum
difference 5.56e-17.

All nine REML fits were checked using a scalar covariance definition, slogdet and
inverse quadratic forms and a different bounded optimizer (Powell). Nine runs
converged; largest objective improvement over the original fit was 1.29e-10.
This is a local optimization check, not a global optimum theorem. Original models,
scores and reported results are unchanged. No new material or model inference.

Artifacts: `results/grain_gptpro_foundation_v3/F11/source_audit_r1` with
`optimizer_checks.csv` and `summary.json`. Implementation:
`scripts/audit_grain_gptpro_f11_source.py`. Same-agent alternate computation,
not third-party replication. Source-refit uncertainty remains part of the F13
integration; this audit does not relabel fixed-fit intervals as total intervals.

## Next executable interface

F12 previously had only forward transformation functions. The new
`grain_research/gptpro_foundation_v3/f12_transport.py` provides source-pilot inverse
calibration for global affine, conditional affine and cubic inverse regression.
It preserves negative affine signs, rejects rank-deficient anchors, rejects a
nonmonotonic cubic inverse on the fitted range and returns missing outside cubic
range or condition support. Five exact tests pass; 17 F11/F12 related tests and
Ruff pass. This is implementation evidence, not a completed F12 domain experiment.

Next run must follow frozen source300/pilot100(split60/40)/application200 protocol,
shifted population and four channel worlds, with support and missing-condition
certificates. No empirical success is inferred from these unit tests.
