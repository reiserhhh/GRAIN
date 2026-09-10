# NEXT80 runtime-only metric compatibility correction

2026-09-11. Primary `2514a40` source capture (76 native calls) and candidate
predictions completed successfully. First cached evaluation wrote its tables,
then failed the frozen baseline replay check. The only metric exceeding1e-10
was `reader_metrics.csv:vector_error_over_native`, max difference1.074491e-8.

Cause: the new range helper computed the native float32 norm by dot product,
whereas NEXT79 used elementwise square followed by sum. Restore the exact old
denominator evaluation order. This does not change source secants, any reader,
prediction, weight, step, evaluation target, candidate or decision rule.

Keep first evaluation in private `analysis_initial_runtime/`. Update provenance
with both original and amended module hashes; rerun only cached evaluation and
the planned cached replay. Native capture is not repeated. Failed evaluation
CPU was not separately recorded and is therefore unknown, not counted as zero
or silently included in successful canonical-stage timing. One focused float32
compatibility test is added after the primary freeze in a separate test file.
