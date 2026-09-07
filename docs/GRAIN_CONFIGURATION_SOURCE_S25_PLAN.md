# S25: actual configuration-head features connected to source N

2026-09-08. Planned next, not executed. Follows completed S24, not another
gradient/template grid. Its linear no-fit reconstruction of the measured head
allocation already gives r=.9543 and energy-normalized error=.0230.

## Mainline question and fixed formula

Does z_config(u)=mean_v g_N,S20^T [h_native(u,v)-h_projected(u,v)] retain the
real-author N association of the useful configuration route? Keep S20's original
24-author g_N fixed. No questionnaire fitting, calibration, sign inversion,
post-outcome blend or new trait selection. This is an exposed source mechanism
experiment, not an untouched PANDORA benchmark.

Use the same213Big5-only N authors and three existing12-comment views each.
S9 body projection is unchanged; no source corpus preprocessing change.
Both conditions have the same final query at the block16 boundary by design.
Capture Q,K,V using the already verified prefix-only path, then reconstruct
the full final head as concat_h sum_j softmax(q_h^T k_j) v_j. All tokens count
in this head sum, not only body positions; the intervention is body-only.
No upper decoder, LM head or backward passes are required for production scores.

An implementation-level reduction is available before collection: the native
prefix already saves block16 boundary H. Local Qwen2DecoderLayer code applies
tokenwise RMSNorm then affine K,V projections. Therefore the projected condition
can be computed from the exact S9 P(H), without rerunning blocks0-16:

K0=RoPE[W_K RMSNorm(P(H))+b_K], V0=W_V RMSNorm(P(H))+b_V.

The last token is not projected, so q0=q1; retain its native scaled/rotated query.
Use the original model modules, exact token positions and floating-point S9
projection expression, not a new approximate normalization. This reduces the
preferred path to one prefix plus one local KV reprojection per view. Check
against stored S24 heads before counting it as a verified shortcut; no need for
a separate long optimization series. Hash the local Qwen implementation too.

The mathematical basis is the exact attention readout and the line integral
of the suffix differential, with a transported mean covector from S19/S20.
[Sundararajan et al.2017](https://proceedings.mlr.press/v70/sundararajan17a.html)
supports the integration identity; S24 supplies empirical large-displacement
transport. Approximation quality and source association remain different metrics.

## Bounded execution

Prepare the full membership, prompts, reader hash and source comparisons before
collection. First eight authors from S24 indices24:32, all three views:24prefix
calls plus24local KV reprojections. Compare view0 heads and z with stored S24 heads, with
maximum head error1e-4 and linear-score error2e-5. Require zero forbidden suffix
calls and zero final-query change under projection. Keep the original model
graph/hooks; no frozen helper rewrite.

Use the pilot only for numerical validity, runtime and decision value, not
questionnaire associations. If useful and projected compute fits20minutes,
complete all213authors: preferred639prefix calls plus639local KV reprojections,
0full-suffix/0backward,1200seconds after loading, one model process. Reuse pilot packets.
The conservative two-prefix method remains an explicit fallback if local
reprojection fails numerical replay, with an absolute1278prefix-call cap.
Record a fallback decision before resuming; never alter the measured formula.
Every author is resumable. A resource pause preserves outputs and is not
statistical futility. Do not silently expand the cap if projected cost is too high.

Save both head vectors, their difference, three per-view predictions, query
identity checks and source prompt hashes. Freeze all213mean scores before
joining only the already-authorized source N labels from
response_student_source_connection_r2/private_predictions.csv.
Compare on identical authors with S20 pairing scores and S9 C, native and
projected endpoints. Keep primary189outside gradient reference, discovery24
and all213 clearly separate. No protected bridge/MEPS/lockbox access.

Report trait-source Pearson/Spearman, bootstrap intervals, paired delta against
S20, covariance with Y and with C, prediction spread and three-view consistency.
The new feature is a component/alternative to S20, not an independent additive
signal; no ensemble is selected from these source labels. Preserve both routes
even if one has lower source correlation. If z_config carries useful source
association, it connects S9's larger configuration contrast to a compact
operative formula. If weaker, distinguish head allocation from the larger
other-path contribution rather than blaming the mathematics or erasing S20.

## Files and continuation

Proposed runner: scripts/run_grain_configuration_source.py.
Proposed helper/tests for reconstructed native/projected heads, frozen reader
replay, query preservation, hook cleanup and author-ID matching.
Output: results/grain_vector_mechanisms/configuration_source_s25_r1/.
Use the parent venv, local frozen Qwen, MPS/FP32 and existing prefix module.
No new training, hosted APIs, model downloads, old queues or released scorer edits.
Do not invoke Claude while this discriminating next experiment is executable.
