# HM228: specify the observable contract, not hidden-coordinate precision

2026-10-12, before any remaining graph observations. Protocol revision is
post-HM227 compatibility output, NOT a prospective validation of that failed
state-coordinate gate. HM227 remains FAILED. Neither its code nor result is
modified, and neither compatibility batch is rerun.

## Why the observation contract matters

HM153 already specifies a relative output field instead of an arbitrary hidden
coordinate. HM226's scientific object is the A/B log odds z, candidate mass,
and graph geometry. HM227 added an absolute hidden-coordinate tolerance that
HM226 did not need. In batch026 it rejected max state drift .0002670288 even
though selected-logit, normalizer and log-odds drifts were .0000534058,
.0000477151 and .0000228882. Batch000 had exact equality. There is no evidence
here that more hidden-coordinate optimization would change the scientific
decision. We therefore stop pursuing that state precision and explicitly
re-specify the instrument contract around the observable used by the science.

For W_A,W_B and the frozen state h:

    z(h) = (W_A-W_B) h
    delta z = (W_A-W_B) delta h

A large component of delta h need not imply a large delta z. Equality of z
does not establish full-vocabulary/state/KV/future-generation equivalence.
This is mature linear observation/quotient reasoning, not a new theorem.
Do not call the discarded state gate meaningless for ALL uses of the backend.
Full-key query blocking is algebraically unchanged; finite float precision is
documented by the two already captured same-input checks, not newly optimized.

## Fixed finite execution

Keep all HM226 24 authors/288 events/32 edges/9 cycles/5 targets/both orders/
3 self controls/335 endpoint input IDs, old model/weights/float32/MPS/batch5.
Keep the SAME HM227 row256 backend and unchanged eager causal/padding mask.
No backend/row-size/model/head/weight/prompt sweep or old-job resumption.

Verify old prepared/partial receipt, failed HM227 code and prepared seals, its
two saved compatibility arrays and reports. Existing selected-logit,
normalizer and log-odds differences satisfy the ALREADY CHOSEN 1e-4 output
scale. Hidden-coordinate failure is retained separately, not relabeled PASS.
This is a narrowed INSTRUMENT contract, NOT a smaller scientific scope.

Copy old27 graph batches and reuse both compatibility observations. Capture
only original027..066, 40 batches/200 endpoints, with no additional replay.
Batch027 explicitly replaces the interrupted original attempt. It was never
attempted by HM227. Do not retry any failed HM228 batch.
Total RECOVERY model cap remains1200, INCLUDING HM227's50.69675675 seconds.
HM228 remaining cap =1200-old HM227 runtime, not an expanded budget. Check
before each batch and retain any failed/partial output. Never change settings
or truncate histories to complete. Parent HM226/227 failures remain terminal.

## Scientific interpretation

Run the sealed HM226 full analysis and independent verifier unchanged. All
positions, order effects, self effects, candidate mass, signed cycles and100
historical contacts are retained, whether positive or negative. An improved
r is not success. No new labels enter prompts, graph, or backend selection.

Before receiving the remaining outputs also specify a numerical-only global
ordinal check: orient each observed edge by sign(d), and find directed cycles
by strongly connected components. Zero sign edges do not impose strict order.
Report both exact signs and robust signs after excluding abs(d)<=1e-4 (the
already chosen instrument output scale). Exclusion here is ONLY a diagnostic
of strict-order feasibility, not deletion from the recorded measurement or
Hodge analysis. A directed cycle rules out a strict scalar ordering on those
observed edges. An acyclic graph permits a scalar order but does not establish
human trait validity, unique cardinal calibration, or unobserved pair behavior.
No coefficient fit/optimization/extra model outputs for this companion check.

The useful theory decision is which accounts the author-measurement interface
must retain: position, counterpart relationship, and presentation condition.
Global ordinal compatibility is not inferred from only8 triangle counts.
Read Jiang etal Sec2.2-2.3 https://arxiv.org/html/0811.1067 . Both ordinal and
cardinal treatments already exist. OpenAI/math cocycle remains inspiration
only; no free-group theorem is applied to authors or text.

All original18/100/44 requirements stayOPEN. No Git/API/Claude/automation/
private/new training/download/lockbox/oldqueue. Same-agent audit, not blind.

