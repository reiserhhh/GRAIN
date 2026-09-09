# Source-affine navigation program: build and use

2026-09-09. Complement to [portable navigation](GRAIN_NAVIGATION_RUNTIME_BUILD_V1.md).
This selector reproduces one source-expanded state approximation, not a newly
validated generic personality scoring instrument.

## Input, formula, dependency

Input is a JSONL record with id, texts (list of raw excerpts), and query.
The supported query keys are extraversion, TF_cont and adapted_dejected5.
The tokenizer and query recipe are frozen. Scores are unclipped local-function
units, not calibrated questionnaire scores; T/F positive direction is F.

~~~text
raw excerpts -> frozen query/token recipe -> four extracted blocks
            -> x = flattened query rows + final input slot
            -> parts = x @ C + d
            -> score = anchor + sum(parts)
~~~

No full observer, decoder suffix, LM head, teacher P/Q, kernel snapshots, human
labels or source repository are required at scoring time. This is still a
four-block neural prefix; only the separate token selector is Transformer-free.
Construction used frozen LLM states and48 synthetic plus12 unlabelled reference
author snapshots per query. All coefficients and model weights remain local.

## Reproduce and run

From the independent GRAIN repository, using the existing parent venv:

~~~bash
PY="../project persona/.venv/bin/python"
"$PY" scripts/export_grain_source_affine.py prepare \
  --out results/grain_vector_mechanisms/navigation_v1/NAV-11/new_source_affine
"$PY" scripts/export_grain_source_affine.py replay \
  --out results/grain_vector_mechanisms/navigation_v1/NAV-11/new_source_affine
~~~

Preparation consumes verified source_span_r1 construction artifacts and the
earlier runtime_export_r1 scaffold. It does not recollect or refit human data.
It rejects an existing destination. The observed export is source_affine_r1.

For entirely new inputs, use the generated bundle directly from any directory:

~~~bash
python /absolute/path/to/bundle/run.py \
  --bundle /absolute/path/to/bundle \
  --input /absolute/path/to/excerpts.jsonl \
  --output /absolute/path/to/scores \
  --program source_affine --device mps
~~~

CPU is also supported. Dependencies are NumPy, SciPy, tokenizers,
threadpoolctl, Torch, Transformers and safetensors; exact installed versions are
in bundle/manifest.json. No automatic download or installation occurs.

The operator contains about7.11MB of coefficients versus289.33MB in its source
snapshot representation; the retained neural weights are additional. Fifteen
raw-text source-denied replays pass with maximum error6.33e-15. This means exact
implementation of this candidate, not improved empirical performance. See the
[source-span report](../reports/grain_vector_mechanisms/NAVIGATION_SOURCE_SPAN_REPORT.md)
for mixed natural-response results and complete external comparisons.

Never use an unrelated candidate's reference-author mean/SD to standardize these
scores. A future norm must name its own independent reference population.
Do not distribute private examples, questionnaire joins, pretrained weights or
source-derived coefficients without the appropriate rights and privacy review.
