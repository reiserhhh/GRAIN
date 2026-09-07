# Cached entity-state operator: P1L

2026-09-07. Complete, independently audited, zero new neural forwards.
Protocol/source frozen ine300353. This is a model-function readout, not real
personality prediction or an LLM-free system.

## Decision result

The common-prefix name-state difference predicts some held-setting response
variation beyond its norm or a name-only mean. But the result has only4 authored
settings, is conditional on this prompt family and also predicts unwanted venue
effects. It does not supply a universal personality readout or a selective circuit.

| Fixed readout | Total OOF SSE | SSE / training-mean SSE |
|---|---:|---:|
| Training mean |19.6322|1.0000|
| Same-name training mean |31.8273|1.6212|
| Vector norm + Ridge |8.5086|.4334|
| Full vector + Ridge |3.7523|.1911|

Each fold holds out an entire setting and both named entities. Input is the
canonical event-query name state at role+ minus role-, with identical name and
prefix definition.8 setting/entity rows,1536 dimensions,24 response coefficients;
four folds with6 training rows each. All centering/scaling occurs on training only.
Alpha1 is fixed; no hyperparameter or layer selection. EQD duplicate targets omitted.
Source X has centered rank7, which is bounded by8 rows and is not evidence for7
latent human factors. Do not interpret the total SSE as prediction accuracy or r.

## Components matter more than the aggregate

Full-vector OOF squared errors relative to the training-mean predictor, specifically
for the polarity-times-code component:

| Response source | Event | Attribution | Unchanged venue |
|---|---:|---:|---:|
| Native role contrast |.0676|.1224|.8900|
| Name-state patch contrast |.9682|.2489|.5424|

Native contrasts become substantially more predictable. The intervention's event
component barely improves over its already strong mean, while its attribution and
irrelevant venue components both improve. These are different questions and cannot
be collapsed into a single "mechanism solved" statistic. Norm-only already explains
much of the total variation; vector directions add finite held-setting information,
without identifying a unique mechanism or proving a low-dimensional binding code.

The24 complete setting permutations retain each pair of names and all target
columns together. Two permutations (including identity) achieve an SSE no greater
than observed:2/24=.08333. This is not conventional population significance;
exchangeability of four authored settings is questionable and the grid is tiny.
No bootstrapped pseudo-people or selected seed creates extra evidence.

## Independent audit and reproducibility

`scripts/audit_grain_entity_operator.py` reconstructs X from source activations and
Y from the P1K coefficients (source error0). It independently fits sklearn Ridge
rather than the collector's dual solve. OOF replay error1.78e-15; all coefficient/
aggregate SSE metrics1.07e-14;24 permutation SSEs7.11e-14. Independent Gram-eigenvalue
vs singular-value-square error2.27e-13. No new neural inference or human material.

Artifacts under `results/grain_vector_mechanisms/entity_operator_r1/`:
arrays.npz, predictions.csv (768 rows), metrics.csv (96), summary.csv, all24
permutations, manifest/source snapshot/output hashes/run_status. Independent receipt:
`entity_operator_audit_r1/summary.json`. CPU analysis used0.16seconds excluding audit.

## What this contributes to GRAIN

We now separate **evidence representation** from **conditional response**. For a
fixed patch, with the other block states held fixed,

    d_q = R_q(n+delta,h_other)-R_q(n,h_other)
        = integral_0^1 <gradient R_q(n+t*delta,h_other),delta> dt.

This standard directional integral explains why a shared delta can support many
different, sometimes incorrect, judgments: the downstream response operator depends
on the query. P1K decomposes its observable finite responses; P1L shows some of those
responses can be approximated mathematically from existing vectors on held settings.
Neither packet estimates the integral's internal neural gradients, identifies the
unique path used in human personality inference, or replaces the LLM with mathematics.

The next cheapest distinction is not more layers or a larger author queue. P1M
will compare the awkward person-prefixed room question with an explicitly room-
targeted question and reverse the two action-record lines. This tests pragmatic
targeting and recent-record alternatives before extending a conditional operator.
It is a new planned packet, not a completed part of this report.
