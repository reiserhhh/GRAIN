# GRAIN toolchain: current `main`

[English manual](GRAIN_USER_MANUAL.md) · [简体中文手册](GRAIN_USER_MANUAL_ZH.md) · [日本語の手引き](GRAIN_USER_MANUAL_JA.md)

This integration combines the existing native, liquid and sparse-solver performance improvements with the open-stream operators, response calibration, and multi-source evidence panel. It is a software update to `main`, not a new research lockbox or replacement for an existing frozen instrument. Public modules import as `grain_core`; historical `suica_core` imports remain available. The older released tags keep their own contracts.

## Choose the tool by the quantity you want

| Task | First tool | What to check before interpreting |
| --- | --- | --- |
| Build coordinates and place a new writer | [Handbook study and frozen placement](GRAIN_USER_MANUAL.md#2-run-an-entire-small-study-offline), `liquid_instrument`, `liquid_reference_error` | Use separate discovery, calibration and target authors. Keep identical basis and axis order. |
| Combine natural writing, scopes, multiple readers and recorded task menus | [Evidence Panel](GRAIN_EVIDENCE_PANEL.md), `evidence_panel`, `evidence_stream`, `evidence_text` | A complete feed, an eligible opportunity, a known outcome and a usable coordinate are four different declarations. Namespace independent sources; share an identity or coordinate only when linked. |
| Locate an addressed behavior and compare conditions | `event_response`, [response profiles](GRAIN_RESPONSE_PROFILE.md) | Bind each response to its evidence address and original opportunity. Report rates among replies separately from rates per opportunity. |
| Adjust for reader errors or audit score comparability | [Open-stream psychometrics](GRAIN_PSYCHOMETRIC_OPEN_STREAM_V2.md), `response_profile.measure_calibrated` | Freeze calibration separately. Out-of-envelope rates return `reader_incompatible`, and weak discrimination returns `reader_unidentified`; neither gives a point estimate. |
| Account for a changing population, feed or event clock | [Open-stream G-1 to G-6](GRAIN_OPEN_STREAM_OPERATORS_G1_G6.md) | Distinguish participant change, weights, entry and exit. Do not treat a missing feed as a quiet feed. |
| Trace retention, withdrawal, or common dynamics | [Liquid tool](GRAIN_LIQUID_TOOL.md), [Riccati flows](GRAIN_RICCATI_AND_INTERPERSONAL_UPDATE.md), [dual-view comparison](GRAIN_EVENT_AND_VIEW_INCREMENT.md) | Compare direct, simple-lag and full-output intervals on the same cutoff and calibration. Report separately if only interval width changes. |

## Start with an offline check

After installing `requirements-lock-main.txt`, run the following from a fresh checkout. Use a new path for each result file.

```sh
python -m examples.handbook_study /tmp/grain-handbook
python -m examples.evidence_panel examples/evidence_stream_demo.jsonl /tmp/grain-panel.json \
  --queries examples/evidence_stream_queries.json
python -m grain_core.response_profile examples/response_profile_demo.json
python -m grain_core.psychometric_open_stream examples/psychometric_open_stream_demo.json
python -m grain_core.dual_view_increment examples/dual_view_increment.json
```

The stream example returns participation flow, one observed three-cell shift, and an interval for a matched choice under an unknown outcome. The dual-view example changes the threshold decision relative to `simple_lag`. Neither needs credentials, a model download, or a private corpus. The illustrated inputs are construction examples, not population norms.

## Throughput, verification and data

- [Runtime and numerical compatibility](GRAIN_PIPELINE_PERFORMANCE.md) covers native placement, factor screening, frozen batch scoring and sparse cutoff bounds. [Open-stream engineering measurements](GRAIN_EVIDENCE_PANEL.md#性能与兼容性) separately report panel indexing, elapsed time and peak process memory. Whole-document JSON can be faster for moderate inputs; line-delimited input uses less peak memory and still holds its index in memory.
- The `Check the integrated GRAIN toolchain` CI step covers the supported software interfaces. All 119 selected tests passed locally. The exhaustive `python -m pytest -q -p no:cacheprovider` run in a clean checkout still had 89 failures and 10 errors; the same 99 test cases failed or errored when replayed on the previous `main` (`b36764f`). The runner also names 17 historical modules whose registered local artifact is absent; it does not report them as passing. They are collected when the expected assets are present. Use `--require-local-artifacts` to make missing artifacts an error. The strict naming audit and sealed-tag integrity step are separate checks.
- The repository ships small examples and aggregate engineering replay reports. Do not upload downloaded corpora, local manifests with identifiers, or raw JSONL. The sole JSONL exception is the synthetic fixture in `examples/`. [Data access and permissions](DATA_ACCESS.md) still apply.
