# GRAIN v1.1.0 Naming Migration

Release date: 2026-09-03

## Decision

The current method is named **GRAIN: Group-Referenced Analysis of Individual
Natural-Language Expression**. In Chinese it is described as **GRAIN
个体语言纹理分析法**; in Japanese, **GRAIN 個人言語テクスチャ分析法**.

This release changes the public identity and explanatory center of the method.
It does not alter a scientific estimand, rerun an experiment, or reinterpret a
stored result.

## Changed

- README files, current theory documents, publication plans, and English,
  Japanese, and Chinese manuscripts now use GRAIN as the canonical name.
- Individual language texture is defined as the reproducible multiscale
  organization of local language events relative to an explicit reference
  population, opportunity distribution, observation operator, and calibration
  version.
- `grain_core` and `grain_sim` are the canonical Python namespaces.
- Canonical GRAIN command wrappers are provided for the public smoke,
  synthetic-ground-truth, theory-closure, and legacy-lockbox paths.
- GRAIN-prefixed environment variables are canonical; selected runtime paths
  retain fallbacks for their former names.
- CI now enforces the naming boundary and verifies the immutable v0.2.1
  scientific lockbox independently of later software and naming releases.

## Preserved

- The `v0.1.0-prereg-sealed`, `v0.2.0`, `v0.2.1`, and `v1.0.0` tags retain
  their historical identities.
- Release bundles, lockbox manifests, persisted reports, registered estimand
  IDs, and result paths are not renamed.
- `suica_core`, `suica_sim`, `run_suica_*`, `SUICA_M4_*`, and equivalent
  historical identifiers remain compatibility and provenance surfaces.

The binding boundary is specified in
[`GRAIN_NAMING_MIGRATION.md`](GRAIN_NAMING_MIGRATION.md), and the machine-readable
inventory is written to `reports/GRAIN_NAMING_AUDIT.json`.
