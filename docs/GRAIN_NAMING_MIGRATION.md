# GRAIN Naming and Compatibility Policy

Status: **binding naming policy, 2026-09-03**

## 1. Canonical name

The canonical name of the methodology is:

> **GRAIN: Group-Referenced Analysis of Individual Natural-Language Expression**

- Chinese: **GRAIN 个体语言纹理分析法** / **基于群体参照的个体自然语言表达分析法**
- Japanese: **GRAIN 個人言語テクスチャ分析法** / **参照集団に基づく個人自然言語表現分析法**

`GRAIN method` or `GRAIN analysis` is preferred in English prose. GRAIN is a
foundational analysis method for constructing future natural-language
measurement procedures. It is not itself a finished scale, diagnostic test,
or validated personality taxonomy.

## 2. Operational meaning of language texture

Language texture is the multiscale, reference-relative organization of an
individual's observed language events. It includes the distribution of local
positions, condition selection, conditional response, temporal transitions,
interaction, and higher-order path structure. A single embedding or sentence
is not a texture. Texture arises only from the reproducible arrangement of
multiple observations under an explicit population, opportunity distribution,
observation operator, and calibration version.

The existing author object remains the formal core:

\[
\mathcal A_u=(\pi_u,\mu_u,\mathcal R_u,\mathcal K_u,\mathcal C_u,\mathcal P_u).
\]

GRAIN changes the public name and explanatory center; it does not silently
change an estimand, result, or mathematical definition.

## 3. Historical-name rule

The project was developed under the former name **Sliced Utterance In-Context
Assessment**. Historical names remain only where changing them would damage
provenance or compatibility:

1. Signed tags, release bundles, lockbox manifests, hashes, and persisted
   reports remain byte-for-byte historical evidence.
2. Registered estimand IDs such as `SUICA_M4_*` remain unchanged.
3. Existing filenames, commands, configuration keys, environment variables,
   and the `suica_core` / `suica_sim` namespaces remain supported legacy
   identifiers.
4. New prose, papers, code, tests, configs, and public interfaces use GRAIN.
5. `grain_core` and `grain_sim` are the canonical Python import namespaces;
   the legacy namespaces remain available so recorded experiments still run.

Retaining a historical identifier does not preserve the former name's status
as the name of the current methodology. It preserves the audit chain.

## 4. Repository naming rule

Use the following vocabulary by object type:

| Object | Preferred term |
| --- | --- |
| Methodology | GRAIN method / GRAIN analysis |
| Theory | GRAIN measurement theory |
| Software | GRAIN system |
| Technical output | GRAIN language-texture object or score candidate |
| Future validated tool | GRAIN-derived instrument |
| Historical artifact | legacy SUICA artifact, followed by its exact ID |

Do not call GRAIN a scale. Do not call a language texture personality or state
until independent construct evidence licenses that interpretation.

## 5. Automated audit

Run:

```bash
python3 scripts/audit_grain_naming.py --strict \
  --json-out reports/GRAIN_NAMING_AUDIT.json
```

The audit inventories every tracked former-name occurrence. It rejects former
brand prose on active documentation surfaces while allowing frozen evidence,
registered IDs, and explicitly marked historical compatibility references.

