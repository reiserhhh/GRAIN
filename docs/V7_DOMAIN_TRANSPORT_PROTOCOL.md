# V7 Domain and Language Transport Protocol

## Non-Negotiable Boundary

GRAIN V7 can transfer a procedure to a new corpus. It cannot transfer PANDORA
numerical coordinates, landmark vectors, reference norms, named factors, or
psychological interpretations merely because both corpora were embedded. A
different language, trading corpus, platform, tokenizer, runtime, operator or
sampling frame is a new domain.

For source domain S and target domain T,

\[
X^S=f_S(Z), \qquad X^T=f_T(Z).
\]

Without paired bridge observations or independently validated alignment,
f_S and f_T are not identified relative to each other. In particular, an
unpaired isotropic target may be rotated without changing its distribution.
Unpaired distribution matching cannot prove the correct source coordinate
orientation.

## Permitted Modes

### LOCAL_REFERENCE_ONLY

Build a target-local reference population before any market/outcome labels are
opened. Use the same frozen V7 procedure only as a recipe. Emit target-local
geometry profiles and support/refusal states. Cross-domain numerical profile
comparison is forbidden.

This is the default and the only mode available for a new trading corpus until
a bridge is collected.

### PAIRED_ALIGNMENT

Use only when source and target have declared paired authors, parallel matched
items, or another justified bridge. Split paired anchors into alignment-training
and held-out confirmation sets. Freeze the map class and thresholds before any
outcome access. A map must pass held-out alignment before numerical
cross-domain geometry comparison is authorized.

### PRETRAINED_ALIGNMENT

Use only with a separately stored, hash-verified and population-independent
alignment validation artifact. Otherwise the validator refuses it.

## Required Target Schema

Every target record must declare a field or an explicit documented absence for:

- author_id
- document_id
- occasion_id
- rind_id
- timestamp
- language

Author and document identifiers are mandatory. The other four may be
UNAVAILABLE_DOCUMENTED, but the resulting loss of occasion, condition,
ordering, or language checks becomes part of the claim boundary.

## Trading Handoff Rule

The first trading phase is measurement-only:

1. freeze target sampling, runtime, operator and target-local reference;
2. run local geometry and support diagnostics;
3. review opportunity, language and drift profiles;
4. only then preregister a separate market-prediction experiment.

The transport manifest refuses a task that requests both early-stage
measurement and market prediction/outcome-label access. This preserves the
measurement-versus-prediction distinction.
