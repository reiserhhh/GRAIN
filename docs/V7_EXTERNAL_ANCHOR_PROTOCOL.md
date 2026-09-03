# V7 External Anchor Protocol

## Boundary

External labels may test a frozen GRAIN object; they may never select its
operator, representation, rank, loading rotation, norm, score sign, or name.
The first contact with labels follows a pre-written manifest and an untouched
anchor cohort.

## Required Freeze

Before label access, record:

1. the scored bundle path and SHA-256;
2. score-cohort and anchor-cohort identifiers plus their overlap count;
3. `pre_freeze_label_access: false`;
4. the complete hypothesis list, directionality, score components, and missing
   data rule; and
5. a multiplicity plan.

The primary anchor test is a frozen-score omnibus across the registered anchor
family. Component follow-ups use the already fixed component list and a max-T
or Freedman--Lane permutation family as declared in the manifest. Labels cannot
be used to rotate, remove, rename, or re-norm a component after the freeze.

## Refusal States

- `REFUSE_PRE_FREEZE_LABEL_ACCESS`
- `REFUSE_BUNDLE_HASH_MISMATCH`
- `REFUSE_ANCHOR_COHORT_OVERLAP`
- `REFUSE_MULTIPLICITY_PLAN`

Passing this check means only that the proposed anchor analysis is eligible to
open labels. It does not establish an association or a psychological construct.

## Required Construct-Validation Extension

When an eligible anchor cohort exists, it must separate at least three claims:

1. convergence with a predeclared external measure or observable behavior;
2. discriminance from a method-matched text artifact such as length, format,
   prompt condition, or scorer version; and
3. stability of the frozen score under registered scorer/window perturbations.

This is an MTMM-style validation plan, not a license to rotate, rename, or
delete anonymous components after seeing correlations. A positive association
would make an `L4_EXTERNALLY_ANCHORED_CANDIDATE`, not a completed personality
or clinical scale.

## Stronger Construct-Validation Gate

```bash
python scripts/validate_suica_v7_construct_validation.py \
  --manifest path/to/frozen_construct_validation_manifest.json \
  --output results/v7_construct_validation/preflight.json
```

The stronger gate additionally requires predeclared convergence hypotheses,
text-length/format/prompt artifact controls, scorer/window perturbation, and
an independently collected anchor method. A template is available at
`configs/v7_construct_validation_manifest.template.json`.

It also freezes separate score- and anchor-cohort membership commitments,
requires an anchor reliability estimate or independent source, and declares
how attenuation will be handled. Smallest effects of interest, incremental
validity beyond method artifacts, and an independent replication population
must be written before labels are opened. Association alone is therefore not
enough: the candidate must survive uncertainty, method-artifact controls, and
replication without post-hoc rotation or naming.

## Machine Check

```bash
python scripts/validate_suica_v7_external_anchor.py \
  --manifest path/to/frozen_anchor_manifest.json \
  --output results/v7_external_anchor/preflight.json
```
