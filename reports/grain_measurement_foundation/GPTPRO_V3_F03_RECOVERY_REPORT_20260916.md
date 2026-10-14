# F03 Recovery: Opportunity, Selection and Response

2026-09-16 JST. Original V3 goal remains ACTIVE; this report does not close the 100-card plan.

## What Changed

The previous generator consumed different random streams for different designs,
included design in author IDs, and refitted propensity slopes on each application
prefix. The restored implementation uses paired authors and exogenous draws,
source-only shared selection slopes, and application-prefix MAP intercepts.
The published-only adapter actually removes opportunity, time-index and global
event-address fields that could reveal missing publication counts.

All four frozen worlds, three designs and three seeds were run with
300 source / 100 validation / 200 application author instances per world/seed,
and nested prefixes 32/128/512. This gives **7,200 paired simulated author
instances**, **21,600 design-specific instances**, and **11,059,200 records**.
The full run took approximately 254 seconds locally; no API or neural training.
Source learning, prefix prediction and terminal truth scoring are separate processes.

## Main Result

At prefix512, free-design application performance, averaged over seeds and two
response conditions:

| World | Selection assumption | Naive MSE | Source-learned IPW MSE | IPW mean bias | Fixed-design Beta MSE |
|---|---|---:|---:|---:|---:|
| W3-A | MAR, lambda=0 | .006718 | .003413 | +.005730 | .000735 |
| W3-B | MAR, lambda=.8 | .006444 | .004181 | +.001673 | .000729 |
| W3-C | MNAR, lambda=0 | .052649 | .037730 | +.174752 | .000732 |
| W3-D | MNAR, lambda=.8 | .054476 | .038341 | +.168670 | .000710 |

IPW improves MAR response recovery and can partially improve MNAR recovery too.
For example, W3-C IPW minus naive MSE is **-.014920**, paired-author 95% CI
**[-.016102, -.013749]**, but its mean response bias remains **+.174752**.
Consequently, statistical improvement alone is not evidence that selection has
been removed. Fixed selection gives much larger improvement in these worlds.

All comparisons use 2,000 bootstrap replicates, paired authors and seed strata.
At prefix512 all 600 application instances per world have both conditions
available for these comparisons. Smaller prefixes retain and report complete-case
counts and missingness rather than silently replacing missing observations with zero.

## Mathematical Findings and Production Rules

1. **Selection adjustment is assumption-specific.** Exact enumeration over all
   288 frozen population cases verifies the unrestricted response bounds and the
   narrower bounds under independent opportunity. Oracle observed-covariate IPW
   recovers the target in MAR, but generally remains biased in MNAR. This separates
   finite fitting error from a wrong selection assumption.
2. **Unknown denominators cannot be supplied by a predictor.** Published-only
   data return no publication rate or opportunity/selection point estimate.
   Knowing the publication indicator but not opportunity permits partial bounds,
   not separate pi/q estimates.
3. **Different targets can have identical observed data.** An explicit general
   selection counterexample reproduces `[P(S0),P(S1,Y0),P(S1,Y1)]=[.8,.1,.1]`
   for response means from .1 through .9 by choosing `q1=.1/r` and
   `q0=.1/(1-r)`. This witnesses the unrestricted-selection bound; it does not
   assert nonidentification after imposing known MAR or a known selection law.
4. **Keep identification and sampling uncertainty separate.** Every individual
   condition/prefix has 2,000 multinomial endpoint resamples. These are percentile
   intervals for estimated bound endpoints, not automatically simultaneous
   confidence regions for the latent response. Zero opportunity retains `[0,1]`.
5. **The earlier GRAIN readout is retained, not rejected.** The old
   `choice_design.anchored_cells` level and half-contrast are exactly
   `(r_minus+r_plus)/2` and `(r_plus-r_minus)/2` for selected conditional means.
   **32,400 old readouts were replayed with zero transformation error.** IPW
   changes the targeted response distribution under assumptions; it is not
   required to equal the old selected-text readout.

For GRAIN's measurement-production method this gives three separate decisions:
retain observed choice, define the intended conditional response, and determine
whether that response is identified from the available observation design.
Do not import simulated known opportunities into a PANDORA or other natural-text
analysis where those opportunities were not observed.

## Evidence

- Numerical run: `results/grain_gptpro_foundation_v3/F03/recovery_full_r1/`.
- Cold audit and supplementary pi/q/publication/Delta losses:
  `results/grain_gptpro_foundation_v3/F03/audit_full_r1/`.
- **929 numerical artifact hashes** and **36 model/source hashes** verified.
- **7,372,800 paired observation comparisons** verify matched conditions and
  identical potential responses wherever a response is observed.
- **1,004,400 supplemental target rows** explicitly separate pi, q, publication
  probability, response contrast and plug-in q bounds.
- All source selection fits converged. Thirteen F03 tests pass; the combined
  recovery/legacy V3 test suite reports **69 passed in 2.11 seconds**.
- Source snapshots, old failed outputs and smoke revisions are preserved.

Tests establish the implemented contracts and finite computations, not completion
of unrelated F04-F17 tasks. No parent card has been blanket-promoted.

## Remaining Execution Work

The numerical F03 omissions and legacy-target comparison are repaired. Original
card-level artifact packaging, prerequisite acceptance and the shared F00
orchestration still need integration; the old abbreviated F03 CLI is not the
canonical research runner. Use `run_grain_gptpro_f03_recovery.py` for these results.

Next recover F04's full source REML/BLUP, nonlinear four-cell reference and tail
analysis, followed by F05's learned state model. F01 primary-source coverage and
F14 real metadata inventory remain computational tasks, not human-side blockers.
