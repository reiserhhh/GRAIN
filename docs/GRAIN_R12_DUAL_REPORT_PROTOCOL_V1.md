# GRAIN R12 dual-report rule applied to intervention results, protocol V1

Frozen 2026-09-26 JST before computation. Cached results only, no neural calls.

Source rule (P03 pp.7, 31; P01 p.21; P02 p.13): every intervention result reports the
mean effect AND the reliability of individual (unit-level) effect differences.
Mechanism fidelity and individual reliability are separate metrics.

- Datasets with replicate structure:
  1. F10 PV01, conditional_operator_v1/r2 application pairs. The unit is the cell
     (axis x family x situation); the replicates are 3 seeds. Effects: donor, compiled
     and random actions and the full-text contrast, for E and N.
  2. F16/R16 reader interventions. The unit is the template; the replicates are the 2
     matched pairs per intervention. Effects: NLI and GRAIN qualification and pole
     readers under pole and qualification flips.
  3. F12 DIF table. The unit is the condition z (4 levels); the replicates are 3 seeds.
     The effect is DIF bias, per method x world.
- Reported per effect:
  - mean effect with a 95% unit-bootstrap CI (2,000 draws, seed 20260926);
  - ICC(1) = (MSB - MSW) / (MSB + (k-1) MSW) with a unit-bootstrap CI;
  - the reliability of a k-replicate unit mean, ICC(k) = (MSB - MSW) / MSB.
- Cited, not recomputed:
  - F09 rival person simulation (mean estimate and retest r per mechanism);
  - F04 excitation_reliability_r2 (reliability against tokens and diversity).
- Not estimable: R17 natural edits, with one unit per author and no replicate.
- Rule: an effect is individually reliable when the ICC(1) lower bound is above .10.
  A mean effect with an unreliable individual component is reported as
  "population effect only".
