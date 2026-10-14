# F08 polarity versus evidence qualification

## Finding

In the fixed NLI mixed-material experiment, almost all of the XOR prediction degradation comes from evidence qualification rather than the stance reading. This is an actual input substitution experiment on identical histories, not an inference from two unmatched panels.

Four cells use the same observations and frozen source B1/B2/M: L00 has material-truth polarity and qualification; L10 substitutes reader polarity only; L01 substitutes reader qualification only; L11 substitutes both. "Correct polarity" means the stance expressed by the material, including the other speaker's stance in quoted text. It does not substitute hidden author truth for an ineligible report.

The exact loss decomposition is

L11-L00 = (L10-L00) + (L01-L00) + (L11-L10-L01+L00).

This is a declared baseline-dependent factorial decomposition, not a unique causal mediation decomposition of human behavior.

## XOR Brier changes, 600 application authors

| Estimator | Polarity alone | Qualification alone | Interaction | Total |
|---|---:|---:|---:|---:|
| B1 | -.00000224 | +.02314909 | -.00000732 | +.02313953 |
| B2 | -.00010657 | +.03792690 | +.00002632 | +.03784665 |

B1 qualification-only difference has paired95% interval [.020935,.025267], whereas polarity-only has [-.00001262,.00000747]. B2 qualification-only interval is [.034120,.041926]. Small statistically resolved effects, such as B2's polarity shrinkage, are retained without claiming they rival the much larger qualification contribution. All candidates, both worlds and both proper losses are saved.

The practical theoretical interface is therefore not simply "semantic response -> author parameter." It is "semantic response + ownership/actuality qualification -> qualified history -> target-specific scoring." A strong semantic reader can fail at that second interface. This finding supports separating channels, not retuning an already successful stance detector or claiming all language models share this failure.

## Execution and verification

No new text, NLI calls, prompts or fitted heads were used. The two existing cells were reused from the audited NLI run; only two missing cells were computed on the frozen mixed observations. Full output:172,800 updates,648,000 new prediction rows,18,000 new author metric rows,18,000 decomposition rows and80 paired contrasts. Three seeds retain source300/validation100/application200 in each of the two worlds. Bootstrap resamples application authors within seed2,000 times; it does not measure new-material or reader-training uncertainty.

The alternate enumerated-state/augmented-least-squares audit reproduces all648,000 scores, maximum difference1.02e-14. All18,000 author metrics and decompositions replay;80 intervals replay via multiplicity weights, max2.78e-17. Same-agent numerical replication is not third-party replication.

Artifacts: `F08/factorial_full_r1` and `F08/factorial_audit_r1` under `results/grain_gptpro_foundation_v3`. Scripts are `run_grain_gptpro_f08_factorial.py` and `audit_grain_gptpro_f08_factorial.py`. Existing data and failed revisions remain unchanged.

## Next scope

Stop widening the F08 prompt/reader search: the finite channel failure is localized. Standard card/DAG packaging and final production rules remain. Move substantive work to F09's rival-process equivalence and out-of-family detection. Its old belief law was merely direct logistic; the repaired module now models the two-stage belief/report law,18 candidate atoms with12 observational classes, and parity-separated design corners. Four tests establish those identities; full F09 simulation and calibrated detection have not yet run.

Global goal remains active. No API, new neural training, private data, remote execution, lockbox, commit or push; no background job remains from this packet.
