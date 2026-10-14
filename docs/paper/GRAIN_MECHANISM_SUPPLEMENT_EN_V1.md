# Supplement to the GRAIN Theory Manuscript: Mechanism-Round Results (Version 1)

## Addendum, 14 September 2026

The bounded follow-up queue has been executed. Under assignment-person independence,
assignment variance includes `p(1-p)[(dm)^2+dSd^T]`, not only the squared population
mean contrast. A source-estimated symmetric error mixture erased condition contrast
in this small source bank and was not adopted. Metadata-defined conditions supported
86.34% of held-author replies. After separating third-person claims from self-records,
73/400 natural pairs produced scope-guarded self-records; this is rule coverage, not
personality accuracy. Reply contexts do not identify the full exposure denominator.
See [execution report](../../reports/grain_measurement_foundation/CODEX_EXECUTABLE_GAP_CLOSEOUT_20260914.md).

Status: companion to `GRAIN_THEORY_MANUSCRIPT_EN_V1.md`. This supplement reports the
2026-09-13 mechanism round (M01–M14) as inherited methods, local findings, retained
counterexamples, and scope statements. It extends Sections 3, 4, 6 and 8 of the
manuscript and adds no human validity claim. Numbers are located in
`GRAIN_MECHANISM_EVIDENCE_PROVENANCE.md`.

## S1. Purpose and position

The manuscript separates technical structure, measurement comparability, construct
validity and use. The mechanism round asks a narrower question that the manuscript
left to future work: how a unit of evidence is formed from raw text and accumulated,
how a person's level and conditional response are scored by one function, how
observations are chosen for a declared target before the parameters are known, how a
score acquires a reference population and an error statement, how an explanation is
traced from text to number, and whether one construction method produces two local
instruments. All results were obtained on authored public materials and declared
simulated worlds; they concern the possibility and the failure conditions of
measurement construction, not the prevalence of any psychological mechanism.

## S2. Inherited methods and local correspondence

| Inherited method | Source | Local correspondence |
|---|---|---|
| Gaussian conditioning / linear minimum-mean-square readout | Rasmussen & Williams (2006) | Target kernel: person parameters and occasion states as the latent vector, event design as the observation matrix, reader noise as the error |
| Restricted maximum likelihood | Bates et al. (2015) | Population mean, covariance, state variance and recovery rate learned from independent pilot authors only |
| Continuous-time Ornstein–Uhlenbeck transition | Oravecz et al. (2009); Driver et al. (2017) | Occasion state on irregular clocks; window state as a trapezoidal weighting of sampled states |
| Instrumental variable / control function | Wooldridge (2015) | Recorded partner-input assignment as the instrument; common state kept as a separate term |
| Belief revision versus world update | Katsuno & Mendelzon (1992) | Correction of an earlier claim versus a new event; alias, revise, retract as ledger operations |
| Provenance semirings | Green, Karvounarakis & Tannen (2007) | Dependency graph of versions, retractions and referenced summaries; not a subtractable probability |
| Square-root estimation | Morf & Kailath (1975) | QR innovation form repairing numerical cancellation in incremental updates |
| Partial identification under bounded missingness | Manski-type bounds | Finite material mean range when unread values lie in [-1, 1] |
| Schur-complement information gain | standard | Which available report to read for a fixed target; exact duplicates gain zero |
| Derivative-matched function learning | Czarnecki et al. (2017) | Integrable scalar function of within-cell query covariance |
| Dependency syntax, factuality, dialogue dependence | Universal Dependencies; Saurí & Pustejovsky (2012); Bunt et al. (2020) | Frame channels, explicit reference rules, functional-dependence candidates |
| Measurement equivalence versus prediction equivalence | Millsap (1995) | Observation-unit consistency is not established by prediction quality |

None of these methods is claimed as new. The contribution is their connection to raw-text
evidence, source-learned parameters, target kernels and explanation rules, together with
finite witnesses of where each connection fails.

## S3. Local findings

| Finding | Result | Boundary |
|---|---|---|
| Concrete proposition binding retains more of the declared target than content similarity, but selective misreads change the denominator | Held-wording MSE .045369 (content), .018183 (binding), .011838 (all events); mean absolute profile shift .035705 | 192 simulated parameter groups |
| One utterance can update an earlier claim and enact a new speech act; the two accounts must be kept apart | 72/72 operations and 72/72 finite scores on 14 authored dialogues; MAE .958333 (all polarities), .875 (last polarity), 0 (versioned ledger) | explicit reference phrases only |
| New events, restatements, exact copies, corrections, retractions and referenced summaries require different updates | Level MSE .026027 / .035029 / .044062; coverage .9688 / .8750 / .7383 | 3 seeds, 768 application groups |
| Tail choices carry condition-specific information; high correlation can hide parameter error | Uncorrected r .917/.930 versus corrected .771/.796; parameter RMSE 1.47/1.53 versus .633/.640 | declared choice worlds |
| What is shareable is a source-learned response covariance, not a donor's answer | Crossed-world hidden contrast r .8414, MSE .044659 versus .156086; smooth world random slope .014255 versus .014327 | three declared worlds |
| Within-cell query covariance is a cold-computable observation that predicts finite output change | RMSE −33.3%/−25.1% at .02; −46.7%/−22.2% at .10 | frozen model logits, 8 historical authors |
| Same-event relations, not additional scores, restore the joint conditional reading | r .3578 paired, .0408 mispaired, .3594 perfect-event reference | post-result channel |
| Query-conditioned evidence computation lives mainly in question-region states | P share 73.5%–80.5%; other-person quotation retains 94.7% | 560 model calls |
| Query action can be compiled into text-region projection coefficients | State reconstruction and compiled projection agree to 1.55e-15; early C share 98.6% with amplitude .004379 versus native .288880 | 84 inputs |
| One evidence stream yields different readings under different target kernels | RMSE .30905 → .14319; window coverage .69271 → .96875 | known dynamics |
| Source-learned design beats coverage-first reading; resampling repairs a hidden direction | −25.9%/−30.0%; hidden-direction risk exactly Var(d) = .034133; .036596 → .010940 after 64 extra source authors | greedy oracle reference |
| An independent pilot learns population moments, recovery rate and noise; a conditional meter mixes level and response | Coverage .333/.557 → .969/.964; apparent rate .6748 → .2960 | 33 main fits |
| A traceable computational explanation is possible; a time-budget edit moves the query more than a request-pole edit | +.076500 (12/12) versus +.013705 (8/12) | authored stimuli |
| One method builds two cold-runnable instruments; current expression improves in error and ranking | RMSE .2446 → .1189, r .5575 → .6145; independent precision replay 3.77e-11 | 192 new simulated authors per instance |

## S4. Retained counterexamples

1. Identical content scores (.993314) with opposite evidence-identity targets (±.165707).
2. An abstract identity query fails (entailment .000406) where the concrete fact query succeeds (.998065); typed NLI r .166 versus TF-IDF .619 on held wording.
3. Refusal-proposition questions are misread with high confidence (3/24).
4. A local open-weight probe fails basic recognition (English 4/6, Japanese 2/6); its 24 counseling outputs are context-dependence evidence only.
5. Weak assignment inflates instrumental estimates (c RMSE 5.70895); an exclusion violation leaves bias +.296.
6. Reversing states within a cell preserves every moment and K yet changes the native output (16/16).
7. Two histories with identical current score and covariance answer the same correction differently ((.390625, −.390625) versus (0, 0)).
8. A response direction present only in a new condition is invisible to old data; its cost is exactly .034133.
9. Shrinkage lowers error while degrading ranking and coverage (social half-difference r .4898 → .2052, coverage .7656).
10. Third-person past reports are misread more often (4/16), shifting the person's denominator (maximum .152778).
11. On 12 licensed natural sentences no declared target is computable (0/48) because no antecedent question exists; 273 natural event addresses qualify for neither controlled instrument.
12. The original incremental update helper cancels numerically (31/282, maximum 7.88e-7); the square-root companion reduces it to 5.86e-12 without altering frozen results.

## S5. Scope

- Simulated parameter groups are recoveries within declared worlds, not human correlations or behavioral validity.
- Authored public sentence banks are mechanism materials; repeated analyses of one bank are analytical recurrence.
- Readers are fixed (`cross-encoder/nli-deberta-v3-base`, pinned revision; a frozen prefix model); changing a reader requires recalibration.
- Dialogue rules operate only inside explicit reference phrases and declared two-party roles; open ellipsis, modality and coreference remain unresolved.
- Working intervals combine working-model variance and finite source-construction variance; they are not psychological 95% confidence intervals, and common units do not establish invariance.
- Counseling materials were used for validation only; no fitting, parameter selection or export.
- Same-agent replay and cross-agent rebuilding with development context are not blind third-party replication.
- Psychological or behavioral meaning requires independent coding and criteria; no score names itself.

## S6. Additional falsifiable propositions

**Proposition 9: Evidence identity is not recoverable from content similarity.** Two
expression distributions can share a content score while differing in a declared
identity target. The proposition fails if, for a declared target family, a content-only
reader reproduces identity targets within a declared tolerance on held wording.

**Proposition 10: A current score and covariance do not determine the next update.**
Two evidence histories with identical current state can require different revisions
after the same explicit correction. The proposition fails if a fixed-size summary is
shown to reproduce every registered correction operation for the target family.

**Proposition 11: Observation design cannot substitute for identifiability of the
target response.** Optimal reading of old-condition reports leaves a response direction
that only appears in a new condition invisible; the loss equals the variance of that
direction. The proposition fails if additional old-condition observations reduce that
loss without new-condition source evidence.

## S7. What changes in the manuscript

- Sections 3 and 4: the target kernel makes "define the intended inference before
  modeling" operational; source-learned parameters, versioned evidence and finite working
  intervals are the construction-side objects.
- Section 5: no new human-validity row is added; the mechanism results form a separate
  table with the scope statements above.
- Section 6: Propositions 9–11 are added.
- Section 8: the sixth limitation stands; two dependencies are made explicit — independent
  coding and criteria for the new dialogue objects, and a blind third-party construction.
- The nine claims that must remain identical are unchanged; three are added: simulated
  recovery is not human prevalence or validity; explicit rules are not a learned universal
  grammar; a cross-agent rebuild with development context is not a blind replication.
