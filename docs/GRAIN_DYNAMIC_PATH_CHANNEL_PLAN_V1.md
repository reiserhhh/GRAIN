# GRAIN dynamic paths and measurement-channel differences

Date: 2026-09-10. Status: bounded exploratory analysis, not a new lockbox.

## Questions and executable decisions

1. Do within-author changes in frozen GRAIN coordinates track changes in the
   fixed personality-query function? Restore timestamps/subreddits only after
   exact masked-text matching to the existing 276-author source panel.
2. Does temporal order carry information beyond static location and dispersion?
   Score four nonoverlapping three-comment blocks in each 12-comment view with
   the frozen whole/native factor bundles. Use drift, permutation-normalized
   roughness, and within-operator oriented areas. Keep dispersion separate.
3. Are path-linked factors more criterion-linked? Compare the same coordinate
   across query-function changes and actual questionnaires, rather than combining
   an interesting model direction with an unrelated trait correlation.
4. Can context, query information or dynamics reduce the text-to-questionnaire
   gap? Run the eight combinations of these three additions to the existing
   12-coordinate baseline. Five outer author folds, three inner folds, train-only
   scaling/Ridge selection. Account for incremental MSE by exact block Shapley
   values; retain negative contributions and all nine outcomes.
5. What can be said about expression versus action? Inventory existing MEPS
   speaker/turn metadata and already-authorized scored questionnaires. Only
   descriptive external association, no MEPS fitting or tuning. Conversation
   participation is observed behavior, but it is not an independent everyday
   action criterion. Do not read raw surveys, coding duplicates or analysis-master
   tables by default.

## Mathematical objects

Let x[t,k] be a frozen GRAIN coordinate for a three-comment block. These are
sampled, chronologically ordered Reddit comments from different discussions,
not consecutive conversational turns or equal-duration observations.

For each four-point path, report coordinate standard deviation s[k], endpoint
drift (x[T-1,k]-x[0,k])/(T-1), and roughness R[k]=mean(diff(x[:,k])**2).
Under uniform permutation of the same T points,
E[R_perm[k]]=2*T/(T-1)*var(x[:,k], ddof=0).
The normalized order contrast is R/E[R_perm]-1; constant paths score zero.
Center and standardize each coordinate within path to form z. For coordinate
pairs within each operator, oriented area is
mean(z[:-1,j]*z[1:,k]-z[:-1,k]*z[1:,j]). It reverses sign under time reversal.
These are short-path descriptors, not fitted personal dynamical systems.

The existing three period-level P/Q packets define f_q(P,Q), not attention-head
causal flow. Test corr(diff(G), diff(f_q)) with author-cluster intervals and
whole-author permutation; partial out observed context differences descriptively.
As D(G+c_author)=DG, this test cannot replace the between-author level test.

For a set S of extra feature blocks, let M(S) be author-held MSE. The value of
block j is the average over all three-block orders of
M(predecessors)-M(predecessors+j). These sum exactly to M(empty)-M(all).
They allocate predictive gain, not true psychological variance or causation.

## Fixed comparisons and resource boundaries

- Same 276 authors: 213 Big5-only, 63 MBTI-only; historical reference exclusion
  uses the verified raw-name-to-pseudonym mapping from the preceding run.
- No new neural model-weight training, checkpoint changes, new factor discovery, or paid calls.
  Cached model states only. The frozen 12 factors are a bounded bridge to GRAIN,
  not every dynamic object ever developed in the repository.
- Report every criterion, paired author bootstrap, within-fold diagnostics for
  imbalanced MBTI, two fixed negative-control label permutations, and an
  order-shuffled feature comparator. Do not select winners on MEPS.
- Stop after this finite matrix and channel inventory. No automatic search for
  an accuracy threshold. Follow-up only if a specific new mechanism is indicated.
- Save numeric individual data under ignored results; commit only code, tests,
  plan, aggregate report and navigation updates. Leave existing dirty files alone.

## Prior-work alignment

- [Fleeson (2001)](https://pubmed.ncbi.nlm.nih.gov/11414368/): state-distribution
  means and variability are different potential individual-difference objects.
- [Campbell and Fiske (1959)](https://doi.org/10.1037/h0046016): distinguish
  construct covariance from method covariance. We do not claim an identified
  multitrait-multimethod model from text plus a single self-report channel.
- [Jain and Wallace (2019)](https://aclanthology.org/N19-1357/): attention weights
  alone do not settle explanatory importance. Cached P/Q co-change is a relation
  between representations; the prior native intervention remains separate.

## Deliverables

- `scripts/run_grain_dynamic_path_channels.py` and tested numeric helpers.
- `results/grain_vector_mechanisms/dynamic_path_channels_v1/`: input hashes,
  restoration audit, private trajectories, all-query co-change and criterion
  tables, held-author predictions, paired comparisons, gain accounting,
  negative controls and MEPS channel inventory.
- `reports/grain_vector_mechanisms/DYNAMIC_PATH_CHANNEL_REPORT.md`: answers to
  the five questions, numerical conclusions, identifiable versus unidentifiable
  channel differences, and one next mainline decision.

## Material-driven clarification before within-comment analysis

The restoration audit found a median 1,069-day author span and 12.6-day gap
between sampled comments. Therefore the first analysis is explicitly the
between-comment/period branch, not within-utterance flow. Add a second fixed
branch: each eligible comment is split into four contiguous equal-count lexical
token spans (minimum 12 tokens/span); score the four frozen whole coordinates,
then aggregate per-comment order descriptors to the author. No date interval is
required for this branch. Keep source coverage and slice-level mean/dispersion
as comparator features so that a gain is not merely a new observation scale.
Test exact reversal and all 24 possible point orders numerically. Keep the
within-comment branch separate from sampled-comment results.

The first 499-permutation run has a p-value floor .002, too coarse for 108
primary links. Increase resolution to 4,999 permutations for every primary
total-score query/factor link, raw and adjusted, without selecting cells or
changing authors. Intermediate function-part links remain secondary (499).

## Bounded native follow-up after source analysis

The 276-author result selects whole:F02/T-F as the strongest primary co-change
candidate. Before any new native outcomes, fix one intervention: source-fit a
direction from standardized within-author changes in whole:F02 to changes in
the concatenated P/Q state (Ridge alpha=100; remove common period-transition
means). Fit one whole-author-permuted direction and match its norm. On the same
12 reference/source-disjoint native authors from the preceding intervention
packet, replay plus/minus 0.25 direction steps. Exactly 12 captures + 48 suffix
calls, maximum 600 seconds, local checkpoint only. No factor search, new human
labels, or model-weight training. This tests a model-direction effect, not the
claim that a real person's personality changed. Selection is exploratory and
the native author panel is reused, not a newly opened blind cohort.
