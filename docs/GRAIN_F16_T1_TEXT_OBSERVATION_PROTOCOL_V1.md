# F16-R02-LAW: T1 controlled-text observation contract

Status: implementation protocol, fixed before this packet's application scoring.
This is a follow-up using already exposed historical fixtures, not a new preregistration.
Original T1 and all other V3 obligations remain unchanged; this packet does not close F16.

## Object and observation law

The target remains Delta=m[+1]-m[-1], the four-cell F04 L response contrast.
Use all three archived seeds, 300 source and 200 application authors per seed,
32 events each, and all 200 archived source-bootstrap models. Do not refit them.

For eligible reports, Y=m[z]+epsilon, epsilon~N(0,sigma2), B=1[Y>=0].
The numeric unit and sigma2 come from independent numeric source observations,
not from the language model. This is a calibrated synthetic bridge, not a claim
that natural text identifies an absolute psychological unit.

Eligibility Q~Bernoulli(.5) is an independent randomized material-design variable.
For Q=0 the reported pole B~Bernoulli(.5) is independent of the author's Y.
For each B/Q stratum choose uniformly among corresponding archived text materials.
Actual and corrected-actual reports have Q=1; quoted and hypothetical reports Q=0.
Preserve the original event positions and conditions; no hidden Y/Q enters scoring.

Reader outputs are coarsened at a fixed .5 boundary into joint category
C=2*1[pole_readout>=.5]+1[qualification_readout>=.5]. Thresholds are not selected.
This retains a declared, limited observation alphabet instead of treating raw
scores as calibrated probabilities. Channel probabilities E[b,q,c] are estimated
on 32 validation/development texts, disjoint from the 32 texts that fitted the
historical GRAIN Ridge heads. Use independent Dirichlet(count+.5) rows, 200 draws;
all 32 application texts remain outside calibration. The small bank is not 600
independent natural-language tests; repeated materials do not increase its size.

For p_z=Phi(m[z]/sigma), the event likelihood is

L_c(m[z])=.5*((1-p_z)*E[0,1,c]+p_z*E[1,1,c])
          +.25*(E[0,0,c]+E[1,0,c]).

Joint categories retain pole/qualification dependence. A factorized-channel
control replaces E by the product of its pole and qualification marginals.
No assumption of independent readers is silently applied to the primary arm.

## Fixed comparisons

1. Calibrated joint NLI-pole/GRAIN-qualification channel (primary).
2. Calibrated joint NLI-pole/NLI-qualification channel.
3. Factorized NLI-pole/GRAIN-qualification channel (same draws and marginals).
4. Naive Gaussian surrogate: y=2*p-1, fractional precision q/sigma2, using the
   same primary reader outputs. Explicitly misspecified; never the main score.
5. Oracle B/Q categorical channel: hidden material metadata, reference only.
6. Matched eligible continuous Y: reference with strictly richer observed values.

Use the same author histories and source priors for all arms. Report MSE, bias,
Pearson r, interval coverage/width and paired author-bootstrap differences.
Do not choose an arm or threshold from application performance. Known source
spectral floors are inherited and counted, not hidden or retuned.

## Integration and reference

For each archived source model b pair one independent channel-posterior draw.
Condition the four-dimensional Gaussian prior using L; integrate with scrambled
Sobol prior-importance points, initially 1024 per component. Average the 200
conditional posteriors with equal weights, preserving the existing cut/bootstrap
source convention; do not call it a fully jointly fitted Bayesian posterior.
Save conditional variance, between-source-and-reader variance and quadrature ESS.
Moment-normal 95% intervals are an approximation; failed coverage stays failed.
Use a second independent scramble on the fixed first 12 application authors per
seed and primary arm, and repeat that comparison at 4096 points if score/SD
differences exceed .02. This is numerical resolution, not a prediction search.
Report any remaining discrepancy; do not silently upgrade the whole run.

Score the 300 source authors with source text frames to freeze the primary
reference mean/SD before application scoring. Reference scores are descriptive
source norms, not independent estimates of predictive validity. Cold processes
receive only a hashed bundle and text-event inputs, not source or target truth.
The bundle embeds pinned actual reader outputs/text addresses; it is not a new
raw-natural-text inference service. Check all 600 primary cold/warm objects.

## Eight evidence cards and safeguards

Two complete author histories, two missing-condition histories, two eligibility
composition changes, and two attribution-ineligible histories. Include alias,
same-event revision, retraction and summary checks without counting copies twice.
Cards show actual material addresses, joint categories, score/variance changes,
support and the target-relative interpretation. Eligibility remains latent to the
reader; an ineligible oracle card may still affect an imperfect real reader.

Tests: marginal likelihood by explicit latent-state summation; normalized
channels; independence counterexample; no-information channel; duplicate/future/
cross-author rejection; posterior quadrature against a one-dimensional integral;
calibration/application isolation; cold equality; scale nonidentifiability.
Scaling all m and sigma by c>0 preserves text likelihood but rescales Delta by c.
This is a constructive unit-identification boundary, not measurement failure.

## Mathematical sources

- Albert & Chib (1993), *Bayesian Analysis of Binary and Polychotomous Response
  Data*, JASA 88, 669-679, DOI 10.1080/01621459.1993.10476321:
  https://www.stat.cmu.edu/~brian/905-2009/all-papers/albert-chib-1993.pdf
  Supports a continuous normal latent response underlying a binary observation;
  our integration is not their Gibbs sampler.
- Stan User's Guide, *Latent Discrete Parameters*, noisy categorical measurement:
  https://mc-stan.org/docs/stan-users-guide/latent-discrete.html
  Supports explicit rater-error channels and marginalizing latent categories.
  We use a joint channel rather than assuming independent reader errors.

No neural training/API, private human text, MEPS, questionnaire fit, remote job,
lockbox, commit or publication. Archive failures and code/input hashes. Do not
turn this controlled text construction into new human personality accuracy.
