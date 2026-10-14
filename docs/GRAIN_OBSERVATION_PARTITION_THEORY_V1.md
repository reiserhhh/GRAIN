# Source measurement under provenance-preserving observation programs

HM205,2026-10-12. Construction and finite numeric experiment complete.
Original18/100/44 remain OPEN. No new Gaussian/sufficiency theorem claimed.

## 1. Separate evidence from its presentation

The same canonical event family X can be displayed as individual vectors,
disjoint averages, overlapping averages or repeated summary rows. These are
observation programs A,not newly independent events. A new row identifier does
not turn reused content into a new observation. Keep canonical IDs and A.
Our operation is linear averaging of existing E5 vectors,not re-encoding
newly cut text. Token-level segmentation can change the encoder and is a
different operation; this construction does not prove invariance to it.

Declare a root location experiment conditional on SOURCE identity U:

    X_t=z_U+S+eps_t, Q=sigma2 I+tau2 11'.

S is a common frame fluctuation; its psychological interpretation is unknown.
It is not removed from X,called noise,or estimated as a state/trait fraction.
The Gaussian law and isotropic vector coordinates are stipulated. A general
natural event law need not be Gaussian,stationary,or location-only.

For G=A X,C=A Q A',c=A1, the singular Gaussian log-likelihood has source part

    -1/2 tr[(G-c z_i')' C^+ (G-c z_i')]
    = source_common_term + z_i' G' C^+c - (c'C^+c)||z_i||^2/2.

Therefore alpha=c'C^+c,m=G'C^+c/alpha is enough for THIS location reader.
The prototype weights are proportional to exp(-alpha||m-z_i||^2/2).
All five targets and all32 profiles share these weights. No q-conditioned
weights need to be invented to repair a grouping artifact.

The residual that cancels in this likelihood is not universally irrelevant
text. It can carry path,relation,choice,state and target information under
another observation law. Nothing is deleted from the retained input archive.

## 2. Why lossless regrouping agrees

Here the root likelihood depends on mean(X) and

    alpha_root=T/(sigma2+T tau2), variance_mean=sigma2/T+tau2.

If the root all-ones direction is in the row space of A,that mean is recoverable
from G,so the grouped and root target posteriors agree. This is a sufficiency
statement for the specified location parameter,not losslessness for the full
text or every psychological function. Overlap creates a singular covariance;
the generalized inverse operates on its support. Impossible alias rows are
rejected,not projected to make the audit pass.

If only a prefix is observed,the location information changes. The agreement
requirement must NOT forbid this legitimate measurement change. Similarly,
observing genuinely new independent events can increase information. Repeating
an old average or slicing its presentation cannot do that by itself.

As T grows,alpha_root approaches1/tau2 when tau2>0,while it grows without bound
in the independent world. This standard random-effect limit is not evidence
that all natural text has a floor,or that social-scale error is identified.
It explains why declared opportunity/common-frame dependence belongs in a
measurement recipe rather than being eliminated by counting more chunks.

## 3. Fixed empirical construction

SOURCE386/two prototype banks,24 historical disjoint application authors,
576 canonical cached comments represented as cue-mask/raw. The480 frames are
reused subsets,NOT480 people. No application labels,raw text or LLM answers.

The independent world uses sigma2=12s2,tau2=0. The common-occasion world uses
sigma2=6s2,tau2=s2/2,where s2=.05525111588037168 is inherited from HM204.
Both exactly match the old12-event mean law. This half-shared decomposition is
a designed competitor,NOTan empirical covariance estimate. For T4/8 it differs.
Do not infer which world is natural from these calculated differences alone.

Each frame/world/reader/source-bank was read under atomic,pooled,balanced,
unequal,overlap-with-global,aliases and lossy-prefix programs. The diagonal
competitor keeps every marginal variance but discards off-diagonal dependence.
26,880 grouped readings/134,400 target values,not independent experiments.

Lossless joint readings agree to1.887379e-15. In the independent world all
disjoint partitions also agree with the diagonal reader,as predicted. Overlap
or aliases break the diagonal account even there. In the shared world,even
disjoint atomic/partition rows are dependent and the diagonal reader changes.

| World/program,diagonal vs same root joint | Five-target RMS range | Maximum difference | Relative sign reversals/4800 |
| --- | ---: | ---: | ---: |
| independent/overlap | .002167-.005002 | .011522 |129 |
| independent/aliases | .001113-.002377 | .005945 |321 |
| shared/unequal | .000259-.000575 | .001747 |54 |
| shared/overlap | .003067-.006698 | .014486 |189 |
| shared/aliases | .003024-.006398 | .018296 |351 |

The signs refer to departure from the inherited SOURCE label prior. Departures
can be small; these counts are not substantial psychological changes or
human prediction errors. Max1.83 percentage points is not a clinical effect.
All five targets,source banks,frames and signs remain in partition_effects.csv.

Independent versus shared worlds agree on every full12-event frame. Their
atomic reading RMS difference is .000805 atT4 and .000593 atT8. This separates
same-evidence re-expression from changing the observation law/exposure. A
single full-frame calibration cannot select between these worlds.

## 4. Mainline consequence and the next question

Use a measurement contract of the form

    reading_q = M_q(reference,qualified root evidence,observation law,A).

If A only expresses the same sufficient observation differently,require
M_q^A(A X)=M_q^I(X). Do not enforce this when the observation removes relevant
evidence or the encoder changes. A posterior update can change for different
targets under one common evidence law;that alone does not locate independent
personality circuits or prove q-conditioned organization is essential.

The added ability is a tested provenance-to-covariance grouping compiler,not
more agreement with neural answers. The distinction needed next is a source-
supported conditional RELATION beyond this location-only law. Use actual same-
unit condition/event interfaces and source anchors; do not fit tau2/bandwidth/
chunk size until a mechanism decision demonstrably requires that estimate.
HM192/193 relation results remain valid at their own support;do not falsely
join their17-event windows to these12-event frames. Preserve their choices and
all negative results. Neural target organization and psychological meaning
remain separate links,not repaired by this constructor or by higher r.

## 5. Prior art and execution

[D'Agostini](https://www.roma1.infn.it/~dagos/curious/node2.html) explicitly derives
Gaussian grouped-average sufficiency;[Tardiff et al.,2025 indexed primary record](https://pubmed.ncbi.nlm.nih.gov/40657889/)
connects correlations and evidence weight in human perceptual decisions. The
indexed record/figure description were read,publisher full text challenged;
do not claim a replication or all-methods review. [Vehtari et al.,2020](https://jmlr.org/papers/v21/18-817.html)
addresses partitioned Bayesian inference;this is not an EP implementation.
OpenAI/math adc7f124 posterior foundations inspire explicitly specifying the
observation family. Tree/spin reconstruction theorems are not applied.

Canonical results/grain_math_response_v1/observation_partition_r1. Prepare,
compute,verify each once,1.982451s summed in-stage elapsed,0neural/API/training.
14new/28focused tests,3Python lint+format PASS preseal. One same-agent audit:
28 literal quadratic likelihoods,max1.22e-15; separate whitened-SVD information
route,max8.88e-14;old full-frame replay1.44e-15. Not a blind independent audit
or a request to improve precision. All artifacts and old seals preserved.
