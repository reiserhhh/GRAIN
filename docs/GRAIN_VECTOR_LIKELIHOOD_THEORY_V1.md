# Source-supported positive vector observation and target fields

2026-10-12, HM204. Construction complete; original18/100/44 remain OPEN.
This is a standard finite Gaussian-mixture observation law, not a new theorem.
Its utility is an explicit frozen numeric reader and a clear boundary between
source association and target-conditioned neural evidence organization.

## 1. Defined measurement, not an assumed human posterior

Let SOURCE author i have a fixed mean vector z_i and inherited binary target
profile Y_i in E/N/A/C/O order. Declare an experiment:

```text
U ~ uniform SOURCE
X | U=i ~ Normal(z_i, s2 Identity)
w_i(x) = exp(-||x-z_i||^2/(2s2)) / sum_j exp(-||x-z_j||^2/(2s2))
psi_q(x) = sum_i w_i(x) Y_iq
relative_q(x) = psi_q(x) - mean_SOURCE Y_q
joint_profile_b(x) = sum_{i:Y_i=b} w_i(x)
```

The last line retains the joint distribution over 32 profiles. Five marginal
outputs are not independent evidence or five separately learned circuits.
Every observation law has the same frozen source reference. Two source-window
laws use the same variance, but are not two independent human studies.

The Gaussian observation channel is stipulated. The posterior is exact under
that channel; neither natural text error nor human scale-probability calibration
is established by stipulating it. Context selection, state and author habits
remain inside the vector observations, not classified or removed as noise.
Historical E5/GRAIN encoder/development overlap is retained. The 24 application
authors are source-disjoint but are not a fresh lockbox.

## 2. Shared observation and target-indexed reading

Write zbar_w=sum_i w_i z_i. Direct differentiation yields:

```text
grad log p(x) = (zbar_w-x)/s2
grad psi_q(x) = Cov_w(Y_q,z)/s2
grad log odds_q(x) = (E_w[z | Y_q=1]-E_w[z | Y_q=0])/s2
grad psi_q = psi_q(1-psi_q) grad log odds_q
```

Thus a common observation law supplies distinct target fields through its
source target linkage. The shared density gradient is not a personality field.
This is Gaussian score/Bayes algebra, related to Tweedie's formula; these
identities alone do not constitute GRAIN's novelty.

An equivalent expression clarifies what this constructor actually does:

```text
w_i(x) proportional to exp(x.z_i/s2 - ||z_i||^2/(2s2))
```

The common ||x||^2 term cancels in posterior normalization. That cancellation
does not mean the corresponding natural-text content is noise. Likewise, any
variation outside the prototype span is invisible to this declared reader,
not thereby psychologically irrelevant. A source-dependent finite feature
model can miss information.

Here x is an identified subset mean. IDs prevent a duplicated event from
being counted twice within a record, but the mean does not preserve its full
ordered path, occasion, syntax or opportunity structure. There is no product
of independent comment likelihoods and no claim that comments are independent.

## 3. Executed construction and mathematical finding

386 source authors x2 windows x12 comment vectors, all1024 dimensions; 24
disjoint historical application authors, cue-mask/raw,2 windows,5 subsets.
Only inherited source labels are read. No application labels or LLM answers
enter compilation or cold scoring. One label-free median pair-distance variance
is s2=.05525111588037168. No bandwidth, rank, metric or answer-fitting search.

The local field/global unweighted class-contrast mean cosines are
.983356..990094. Relative field differences RMS are .139512..183345; finite
log-odds actions differ by .089654..200136 of local-action RMS. All targets,
windows and subsets are retained. Local reweighting has an effect but on this
support it mostly preserves global directions; it is not an independent new
nonlinear mechanism merely because the formula is nonlinear.

Effective source counts are366.241..381.337 out of386. The kernel reader is
near-global at this fixed scale. Posterior outputs stay within .465789..510479
across all five targets/windows/representations/subsets. Their relative RMS is
.002045..008046. This compresses probability-scale output differences. Neither
small amplitudes nor high normalized correlations alone determine measurement
power; a scale can be re-expressed without changing author ordering. What is
missing here is psychological calibration, not proof that small numbers contain
little information.

Descriptive matched deletion contacts with existing native model score actions:

| Target | Source window0 | Source window1 |
| --- | ---: | ---: |
| E | .022848 | .275305 |
| N | .436785 | .422722 |
| A | .685583 | .465214 |
| C | .118891 | -.114038 |
| O | .187956 | .240538 |

These are correlations between TWO text-operation readouts on the same24
authors, not human trait prediction r. Historical held-author N/window1 is
-.123033. A after one full-profile permutation remains .268929/.279487,
versus .685583/.465214 before permutation. A single permutation neither proves
specificity nor supplies a significance test; all null and negative contacts
are preserved in all_native_contacts.csv. No target or source window is promoted.

## 4. What the constructor does and what the next mechanism must add

We now have a positive joint law, normalized all-target cold readings and their
local gradients. This is more usable than calling a signed covariance witness
a probability. It also exposes a mainline distinction: in this reader w_i(x)
does not depend on query q. Target variation enters via the frozen Y_iq only.
That is NOT a mathematical defect or proof that target-dependent weights are
necessary. For any function f of the declared joint target profile,

```text
E[f(Y) | X=x] = sum_b joint_profile_b(x) f(b)
```

The common 32-profile posterior therefore suffices for every target-profile
function UNDER THIS LAW. It is not established as sufficient for natural text,
all human characteristics or the model's response. This separates a common
observation plus different readers from a genuinely target-specific observation.

HM200-203 experimentally changed how the neural model organizes or reads the
SAME evidence by target. HM204 has not compiled that process. Its native contacts
therefore cannot be used as proof that its Gaussian weights implement the neural
mechanism. A successful source classifier and a recovered neural evidence rule
are different research objects.

Next inventory existing HM168/182/184/192-199 conditional, event and native
interfaces. The competing constructions are a sufficiently rich common
observation with target-indexed readers and target-conditioned evidence
organization required by a compact observation. Neither is assumed correct.
Construct only the genuinely missing source-supported link on the same units;
preserve choice, state, complete joint profiles and the source reference. Do not
force query dependence, tune bandwidth or rescue weak targets. The next plan
must identify an observation that distinguishes these accounts and a new
construction ability, not demand better r or repeat PSR/Bayes/kernel-mean theory.

Cold scoring is numerical AFTER encoder vectors exist. E5 remains neural; this
is not a fully LLM/neural-free raw-text method.

## 5. Prior art and provenance

[Efron2011, Tweedie's Formula and Selection Bias](https://statistics.stanford.edu/technical-reports/tweedies-formula-and-selection-bias)
is Gaussian posterior-score background. This turn checked the institutional
record; no new fulltext-reading claim follows from that record.
[Kernel Bayes' Rule2013](https://jmlr.org/papers/v14/fukumizu13a.html) is related
inference literature; HM204 is an explicit finite-mixture posterior, NOT its
RKHS inverse estimator or a replication of its consistency results.
[OpenAI/math foundations](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
motivated making an observation experiment explicit. The spin/tree independence,
degradation and reconstruction theorem are not transferred to text embeddings.

Canonical results/grain_math_response_v1/vector_likelihood_compiler_r1.
Five stages executed once,7.137236s total measured in-stage CPU/process elapsed;
zero new neural calls/API/training/private inputs/Git/Claude/automation/lockbox.
14 focused tests/4 Python lint+format PASS before sealing. One same-agent
literal-mixture and signed-account audit:32 point checks,960 cold window/frame
outputs,6000 contacts,maximum discrepancy1.387779e-15. Not independent blind
replication or6000 independent participants. Stage/source seals remain unchanged.
