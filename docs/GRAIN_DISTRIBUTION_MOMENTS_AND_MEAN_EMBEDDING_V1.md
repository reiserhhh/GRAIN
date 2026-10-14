# Distribution Moments and Mean Embeddings in GRAIN

2026-10-09. HM62 finite empirical update. Parent goal active; original scope open.
No new model calls, fitting, private data or neural training.

## Three Distinct Objects

1. A distribution p over qualified evidence configurations.
2. A target-conditioned observation R_q(x) of a configuration.
3. A retained summary mu(p), potentially a full nonlinear mean embedding.

An averaged observation is s_q(p) = sum_x p(x) R_q(x). This is NOT a new model
judgment on concatenated author history, nor a human personality ground truth.
On the six-bit design, the standard Fourier identity gives

    s_q(p) = sum_S a_qS m_S(p),
    m_S(p) = E_p product_(j in S) X_j.

With measured moments A p = mu and an interior reference probability, a scalar
target is identified iff R_q belongs to the row space of A. Otherwise a target-
active null direction of A gives two valid local probabilities with the same
summary but different target values. Support restrictions matter at boundaries.
This is elementary finite identification, not a new mathematical theorem.

## Actual Findings

HM62 reuses two families of 64 frozen HM61 configurations and six target queries.
All 63 nonempty parity subsets receive full-support paired profiles. Fixed
orders1/2/3/6 define 96 sharp intervals, evaluated with 192 LPs and saved primal/
dual witnesses. Family0 witnesses were sealed before family1 analysis, but all
raw observations were already available: this is result-after analysis.

Score interval widths, mean / maximum across queries:

| Fixed information | Family0 | Family1 |
| --- | ---: | ---: |
| First moments | .074422 / .167552 | .061457 / .105378 |
| Through second order | .011837 / .018395 | .009827 / .014065 |
| Through third order | .004123 / .008795 | .003144 / .005419 |
| Full moments | numerical zero | numerical zero |

These are attainable ranges, not sampling confidence intervals. Matching means
and covariances does not always match the actual target response. Source witness
directions can reverse in the second family; formula transport is not automatic
coefficient transport.

## Correcting the Averaging Claim

The full body encoder can already represent nonlinear combinations. Let C contain
its 64 Fourier coefficient rows in 1536 hidden coordinates. Then

    mean_h(p) = c0 + m_nonconstant(p) @ C_nonconstant.

Both families have nonconstant row rank63 under the frozen machine tolerance.
The separate result-after audit recovers all 252 tested probabilities from their
full vector means (maximum error3.997e-14). Condition numbers2324.41/1206.03
and inverse norms511.74/234.81 distinguish exact retention from stable estimation.

Thus six behavior means lose information that these full hidden-vector means
retain. Neither finite rank nor this recovery proves characteristicness on all
natural text, psychological interpretation, or an LLM-free new-text encoder.
The count63 is not a psychological factor count. The right inverse is a finite
diagnostic, not a deployable new-author predictor.

## Construction Implication and Research Decision

The next useful object is target-relevant information retained by GRAIN's actual
compressed author representation, with choice/condition/state kept distinct.
Do not delete common modes or low-variance directions by default. Do not demand
recovery of all encoder information when only particular targets are measured.

Observer repair is authorized as mainline work only when it resolves a named
mechanism ambiguity and changes a construction decision. Teacher fidelity,
observer accuracy and human measurement validity are separate estimands.
HM45 fidelity remains deferred; HM59 template diagnostics remain closed.

Sources: Bertsimas/Popescu2005 introduction/Definitions1.1--1.2 (moment-feasible
sets); Sriperumbudur et al.2010 JMLR abstract (mean embedding identification).
OpenAI/math pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a Gaussian-regression
endpoint lines1--190 supplied residual-fiber design inspiration, not an imported
Gaussian/Haar/finite-memory theorem. All source and method boundaries are listed
in the parent HM62 report. Human psychological meaning and original scope remain
open; numerical verifiers are same-agent alternate arithmetic, not independent
scientific replication.
