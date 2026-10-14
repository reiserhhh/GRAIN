# Active Reading: Pair Means, Instance Contrasts and Conditional Transport

HM38 completed a finite result-after diagnostic of the immutable HM37 bank.
Its purpose was to locate the missing function, not to fit a better predictor.
The existing source/application material is reused; all original requirements
remain open where this finite diagnostic did not address them.

## Finding and Decision

Source pair means are quite well reproduced: Simpson relative action errors
are1.03%-5.21%. Individual source wordings nevertheless retain8.65%-91.97%
relative errors. For the AI-trust query, coefficient-contrast squared moments
are99.86%/99.89% of the source wording-error second moment; pairing covariance
is only0.14%/0.11%. These are fixed computation diagnostics, not psychological
variance percentages. A covariance correction to the pair mean is insufficient
to reconstruct either wording.

Application errors additionally contain a transported-mean discrepancy.
For low mood its shared-transport RMS is98.56%-100.46% of total error RMS;
signed cross terms explain why a component can exceed the total. Trust retains
large instance coefficient contrast as well: its RMS is84.42%-87.03% of total
error RMS. Carrier truncation is material for negative affect, not universally
negligible. No term is automatically discarded as noise.

The next object is therefore the instance-conditioned covector field, not
another mean coefficient or a covariance-only correction. Before fitting that
field, separate output scoring curvature from decoder curvature. This asks
whether the observed instance dependence mostly belongs to the numerical meter
or to the internal representation computation.

## Exact Node Identity

For each fixed query, H arm, cell and node, keep two ordered wordings.
Let theta_w be the active coefficient per actual H-direction norm and eta_w
the correspondingly norm-scaled P displacement. The physical supported action
is theta_w^T eta_w. Let r be the frozen HM37 source full-law coefficient.
Define means tm, em and half differences dt, de. Then

    mean(supported) = tm^T em + dt^T de,
    mean(full-predicted) = mean(carrier) + (tm-r)^T em + dt^T de.

The per-wording identity is more informative:

    full_w - r^T eta_w
      = carrier_w + (tm-r)^T em + dt^T de
        + sigma_w dt^T em + sigma_w (tm-r)^T de,
    sigma_0=+1, sigma_2=-1.

The fourth term is the instance reading-coefficient contrast. It can be large
even when covariance dt^T de is small, or when most coefficient norm belongs
to the common component. Input directions determine which small coefficient
changes matter. The two-wording covariance rank<=1 is imposed by the design;
it does not identify a psychological factor.

Integrate these scalar terms at the same three nodes with the original HALF
Simpson weights(1,4,1)/12 or trapezoid weights(1,2,1)/8. Multiplying integrated
marginal vectors instead is a different computation. The node rank statement
does not imply that an integrated covariance matrix has rank<=1.

## Squared Error and Cancellation

For e=sum_i e_i, retain

    mean(e^2)=sum_i mean(e_i^2)+2 sum_{i<j} mean(e_i e_j).

Component RMS values cannot be added, and their ratios need not sum to one.
Source mean and contrast channels are orthogonal under the specified equal
pair measure, but the actual carrier contribution and application contrast
cross terms still remain. Pair-mean averaging can conceal a large individual
error; it is not an improvement in individual prediction.

## Verification and Reproduction

Canonical results: results/grain_math_response_v1/active_reading_pairing_r1.
Alternate centered-sum verification: active_reading_pairing_verification_r1.
48paired groups;288pair nodes/192pair half integrals;576wording nodes/
384wording half integrals;72moment groups.103output hashes and163068values
checked, maximum alternate difference2.221e-16. All10083inherited/code hashes
were checked. Analysis8.78+verification3.94=12.72/120seconds.0fits/0model calls.
10newtests and64math files/579tests pass; Ruff/format pass.

The frozen plan retains its pre-execution status as a historical artifact;
this companion and the parent Chinese report record the executed result.
Code and material hashes were not changed after preparation. A pre-freeze
unit-test fixture was corrected, and its original orthogonal-shift setup was
kept as a separate positive witness; no experiment was rerun or fitted.

## Reading Connection and Scope

The mean/covariance identity and the two-wording expansion are standard finite
algebra. Integrated Hessians retains derivative-input associations along a
specified baseline path, but its attribution integral is not this paired
wording diagnostic. Only its introduction and section2 through the beginning
of2.2.3 were read in this packet, not the54-page paper or appendix.
[Janizek, Sturmfels and Lee,2021](https://www.jmlr.org/papers/volume22/20-1223/20-1223.pdf).

OpenAI/math's fixed encoding manuscript retains ordered computation and
intermediate inverse domains even if an expression later cancels. This is an
inspiration for preserving original paired computation, not a theorem about
softmax readers or personality.
[Fixed encoding section](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Polynomial-Hitting-Lists-for-Noncommutative-Rational-Formulas-September-24-2026/build/sections/encoding.tex).

Actual application derivatives were used to diagnose the error. HM38 itself
is not a cold prediction law. Upstream neural states, supplied experimental
conditions, and the GRAIN author/event/reference/aggregation rules remain
distinct from the inferred reader function and from human trait/state scores.
