# Normalization order: a concrete candidate behind the remaining head displacement

2026-09-08. Mathematical prework while S25 collects, not a new source result.
The exact S26 remainder combines mean value reading, changed attention masses
and normalization. This note identifies one testable mechanism within it rather
than labelling that remainder as noise.

Let P average token vectors within each comment and preserve other positions.
Let R(x)=D_gamma x/sqrt(x^T x/d+epsilon) be the actual tokenwise RMSNorm map,
with epsilon>0. The normalization definition follows
[Zhang & Sennrich2019, Eq4](https://papers.nips.cc/paper_files/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf),
with the implementation's numerical epsilon. This is not a psychological model.

P is a linear idempotent within-comment averaging projection. The nonlinear
composition difference J(H)=R(PH)-P(RH) obeys PJ(H)=J(H), and vanishes outside
the averaged groups. Thus within-comment dispersion can become a group-mean
shift after normalization. Preserving a mean at one decoder boundary does not
preserve it after the next nonlinear operation; this does not invalidate the
earlier mean-preserving intervention, but explains one possible transmitted path.

For a group with mean mu, centered deviations delta and covariance Sigma,
let s=(mu^T mu/d+epsilon)^(-1/2). A direct Hessian contraction gives

J_G approximately equals
D_gamma [ s^3/(2d) (2 Sigma mu + tr(Sigma) mu)
           - 3s^5/(2d^2) (mu^T Sigma mu) mu ].

This is our derivative of the cited normalization equation, not a new theorem
claimed from that paper. The approximation is second order in local deviations.
The remainder is controlled by third derivatives and third absolute moments;
for a fixed symmetric finite deviation set, odd terms cancel and small-scale
error is fourth order. Actual text deviations can be large, so use exact J
first and test the quadratic approximation rather than assuming it is accurate.

This yields a future three-way local computation from the same cached boundary:
native normalized tokens R(H), mean-normalized tokens P(R(H)), and normalized
means R(PH). Reconstruct their attention heads with the same final query.
The two successive differences add to the already measured configuration head
contrast. Their interpretation depends on the declared order; interactions with
attention weights are retained rather than called independent trait components.

Finite tests verify group-mean support, zero effect for constant groups, agreement
of the closed Hessian formula with autograd, and fourth-order small-scale error
on a symmetric example. These tests do not establish its size or source N value
in real text. Run a bounded real-prefix extension only after S25/S26 indicate
whether decomposing the remaining configuration channel changes the mainline
decision. Do not restart normalization/embedding grids.
