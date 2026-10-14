# Author Profiles and Observation Change

2026-10-09. HM69 finite numerical accounts, verified on complete HM68 readouts.
This is a measurement-interface connection, not a new Shapley theorem, a new
personality predictor, or a completed psychological instrument.

## 1. Three Different Objects

Let x be a high/low focal behavior, y a companion behavior, and g its event
ownership. All eight cells have captured observations for both lexical banks.
For this experiment define C=(y,g), and disintegrate an interior joint profile:

\[
P_u(x,C)=\pi_u(C)\frac{1+x\mu_u(C)}2,
\quad \mu_u(C)=\mathbb E_u[x\mid C],
\quad x\in\{-1,+1\}.
\]

The full profile is retained. Composition pi is not removed as nuisance.
Conditional expression mu is not automatically a causal author-response
parameter. C here includes another behavior, not only an external situation.
Calling the pi account "choice" is a computational shorthand: it does not
identify preference without observing the opportunities available to an author.

For an explicitly declared observation law F:

\[
\Theta_u(\pi,\mu,F)
=\sum_C\pi_u(C)\sum_x\frac{1+x\mu_u(C)}2F(x,C).
\]

F is the common reader's actual vector-valued observation, not a person's
psychological response Bu. Material bank F0 versus F1 changes semantic content
and wording while retaining coded roles; it is not a pure device-calibration
change or the same person's observed longitudinal development.

M03 already supplied free-choice/fixed-condition designs. HM65 supplied the
Tq/Pu/Bu distinction; HM68 supplied actual content/ownership interactions.
HM69 connects these objects without improving the reader or fitting labels.

## 2. Exact Change Accounts, with Interactions Kept

For endpoints (pi0,mu0,F0) and (pi1,mu1,F1), G(S) uses endpoint1 for factors
in S and endpoint0 otherwise. All eight hybrid probability laws are valid.
Save the anchored subset differences:

\[
d_S=\sum_{A\subseteq S}(-1)^{|S|-|A|}G(A).
\]

Every replacement order telescopes to G(all)-G(empty), but the separate factor
accounts can depend on order. Average all six orders, or equivalently use:

\[
\phi_j=\sum_{S\ni j}\frac{d_S}{|S|},
\quad \sum_j\phi_j=G(\mathrm{all})-G(\varnothing).
\]

This is established Shapley accounting coordinatewise, not a unique causal
mediation decomposition. Retain each order and all interactions alongside the
symmetric convention. Norms are not additive contribution percentages.

For fixed F write A(C)=(F(+1,C)+F(-1,C))/2 and
B(C)=(F(+1,C)-F(-1,C))/2. Then

\[
\Theta=\sum_C\pi(C)[A(C)+\mu(C)B(C)].
\]

With bars denoting arithmetic endpoint means, the exact two-factor version is:

\[
\Delta\Theta
=\sum_C\Delta\pi(C)[A(C)+\bar\mu(C)B(C)]
+\sum_C\bar\pi(C)\Delta\mu(C)B(C).
\]

Thus composition and conditional-expression accounts can cancel. This is a
bilinear identity; it does not need an accuracy improvement to be useful.

## 3. A Six-Score Quotient Is Not the Whole Author Profile

Let Fscore be the eight-by-six score table. A vector v in
ker([1,Fscore]^T) defines two positive normalized profiles
Pplus/minus=1/8+/-.05*v/maxabs(v). Their six scores coincide by construction.
This uses ordinary nullspace algebra, not discovery of hidden psychological
traits or a fitted model of real authors.

All60 actual ordered-pair/family tables have constraint rank7. The constructed
profiles' largest six-score vector gap is 9.505e-17. Nevertheless every pair
has differing pi and mu (minimum Euclidean gaps .020842/.273684), and differing
body and centered-logit expectations (minimum gaps .141901/.044111).

Across30 pairs in each bank, native-score changes have RMS vector norm near0,
while symmetric composition and expression accounts have equal opposite
vectors: RMS norms .012953/.009990. Body expectation gaps have RMS vector norms
.425611/.415420. These are different units, not personality effect sizes.

The dimensional existence of a score fiber is guaranteed for eight probabilities
and six outputs plus normalization. The empirical part is that the retained
conditional profiles and other captured observations distinguish all60 chosen
fibers. No claim that every fiber, every embedding, or every population does so.
HM62's full-embedding-mean counterexample remains valid: full vector averaging
does not necessarily discard joint distribution information.

## 4. Actual Findings and Their Use

The full30-pair x8-case x6-stage protocol produced1440 account rows and240
profile pairs, without neural calls or fits. Vector-norm RMS for the score stage:

| Case | Total | Composition | Conditional Expression | Observation Law |
| --- | ---: | ---: | ---: | ---: |
| Composition, F0 | .109989 | .109989 | 0 | 0 |
| Composition, F1 | .087508 | .087508 | 0 | 0 |
| Expression, F0 | .027898 | 0 | .027898 | 0 |
| Expression, F1 | .025427 | 0 | .025427 | 0 |
| Law only | .116415 | 0 | 0 | .116415 |
| All three | .155590 | .084569 | .048809 | .116474 |

Fixed archetype magnitudes are chosen by the design, not estimated natural
prevalences. Do not interpret the table as which human mechanism is larger.
The mixed order-allocation range has RMS norm .134521. Different replacement
orders can give different account vectors, even with the same final total.

The measurement interface should return an observed score together with the
declared composition/conditional profile, observation-law version, and remaining
joint interactions. A score-only account can hide opposite changes. A changed
score can result from changed materials without any profile change.

The next mechanism must determine which conditioning/observation objects have
an interpretable, reproducible identity under material changes. Do not simply
invent extra Shapley factors, rotate coordinate choices to get a preferred
explanation, or tune teacher fidelity. Statistical disintegration alone does
not declare C to be the external environment.

## 5. Reproduction and Mathematical Ancestry

Canonical: results/grain_math_response_v1/author_observation_accounts_r1.
Plan/code/dependencies are frozen in manifest.json and code/ before the run.
All stage games, subset interactions, six sequences and profiles remain in01/.
The collector took2.670102 seconds. An alternate direct-sum/permutation verifier
checked18785176 scalar values with maximum difference1.421086e-14. This is
same-agent alternate arithmetic, not an independent human replication.
46 focused tests include18 new tests; Ruff lint and formatting passed.

```bash
python scripts/run_grain_math_author_observation_accounts.py verify
python -m pytest tests/test_grain_math_author_observation_accounts.py \
  tests/test_grain_math_ownership_composition.py tests/test_grain_math_joint_law.py -q
```

Do not rerun prepare/run into the existing frozen directory.
No new neural/API/private/label/training/generation inputs were used.
Original18/100/44 scope, psychological meaning, Bu, state dynamics and independent
new-text mathematical scoring remain separate open items.

[Shorrocks1999 draft section2](https://www.komkon.org/~tacik/science/shapley.pdf)
was read at PDF pages5--8 (zero-based4--7). Replacement-order averaging is prior
art. Kitagawa1955 metadata was found; its paywalled full proof was not read.
The pinned OpenAI Conditional-coordinate-sweeps-and-analytic-transfer paths
section was read/re-read completely. Its conditional permutation laws illustrate
the importance of keeping a conditioned object explicit; its independent-line,
random-bijection/Schatten and analytic-transfer assumptions were not established
here, so no principal theorem was transferred to LLMs or authors.
