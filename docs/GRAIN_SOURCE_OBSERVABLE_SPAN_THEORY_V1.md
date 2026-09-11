# From sufficient vector span to evidence-coordinate reading

2026-09-11. NEXT84 completed; NEXT85 is a derived candidate, not an achieved result.

## 1. An actual source restriction has now been removed

Let A be the fixed whitened source observation kernel and Y the reader residual
targets. NEXT83 minimized a vector/functional objective over row(Y). Its precise
sufficient-space proof gives S*=span(Y,R), for fixed source rulers R, scalar
input kernel and isotropic coefficient penalty. NEXT84 implements this full
numerical space and verifies the full-space synthetic normal equations.

On the real source panel, S* has47 dimensions rather than27. The old optimum's
missing-coordinate gradient is nonzero and the same objective decreases.
This establishes an active mathematical restriction and its correction. It
does not identify a new trait or a universal optimum over all possible readers.

Background: vector-valued RKHS regularization and representations are standard;
see [Alvarez, Rosasco & Lawrence (2012), sections3 and5](https://arxiv.org/html/1106.6251v2).
Our finite input contract, functional probes and basis completion are specific
application choices. We do not claim to invent the representer theorem.

## 2. Three different objects were previously easy to conflate

1. Which output vectors the function may express: S*.
2. Which coefficients the source observations teach: Theta*.
3. How the resulting error acts on the current material: r(x)^T e(x).

NEXT84 enlarges(1), materially changes the source optimum(2), but barely improves
the recipient scalar and leaves response amplitude open(3). Newly available
directions carry only3.32e-7 actual recipient field energy. This narrows the next
question from 'more dimensions?' to 'which material-conditioned relations teach
the function to use relevant directions?'. The current source panel is small;
it does not distinguish every possible input omission from limited supervision.

For any accounting subspace U, e=e_in+e_out and

```text
(r^T e)^2 = (r^T e_in)^2 + (r^T e_out)^2
            + 2(r^T e_in)(r^T e_out).
```

The same unchanged control has95.62% less outside projected energy under the
new basis, yet identical total error. This is a concrete witness that representer
coverage is not recovery. Changing a decomposition must not be sold as a gain.

## 3. A smaller material-relative object can be sufficient for a stated reading

For two queries, write their readers as m +/- d/2. For common evidence pbar and
four common GRAIN direction vectors bbar_f, define a five-row frame

```text
R(x) = [pbar(x)^T; bbar_1(x)^T; ...; bbar_4(x)^T].
c(x) = R(x) [d(x)-d0].
```

These five coordinates measure how the differential reader acts on this
material. They are not five new personality factors. They are sufficient for
the differential-reader term in the *specified* local contrast and its four
anchored actions. The common-reader/evidence-change term remains separately.
Other compositions/queries can require other observables; no all-task claim.

This suggests teaching c(x) directly and using an application-available frame
Rhat(x) built from the existing source-compiled selector/value program. Use the
same Rhat construction in teaching and application. Do not substitute current
native evidence or current native d at application. H16 remains an explicit input.

Given baseline db(x), a full-row-rank Rhat and predicted chat(x), the smallest
Euclidean adjustment satisfying the five coordinates is

```text
dtilde = db + Rhat^+ [chat - Rhat(db-d0)]
       = d0 + (I-P_Rhat)(db-d0) + Rhat^+ chat,
P_Rhat = Rhat^+ Rhat.
```

Then Rhat(dtilde-d0)=chat. The old field is retained in the unobserved complement;
it is not declared psychological noise or discarded. This is ordinary linear
constraint geometry, proposed here as a GRAIN/query interface. Because Rhat
changes with material, its possible output directions are not confined to one
global source span. A fixed-frame comparator distinguishes this coordinate
choice from generic five-output regression.

## 4. Moving coordinates require a product rule

Teaching only Rhat times the old derivative target would teach the wrong object:

```text
Dc[D] = DRhat[D](d-d0) + Rhat Dd[D].
```

The first term changes the evidence frame; the second changes the reader.
For full row rank, Rhat^+=Rhat^T (Rhat Rhat^T)^-1. Its derivative follows the
ordinary inverse/product rules. All of these terms must appear when the lifted
reader is differentiated. The new diagonal scalar derivative must again remain
distinct from its anchored local action; no double-counting of reader motion.

This links the user's two views operationally: the same GRAIN-indexed text
directions define a local evidence frame, and the query path defines how to
read that frame. Their relation is conditional and signed, not necessarily an
addition of two independent positive personality scores.

## 5. Interpretation stays downstream of the numerical link

The numerical object identifies which text direction changes a declared query
reading under a specified model/program. It can guide an independent dialogue
coding or candidate relational interpretation. It does not, by itself, name an
ego state, defense or diagnosis. No new clinical data is required for NEXT85.
Retain evidence addresses, numerical effect, independent coding and proposed
human explanation as distinct linked records.

[NEXT84 report](../reports/grain_vector_mechanisms/SOURCE_OBSERVABLE_SPAN_REPORT_V1.md)
and [NEXT85 execution plan](GRAIN_EVIDENCE_COORDINATE_READER_PLAN_V1.md).
