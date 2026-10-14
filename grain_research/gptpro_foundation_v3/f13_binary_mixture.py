#!/usr/bin/env python3
"""Source-ensemble finite-state history filtering and total variance."""

import numpy as np


class BinarySourceEnsemble:
    """Parallel source fits with the same frozen F08 observation semantics."""

    def __init__(self, fits, complete=False):
        """Use source-only transitions and initial distributions."""
        self.transition = np.array([f["b1"]["source_p"] for f in fits]).reshape(
            -1, 2, 2
        )
        self.pair = np.array([f["initial_pair"] for f in fits])
        self.complete, self.t, self.last = complete, 0, None

    def observe(self, pole, qualification):
        """Preserve missing time positions instead of concatenating qualified text."""
        if not 0 <= pole <= 1 or not 0 <= qualification <= 1:
            raise ValueError("probabilities required")
        if self.complete:
            if qualification != 1:
                raise ValueError("complete panel requires known qualification")
            if self.t:
                self.pair[:] = np.outer([1 - self.last, self.last], [1 - pole, pole])
            self.last = pole
        else:
            likelihood = (
                qualification * np.array([1 - pole, pole]) + (1 - qualification) / 2
            )
            if self.t < 2:
                self.pair *= (
                    likelihood[None, :, None]
                    if self.t == 0
                    else likelihood[None, None, :]
                )
            else:
                joint = (
                    self.pair[:, :, :, None]
                    * np.stack([1 - self.transition, self.transition], axis=-1)
                    * likelihood
                )
                self.pair = joint.sum(axis=1)
            norm = self.pair.sum(axis=(1, 2))
            if np.any(norm <= 0):
                raise ValueError("impossible observation")
            self.pair /= norm[:, None, None]
        self.t += 1

    def components(self, rng, draws=200):
        """Separate history-state, source and event variation; MC is a reference."""
        if self.t < 2:
            raise ValueError("insufficient history")
        w, p = self.pair.reshape(-1, 4), self.transition.reshape(-1, 4)
        means = (w * p).sum(axis=1)
        state = (w * (p - means[:, None]) ** 2).sum(axis=1).mean()
        source = means.var()
        event = (w * p * (1 - p)).sum(axis=1).mean()
        # Multinomial counts are exactly equivalent to drawing 200 categorical
        # states per source fit; no Gaussian posterior or fractional Beta counts.
        counts = rng.multinomial(draws, w)
        mc_means = (counts * p).sum(axis=1) / draws
        mc_second = (counts * p * p).sum(axis=1) / draws
        return {
            "mean": float(means.mean()),
            "state_variance": float(state),
            "source_variance": float(source),
            "event_variance": float(event),
            "probability_variance": float(state + source),
            "outcome_variance": float(state + source + event),
            "mc_mean": float(mc_means.mean()),
            "mc_probability_variance": float(mc_second.mean() - mc_means.mean() ** 2),
        }
