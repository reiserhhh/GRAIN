"""Qualified-history adapters preserving the source B1/B2/M definitions."""

import numpy as np

from .f08_filter import filter_step

FEATURES = np.array([[1, b, a, a * b] for a in (0, 1) for b in (0, 1)], dtype=float)


class QualifiedHistory:
    """Sequential observed evidence with frozen source-transition interpolation."""

    def __init__(self, fitted, initial_pair=None, source_mean=0.5, mode="filtered"):
        """Initialize only from source quantities, never application hidden states."""
        self.fitted = fitted
        if mode not in ("filtered", "independent"):
            raise ValueError("declared observation approximation required")
        self.mode = mode
        self.previous_poles = []
        self.pair = (
            np.full((2, 2), 0.25)
            if initial_pair is None
            else np.array(initial_pair, dtype=float)
        )
        if (
            self.pair.shape != (2, 2)
            or (self.pair < 0).any()
            or not np.isclose(self.pair.sum(), 1)
        ):
            raise ValueError("normalized initial source pair required")
        self.transition = np.asarray(fitted["b1"]["source_p"]).reshape(2, 2)
        self.source_mean = float(source_mean)
        self.counts = np.zeros((4, 2))
        self.gram = np.zeros((4, 4))
        self.rhs = np.zeros(4)
        self.mass = self.total = 0.0
        self.t = 0

    def observe(self, pole, qualification):
        """Append one time position; only qualified mass updates personal statistics."""
        if not 0 <= pole <= 1 or not 0 <= qualification <= 1:
            raise ValueError("probability readings required")
        if self.mode == "independent":
            if qualification != 1:
                raise ValueError(
                    "independent complete panel requires known eligibility"
                )
            if self.t >= 2:
                a, b = self.previous_poles[-2:]
                states = np.outer([1 - a, a], [1 - b, b]).ravel()
                weighted = states[:, None] * np.array([1 - pole, pole])[None, :]
                self.counts += weighted
                self.gram += FEATURES.T @ (states[:, None] * FEATURES)
                self.rhs += FEATURES.T @ weighted[:, 1]
            self.previous_poles.append(pole)
            if self.t >= 1:
                a, b = self.previous_poles[-2:]
                self.pair = np.outer([1 - a, a], [1 - b, b])
        elif self.t < 2:
            likelihood = (
                qualification * np.array([1 - pole, pole]) + (1 - qualification) / 2
            )
            self.pair *= likelihood[:, None] if self.t == 0 else likelihood[None, :]
            norm = self.pair.sum()
            if norm <= 0:
                raise ValueError("initial observation impossible")
            self.pair /= norm
        else:
            step = filter_step(self.pair, self.transition, pole, qualification)
            self.pair = step["pair"]
            weighted = qualification * step["joint"].reshape(4, 2)
            self.counts += weighted
            self.gram += FEATURES.T @ (weighted.sum(axis=1)[:, None] * FEATURES)
            self.rhs += FEATURES.T @ weighted[:, 1]
        self.mass += qualification
        self.total += qualification * pole
        self.t += 1

    def predict(self):
        """Score the next outcome before consuming it; retain model-only status."""
        if self.t < 2:
            raise ValueError("two time positions required")
        source = np.asarray(self.fitted["b2"]["coef"])
        coef = np.linalg.solve(self.gram + np.eye(4), self.rhs + source)
        raw = (self.counts[:, 1] + 1) / (self.counts.sum(axis=1) + 2)
        model = self.fitted["m"]
        right = np.asarray(model["right"])
        projected = np.asarray(model["mean"]) + (raw - model["mean"]) @ right.T @ right
        probabilities = {
            "b1": float(self.pair.ravel() @ self.transition.ravel()),
            "b2": float(self.pair.ravel() @ (FEATURES @ coef)),
            "m": float(self.pair.ravel() @ projected),
            "source_mean": self.source_mean,
            "personal_mean": self.total / self.mass if self.mass else self.source_mean,
        }
        return {
            "probabilities": {
                k: float(np.clip(v, 1e-6, 1 - 1e-6)) for k, v in probabilities.items()
            },
            "raw_probabilities": probabilities,
            "evidence_mass": self.mass,
            "transition_evidence_mass": float(self.counts.sum()),
            "status": "prior_only" if self.mass == 0 else "source_transition_filtered",
        }
