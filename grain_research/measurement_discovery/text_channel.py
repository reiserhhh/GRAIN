"""A bounded, stateless text renderer and frozen lexical observation reader."""

from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

GRAMMARS = {
    "source": (
        ("I joined the conversation.", "I kept away from the conversation."),
        ("I took part in the discussion.", "I stayed outside the discussion."),
        ("I moved toward the group.", "I remained apart from the group."),
    ),
    "validation": (
        ("I entered the exchange.", "I avoided the exchange."),
        ("I contributed to the talk.", "I held back from the talk."),
        ("I approached the discussion.", "I stayed clear of the discussion."),
    ),
    "application": (
        ("I participated in the conversation.", "I did not take part in the conversation."),
        ("I engaged with the group.", "I remained separate from the group."),
        ("I went into the exchange.", "I kept out of the exchange."),
    ),
}


def render_observation(observation, split, family):
    """Render only the current binary observation with a fixed family grammar."""
    if split not in GRAMMARS or not 0 <= family < len(GRAMMARS[split]):
        raise ValueError("unknown renderer split or family")
    if int(observation) not in (0, 1):
        raise ValueError("observation must be binary")
    positive, negative = GRAMMARS[split][family]
    return positive if int(observation) else negative


class FrozenTextReader:
    """Train a source-only lexical reader and score new text without refitting."""

    def __init__(self, max_features=5000, alpha=1.0):
        self.max_features = max_features
        self.alpha = alpha

    def fit(self, texts, outcomes):
        """Fit word/character TF-IDF and Ridge on source records only."""
        texts, outcomes = list(texts), np.asarray(outcomes, dtype=float)
        if len(texts) != len(outcomes) or len(texts) < 4:
            raise ValueError("aligned source text and outcome arrays required")
        self.word_ = TfidfVectorizer(ngram_range=(1, 2), max_features=self.max_features)
        self.char_ = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=self.max_features)
        word = self.word_.fit_transform(texts)
        char = self.char_.fit_transform(texts)
        self.model_ = Ridge(alpha=self.alpha).fit(hstack([word, char]), outcomes)
        return self

    def predict(self, texts):
        """Return a bounded probability-like lexical readout."""
        texts = list(texts)
        x = hstack([self.word_.transform(texts), self.char_.transform(texts)])
        return np.clip(self.model_.predict(x), 1e-6, 1 - 1e-6)


def second_order_soft_probability(previous_a, previous_b, source_table):
    """Integrate a four-cell source table over independent soft prior outcomes."""
    a, b = float(previous_a), float(previous_b)
    weights = np.asarray([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b])
    return float(np.clip(weights @ np.asarray(source_table), 1e-6, 1 - 1e-6))


def second_order_hard_probability(previous_a, previous_b, source_table):
    """Use a fixed .5 threshold as a sensitivity comparison, not model selection."""
    state = 2 * int(float(previous_a) >= 0.5) + int(float(previous_b) >= 0.5)
    return float(np.clip(np.asarray(source_table)[state], 1e-6, 1 - 1e-6))
