#!/usr/bin/env python3
"""Install the declared cached reader-axis layout without scientific changes."""

import numpy as np


def restore(response):
    """Transpose only the known advanced-indexed reader/author/view/target cache."""
    a = np.asarray(response, float)
    if a.shape != (2, 24, 2, 5) or not np.isfinite(a).all():
        raise ValueError("Only the sealed HM217 reader-first cache is supported")
    return a.transpose(1, 2, 0, 3).copy()
