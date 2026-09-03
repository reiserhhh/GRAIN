"""Canonical Python namespace for the GRAIN analysis method.

The implementation was originally published under ``suica_core``. Extending
this package's search path preserves imports of all existing modules without
copying code or changing frozen experiment sources.
"""

from __future__ import annotations

import suica_core as _legacy_core

__path__ = _legacy_core.__path__
__version__ = "1.1.0"

