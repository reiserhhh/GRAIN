#!/usr/bin/env python3
"""Bound one research stage to an explicit data-file allowlist in its own process."""

import io
import os
import sys
from pathlib import Path

import numpy as np


def install(reads, writes, probes, code_roots):
    """Deny undeclared data opens; return recorded denials after testing target paths.

    This is an audit hook for trusted research code, not a hostile-code sandbox.
    Install only in a dedicated process because Python audit hooks are permanent.
    """
    reads = {str(Path(p).resolve()) for p in reads}
    writes = {str(Path(p).resolve()) for p in writes}
    reads |= writes
    roots = [Path(p).resolve() for p in code_roots] + [
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
    ]
    denied = []
    b = io.BytesIO()
    np.savez_compressed(b, x=np.zeros(1))
    b.seek(0)
    with np.load(b, allow_pickle=False) as z:
        _ = z["x"]

    def guard(event, args):
        if event != "open":
            return
        name, _, flags = args
        if isinstance(name, int):
            raise PermissionError("Undeclared descriptor")
        p = Path(os.path.realpath(os.fsdecode(name)))
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        code = p.suffix in (".py", ".pyc", ".so", ".dylib") and any(
            p.is_relative_to(r) for r in roots
        )
        if (writing and str(p) not in writes) or (
            not writing and str(p) not in reads and not code
        ):
            denied.append(str(p))
            raise PermissionError("Undeclared target/data path denied")

    sys.dont_write_bytecode = True
    sys.addaudithook(guard)
    for p in probes:
        try:
            with open(p, "rb"):
                pass
        except PermissionError:
            continue
        raise AssertionError("Forbidden target probe was readable")
    return denied
