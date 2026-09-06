#!/usr/bin/env python3
"""Bounded operational retry for one Windows status-file replacement only."""

from pathlib import Path
from time import sleep


def guarded_writer(writer, status_path, record, sleeper=sleep):
    """Retry transient Windows access/sharing errors without changing values or targets."""
    target = Path(status_path).resolve()
    delays = (0.05, 0.1, 0.2, 0.4, 0.8)

    def write(path, value):
        """Pass all non-status writes through; preserve the final original exception."""
        if Path(path).resolve() != target:
            return writer(path, value)
        for attempt in range(len(delays) + 1):
            try:
                return writer(path, value)
            except PermissionError as exc:
                code = getattr(exc, "winerror", None)
                if code not in (5, 32, 33) or attempt == len(delays):
                    raise
                record(
                    {
                        "event": "retry_status_write",
                        "failed_attempt": attempt + 1,
                        "winerror": code,
                        "delay_seconds": delays[attempt],
                    }
                )
                sleeper(delays[attempt])

    return write
