"""No-result-changing retry, terminal failure and path isolation tests."""

import pytest

from grain_research.vector_mechanisms.status_io_guard import guarded_writer


def windows_error(code):
    """Produce portable test-only Windows-like access errors."""
    exc = PermissionError("synthetic sharing failure")
    exc.winerror = code
    return exc


@pytest.mark.parametrize("code", [5, 32, 33])
def test_status_retry_preserves_exact_payload_and_path(tmp_path, code):
    calls, events, waits = [], [], []
    value = {"status": "RUNNING", "authors": 42}
    path = tmp_path / "queue_status.json"

    def writer(p, v):
        calls.append((p, v))
        if len(calls) <= 2:
            raise windows_error(code)
        return "written"

    guarded = guarded_writer(writer, path, events.append, waits.append)
    assert guarded(path, value) == "written"
    assert len(calls) == 3 and all(p == path and v is value for p, v in calls)
    assert waits == [0.05, 0.1] and len(events) == 2


@pytest.mark.parametrize("wrong_path,code", [(True, 5), (False, 13), (False, None)])
def test_other_writes_and_errors_fail_immediately(tmp_path, wrong_path, code):
    calls, events, waits = [], [], []
    exc = windows_error(code)

    def writer(*args):
        calls.append(args)
        raise exc

    target = tmp_path / "queue_status.json"
    wrapped = guarded_writer(writer, target, events.append, waits.append)
    with pytest.raises(PermissionError) as caught:
        wrapped(tmp_path / "manifest.json" if wrong_path else target, {})
    assert caught.value is exc and len(calls) == 1 and not events and not waits


def test_permanent_error_has_finite_retry_budget(tmp_path):
    calls, events, waits = [], [], []
    exc = windows_error(5)

    def writer(*args):
        calls.append(args)
        raise exc

    p = tmp_path / "queue_status.json"
    with pytest.raises(PermissionError) as caught:
        guarded_writer(writer, p, events.append, waits.append)(p, {})
    assert caught.value is exc and len(calls) == 6 and len(events) == 5
    assert sum(waits) == pytest.approx(1.55)
