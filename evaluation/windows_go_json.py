"""Helpers for validating Windows Go test-parser output against JSON events."""

from __future__ import annotations

import json
from typing import Literal

TestStatus = Literal["pass", "fail", "skip"]


def extract_go_json_test_keys(log: str) -> set[str]:
    """Return test keys that are backed by real ``go test -json`` events.

    The Windows runner may feed a custom, regex-based parser a stream containing
    output from many packages.  A DOTALL parser can accidentally associate a
    later ``Test`` field with an earlier package-level failure and invent a
    status for a test that never occurred.  JSON events are the authoritative
    source for the package/test identity, so use only events with an explicit
    test and terminal action.
    """
    keys: set[str] = set()
    terminal_actions = {"pass", "fail", "skip"}
    for line in log.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        package = event.get("Package")
        test = event.get("Test")
        action = event.get("Action")
        if package and test and action in terminal_actions:
            keys.add(f"{package}::{test}")
    return keys


def filter_unsupported_go_json_status(
    status: dict[str, TestStatus],
    log: str,
    platform: str,
) -> dict[str, TestStatus]:
    """Drop parser statuses that have no corresponding Go JSON test event.

    This is deliberately Windows-only and fail-closed.  If the log does not
    contain recognizable Go JSON events, the evaluator preserves the parser
    output rather than guessing that a different test framework is Go.
    """
    if platform != "windows":
        return status

    event_keys = extract_go_json_test_keys(log)
    if not event_keys:
        return status

    return {
        name: test_status
        for name, test_status in status.items()
        if name in event_keys
    }