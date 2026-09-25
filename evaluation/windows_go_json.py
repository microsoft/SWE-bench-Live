"""Helpers for validating Windows Go test-parser output against JSON events."""

from __future__ import annotations

import json
from typing import Literal

TestStatus = Literal["pass", "fail", "skip"]

_TERMINAL_ACTIONS = {"pass", "fail", "skip"}
_STATUS_SEVERITY = {"pass": 0, "skip": 1, "fail": 2}


def _iter_json_objects(log: str):
    """Yield JSON objects from wrapped or line-oriented command output."""
    buf: list[str] = []
    depth = 0
    in_string = False
    escape = False
    for char in log:
        if depth == 0:
            if char == "{":
                buf = [char]
                depth = 1
                in_string = False
                escape = False
            continue
        buf.append(char)
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                text = "".join(buf).replace("\r", "").replace("\n", "")
                try:
                    yield json.loads(text)
                except (TypeError, json.JSONDecodeError):
                    pass
                buf = []


def extract_go_json_test_status(log: str) -> dict[str, TestStatus]:
    """Return authoritative terminal status for each Go test name.

    The Windows console can wrap JSON objects across physical lines.  Parsing
    complete objects also prevents a package-level event from being associated
    with a later test by a DOTALL regular expression.  Test names are used as
    keys because the evaluator's parser/status format is test-name-only; the
    package is still required to identify a Go test event.
    """
    statuses: dict[str, TestStatus] = {}
    for event in _iter_json_objects(log):
        package = event.get("Package")
        test = event.get("Test")
        action = str(event.get("Action", "")).lower()
        if not package or not test or action not in _TERMINAL_ACTIONS:
            continue
        previous = statuses.get(test)
        if previous is None or _STATUS_SEVERITY[action] > _STATUS_SEVERITY[previous]:
            statuses[test] = action  # type: ignore[assignment]
    return statuses


def extract_go_json_test_keys(log: str) -> set[str]:
    """Return test names backed by explicit terminal Go JSON events."""
    return set(extract_go_json_test_status(log))


def filter_unsupported_go_json_status(
    status: dict[str, TestStatus],
    log: str,
    platform: str,
) -> dict[str, TestStatus]:
    """Filter ghost parser entries and use Go terminal events as authority.

    This is deliberately Windows-only and fail-closed.  If no recognizable Go
    JSON events exist, the original parser output is preserved.  When events
    do exist, a parsed status is kept only when its test name has a terminal
    event, and its status is replaced with that event's status.  Thus a parser
    cannot turn a real PASS event into a false FAIL, while real failures remain
    failures.
    """
    if platform != "windows":
        return status

    event_status = extract_go_json_test_status(log)
    if not event_status:
        return status

    # The generated parser is only a lossy projection of this structured report:
    # it can omit a real event as well as invent one.  The event map is therefore
    # authoritative once recognizable Go JSON is present.
    return event_status