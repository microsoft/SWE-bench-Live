"""Recover authoritative Go test statuses from Windows console JSON output."""

from __future__ import annotations

import json
import re
from typing import Iterator, Literal

TestStatus = Literal["pass", "fail", "skip"]
_TERMINAL_ACTIONS = {"pass", "fail", "skip"}
_STATUS_SEVERITY = {"pass": 0, "skip": 1, "fail": 2}


def _iter_balanced_json_objects(log: str) -> Iterator[dict]:
    """Yield complete JSON objects when their brace/string structure survives."""
    buffer: list[str] = []
    depth = 0
    in_string = False
    escape = False
    for char in log:
        if depth == 0:
            if char == "{":
                buffer = [char]
                depth = 1
                in_string = False
                escape = False
            continue
        buffer.append(char)
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
                try:
                    yield json.loads("".join(buffer).replace("\r", "").replace("\n", ""))
                except json.JSONDecodeError:
                    pass
                buffer = []


def _iter_time_delimited_go_json_objects(log: str) -> Iterator[dict]:
    """Recover Go JSON events whose physical lines were split by Windows.

    A Go `-json` event begins with a Time field.  Windows console capture can
    insert physical line breaks at arbitrary columns, including inside a JSON
    string, which makes a global brace scanner lose synchronization.  Splitting
    only at the next physical Go event boundary and then removing those capture
    line breaks restores each individual object without parsing arbitrary log
    text.
    """
    starts = list(re.finditer(r'(?m)^\{"Time"\s*:', log))
    decoder = json.JSONDecoder()
    for index, start_match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(log)
        candidate = log[start_match.start() : end].replace("\r", "").replace("\n", "")
        try:
            event, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def _iter_go_json_events(log: str) -> Iterator[dict]:
    """Yield normal and Windows line-wrapped Go JSON objects."""
    yield from _iter_balanced_json_objects(log)
    yield from _iter_time_delimited_go_json_objects(log)


def extract_go_json_test_status(log: str) -> dict[str, TestStatus]:
    """Return terminal status keyed by test name from explicit Go JSON events."""
    statuses: dict[str, TestStatus] = {}
    for event in _iter_go_json_events(log):
        package = event.get("Package")
        test = event.get("Test")
        action = str(event.get("Action", "")).lower()
        if not package or not test or action not in _TERMINAL_ACTIONS:
            continue
        prior = statuses.get(test)
        if prior is None or _STATUS_SEVERITY[action] > _STATUS_SEVERITY[prior]:
            statuses[test] = action  # type: ignore[assignment]
    return statuses


def filter_windows_go_json_status(
    status: dict[str, TestStatus], log: str, platform: str
) -> dict[str, TestStatus]:
    """Use structured Go terminal events as Windows parser authority.

    Non-Windows and non-Go logs deliberately retain the existing parser result.
    When recognizable Go JSON test events exist, the structured terminal event
    stream is authoritative: it prevents regex parser ghosts and preserves
    terminal PASS/FAIL status despite console wrapping.
    """
    if platform != "windows":
        return status
    event_status = extract_go_json_test_status(log)
    return event_status or status