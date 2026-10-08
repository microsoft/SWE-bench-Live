"""Windows validation helpers that preserve structured Go test-event integrity."""

from __future__ import annotations

import json
import re
from typing import Literal

TestStatus = Literal["pass", "fail", "skip"]

_GO_EVENT_ACTIONS = {"pass", "fail", "skip"}
_ARTIFACT_SUFFIX = re.compile(r'(?:\r?\n|\\n)"\}\s*$')


def is_corrupted_windows_test_name(name: str) -> bool:
    """Return whether a parsed name contains a captured Go JSON fragment.

    A terminal Go test event has a standalone test identifier. Names ending in
    a JSON record terminator (for example ``TestX (0.00s)\\n"}``) were created
    by a permissive parser consuming wrapped output, not emitted by ``go test``.
    """
    return bool(_ARTIFACT_SUFFIX.search(str(name)))


def parse_windows_go_json_status(log: str) -> dict[str, TestStatus] | None:
    """Parse complete terminal Go JSON events without regex fallback artifacts.

    Returns ``None`` when the log does not contain a complete Go JSON event, so
    callers can keep the instance-specific parser for non-Go logs. If Go events
    are present, only complete JSON records are authoritative; malformed or
    wrapped fragments are never turned into test identities.
    """
    status: dict[str, TestStatus] = {}
    saw_go_stream = False
    for raw_line in log.splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # A wrapped Go record is still evidence that this is a Go JSON
            # stream. Returning an empty status map below fails closed instead
            # of delegating its fragments to a permissive regex parser.
            if '"Action"' in line or '"Test"' in line:
                saw_go_stream = True
            continue
        action = str(event.get("Action", "")).lower()
        if action not in _GO_EVENT_ACTIONS:
            continue
        saw_go_stream = True
        if "Test" not in event:
            continue
        name = event.get("Test")
        if not isinstance(name, str) or not name or is_corrupted_windows_test_name(name):
            continue
        package = event.get("Package")
        # Published Go expectations are package-qualified. Preserve the package
        # emitted by go test -json so same-named tests do not collide and the
        # validator emits identities the evaluator can match exactly.
        identity = f"{package}::{name}" if isinstance(package, str) and package else name
        status[identity] = action  # type: ignore[assignment]
    return status if saw_go_stream else None


def normalize_windows_validation_status(
    parser: str,
    log: str,
    parsed_status: dict[str, str],
    platform: str,
) -> dict[str, str]:
    """Prefer structured Go JSON for Windows validation metadata.

    Validation is the source of ``PASS_TO_PASS`` and ``FAIL_TO_PASS`` fields.
    Go metadata is stored as ``package::Test[/subtest]`` identities, so the
    structured reader preserves both event fields rather than collapsing tests
    with equal names in different packages. When Windows Go JSON is available,
    a permissive regex parser must not add fragment-derived names to those
    dataset fields. Non-Windows and non-Go logs retain the existing parser
    result unchanged.
    """
    if platform != "windows":
        return parsed_status
    go_status = parse_windows_go_json_status(log)
    if go_status is None:
        return parsed_status
    return go_status