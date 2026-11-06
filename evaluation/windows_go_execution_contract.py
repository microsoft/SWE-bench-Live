"""Fail closed when a Windows Go test patch targets tests the command did not run.

A published Windows row may supply a Go ``test_patch`` that changes a regression
function while its published ``test_cmd`` excludes that function through build
tags or package selection.  A result from unrelated tests cannot establish a
candidate verdict for that row.
"""

from __future__ import annotations

import json
import re

_GO_TEST_FILE_HEADER = re.compile(r"^\+\+\+ b/(.+)$")
_GO_TEST_FUNCTION = re.compile(r"^[ +\-]func\s+(Test[A-Za-z0-9_]+)\s*\(")
_HUNK_FUNCTION = re.compile(r"^@@ .* @@(?:\s+.*?\bfunc\s+(Test[A-Za-z0-9_]+)\s*\()?")
_TERMINAL_GO_ACTIONS = {"pass", "fail", "skip"}


def touched_go_test_functions(test_patch: str) -> list[str]:
    """Return Go test roots changed by a unified diff in ``*_test.go`` files.

    A test is considered touched when a changed line appears in a hunk whose
    function context identifies ``func Test...``. New test declarations are
    also recognized directly. Removed-only lines count because they can alter a
    regression assertion just as added lines can.
    """
    current_path: str | None = None
    hunk_test: str | None = None
    result: list[str] = []
    seen: set[str] = set()

    def record(name: str | None) -> None:
        if name and name not in seen:
            seen.add(name)
            result.append(name)

    for line in str(test_patch or "").splitlines():
        if line.startswith("diff --git "):
            current_path = None
            hunk_test = None
            continue
        header = _GO_TEST_FILE_HEADER.match(line)
        if header:
            current_path = header.group(1)
            hunk_test = None
            continue
        if not current_path or not current_path.endswith("_test.go"):
            continue
        if line.startswith("@@"):
            header_match = _HUNK_FUNCTION.match(line)
            hunk_test = header_match.group(1) if header_match else None
            continue
        if line.startswith(("+++", "---")):
            continue
        declaration = _GO_TEST_FUNCTION.match(line)
        if declaration:
            hunk_test = declaration.group(1)
            if line.startswith(("+", "-")):
                record(hunk_test)
            continue
        if line.startswith(("+", "-")):
            record(hunk_test)
    return result


def observed_go_test_roots(log: str) -> set[str] | None:
    """Read complete terminal Go JSON test events, or ``None`` if absent."""
    roots: set[str] = set()
    saw_terminal_test = False
    for line in str(log or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("Action") not in _TERMINAL_GO_ACTIONS:
            continue
        test = event.get("Test")
        if not isinstance(test, str) or not test:
            continue
        saw_terminal_test = True
        roots.add(test.split("/", 1)[0])
    return roots if saw_terminal_test else None


def unexecuted_windows_go_test_patch_targets(
    test_patch: str,
    log: str,
    platform: str = "windows",
) -> tuple[list[str], list[str], set[str] | None]:
    """Return targets omitted by a Windows Go stream when every target is absent.

    The guard is intentionally narrow. It is a no-op outside Windows, for
    patches that do not touch a Go test, and for commands without terminal Go
    JSON events. It blocks scoring only when a structured Go stream exists and
    *none* of the touched official regression roots executed.
    """
    targets = touched_go_test_functions(test_patch)
    if str(platform).lower() != "windows" or not targets:
        return [], targets, None
    observed = observed_go_test_roots(log)
    if observed is None or any(target in observed for target in targets):
        return [], targets, observed
    return targets, targets, observed