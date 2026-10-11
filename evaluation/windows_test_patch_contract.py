"""Preserve the link between Windows Go test patches and expected regressions.

Windows validation publishes exact ``FAIL_TO_PASS`` identities. When a Go test
patch changes a regression function but none of those roots appears in the
published ``FAIL_TO_PASS`` set, the row has no stated target for exact-name
evaluation. That is a dataset-validation problem, not evidence about a
candidate patch.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_GO_TEST_FILE_HEADER = re.compile(r"^\+\+\+ b/(.+)$")
_GO_TEST_FUNCTION = re.compile(r"^\+\s*func\s+(Test[A-Za-z0-9_]+)\s*\(")
_GO_PATCH_TEST_DECLARATION = re.compile(r"^[ +\-]\s*func\s+(Test[A-Za-z0-9_]+)\s*\(")
_GO_HUNK_TEST_FUNCTION = re.compile(r"^@@ .* @@(?:\s+.*?\bfunc\s+(Test[A-Za-z0-9_]+)\s*\()?")


def added_go_test_functions(test_patch: str) -> list[str]:
    """Return distinct top-level Go test functions added in ``*_test.go`` files.

    Only added diff lines in a test file count.  This avoids treating removed
    tests, context lines, helpers, or non-Go files as regression targets.
    """
    current_path: str | None = None
    result: list[str] = []
    seen: set[str] = set()
    for line in str(test_patch or "").splitlines():
        header = _GO_TEST_FILE_HEADER.match(line)
        if header:
            current_path = header.group(1)
            continue
        if not current_path or not current_path.endswith("_test.go"):
            continue
        if line.startswith("+++"):
            continue
        match = _GO_TEST_FUNCTION.match(line)
        if match:
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                result.append(name)
    return result


def touched_go_test_functions(test_patch: str) -> list[str]:
    """Return Go test roots touched by a unified diff in ``*_test.go`` files.

    Added or removed declarations are explicit targets. For edits to an existing
    test body, unified-diff hunk context identifies the containing test root.
    This remains structural: helpers and non-test files are ignored.
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
            match = _GO_HUNK_TEST_FUNCTION.match(line)
            hunk_test = match.group(1) if match else None
            continue
        if line.startswith(("+++", "---")):
            continue
        declaration = _GO_PATCH_TEST_DECLARATION.match(line)
        if declaration:
            hunk_test = declaration.group(1)
            if line.startswith(("+", "-")):
                record(hunk_test)
            continue
        if line.startswith(("+", "-")):
            record(hunk_test)
    return result


def _matches_test_root(expected_name: object, roots: Iterable[str]) -> bool:
    """Compare a published Go identity to a test root in either known delimiter form."""
    name = str(expected_name)
    component = name.rsplit("::", 1)[-1]
    for root in roots:
        if component == root or component.startswith(root + "/"):
            return True
        # Packer-style published identities join package and test with '/',
        # while the raw Go parser uses '::'. Check the final path component
        # without altering metadata or selecting a replacement name.
        if name.rsplit("/", 1)[-1] == root:
            return True
    return False


def windows_go_test_patch_contract_is_gradable(
    instance: dict,
    platform: str = "windows",
) -> tuple[bool, list[str]]:
    """Check that a Windows Go test patch has a represented regression target.

    The guard is deliberately narrow:

    * non-Windows rows are unchanged;
    * rows without a touched top-level Go test are unchanged;
    * a row is blocked only when *none* of its touched test roots is represented
      by any published expected identity.

    It does not infer a replacement name, alter metadata, or decide whether a
    candidate solved the task.
    """
    if str(platform).lower() != "windows":
        return True, []
    roots = touched_go_test_functions(str(instance.get("test_patch", "") or ""))
    if not roots:
        return True, []
    expected = list(instance.get("PASS_TO_PASS", []) or []) + list(instance.get("FAIL_TO_PASS", []) or [])
    if any(_matches_test_root(name, roots) for name in expected):
        return True, []
    return False, roots