"""Preserve the link between Windows .NET test patches and expected regressions.

Windows rows are graded against exact ``FAIL_TO_PASS`` identities. When an
official C# test patch adds xUnit-style regression methods but none is named by
that metadata, the published row has no stated target for exact-name grading.
That is a dataset-contract problem, not evidence about a candidate patch.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_CSHARP_FILE_HEADER = re.compile(r"^\+\+\+ b/(.+\.cs)$", re.IGNORECASE)
_CSHARP_TEST_ATTRIBUTE = re.compile(
    r"^\+\s*\[(?:[A-Za-z_][\w.]*\.)?"
    r"(?:Fact|Theory|Win32Theory|StaFact|StaTheory)(?:\s*\(|\s*\])",
    re.IGNORECASE,
)
_CSHARP_TEST_METHOD = re.compile(
    r"^\+\s*(?:public|internal|protected|private)\s+"
    r"(?:async\s+)?(?:void|Task(?:\s*<[^>]+>)?|ValueTask(?:\s*<[^>]+>)?)\s+"
    r"([A-Za-z_]\w*)\s*\(",
    re.IGNORECASE,
)


def _is_test_source_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return normalized.startswith(("test/", "tests/")) or "/test/" in normalized or "/tests/" in normalized


def added_dotnet_test_methods(test_patch: str) -> list[str]:
    """Return added attributed C# test methods from conventional test paths.

    Only methods added in ``.cs`` files under a test directory and preceded by
    an added xUnit-style test attribute count. This deliberately excludes test
    helpers, production methods, deleted code, and context lines.
    """
    current_path: str | None = None
    pending_test_attribute_line: int | None = None
    result: list[str] = []
    seen: set[str] = set()

    for line_number, line in enumerate(str(test_patch or "").splitlines(), start=1):
        header = _CSHARP_FILE_HEADER.match(line)
        if header:
            current_path = header.group(1)
            pending_test_attribute_line = None
            continue

        if line.startswith("@@"):
            pending_test_attribute_line = None
            continue

        if not current_path or not _is_test_source_path(current_path):
            continue

        if line.startswith("+++"):
            continue

        if not line.startswith("+"):
            pending_test_attribute_line = None
            continue

        if _CSHARP_TEST_ATTRIBUTE.match(line):
            pending_test_attribute_line = line_number
            continue

        method = _CSHARP_TEST_METHOD.match(line)
        if method and pending_test_attribute_line is not None and line_number - pending_test_attribute_line <= 30:
            name = method.group(1)
            if name not in seen:
                seen.add(name)
                result.append(name)
            pending_test_attribute_line = None

    return result


def _matches_added_test_method(expected_name: object, methods: Iterable[str]) -> bool:
    """Match a published fully-qualified .NET identity to an added method."""
    identity = str(expected_name or "").strip()
    if not identity:
        return False
    bare_identity = identity.split("(", 1)[0]
    return any(
        bare_identity == method or bare_identity.endswith("." + method)
        for method in methods
    )


def windows_dotnet_test_patch_contract_is_gradable(
    instance: dict,
    platform: str = "windows",
) -> tuple[bool, list[str]]:
    """Check whether a Windows C# test patch has a represented regression.

    The guard is deliberately narrow:

    * non-Windows rows are unchanged;
    * rows without added attributed C# test methods are unchanged; and
    * a row is unavailable only when no published ``FAIL_TO_PASS`` identity
      names an added method.

    It does not repair metadata, infer replacement test names, change commands,
    or decide whether a candidate solves the task.
    """
    if str(platform).lower() != "windows":
        return True, []
    methods = added_dotnet_test_methods(str(instance.get("test_patch", "") or ""))
    if not methods:
        return True, []
    expected = list(instance.get("FAIL_TO_PASS", []) or [])
    if any(_matches_added_test_method(name, methods) for name in expected):
        return True, []
    return False, methods