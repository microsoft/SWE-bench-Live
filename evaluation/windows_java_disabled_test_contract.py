"""Preserve Windows Java rows whose re-enabled official tests are not scoreable.

A Windows row can use its official ``test_patch`` only to remove a JUnit
``@DisabledOnOs(... OS.WINDOWS ...)`` annotation.  That re-enables a named Java
test.  If none of those re-enabled test identities is represented by published
``FAIL_TO_PASS`` metadata, exact-name grading has no stated target for the
official regression and must not convert unrelated suite outcomes into a
candidate verdict.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_JAVA_TEST_FILE_HEADER = re.compile(r"^\+\+\+ b/(.+src/test/java/.+\.java)$")
_DISABLED_ON_WINDOWS = re.compile(
    r"^-(?!-).*@DisabledOnOs\s*\([^\n]*\bOS\.WINDOWS\b",
    re.IGNORECASE,
)
_JAVA_VOID_METHOD = re.compile(
    r"^(?![+-])\s*(?:public\s+|protected\s+|private\s+)?(?:static\s+)?void\s+([A-Za-z_$][\w$]*)\s*\(",
)


def reenabled_windows_junit_test_suffixes(test_patch: str) -> list[str]:
    """Return ``Class#method`` suffixes re-enabled by an official patch.

    The extractor deliberately recognizes only a removed ``@DisabledOnOs`` line
    mentioning ``OS.WINDOWS`` in a Java ``src/test/java`` file and the next
    nearby unchanged Java ``void`` method declaration.  It does not infer test
    classes from production files or parse unrelated annotation changes.
    """
    current_class: str | None = None
    awaiting_method_for: str | None = None
    result: list[str] = []
    seen: set[str] = set()

    for line in str(test_patch or "").splitlines():
        header = _JAVA_TEST_FILE_HEADER.match(line)
        if header:
            filename = header.group(1).rsplit("/", 1)[-1]
            current_class = filename[:-len(".java")]
            awaiting_method_for = None
            continue
        if current_class is None:
            continue
        if _DISABLED_ON_WINDOWS.match(line):
            awaiting_method_for = current_class
            continue
        if awaiting_method_for is None:
            continue
        method = _JAVA_VOID_METHOD.match(line)
        if method:
            suffix = f"{awaiting_method_for}#{method.group(1)}"
            if suffix not in seen:
                seen.add(suffix)
                result.append(suffix)
            awaiting_method_for = None
    return result


def _matches_reenabled_suffix(expected_name: object, suffixes: Iterable[str]) -> bool:
    name = str(expected_name)
    return any(name == suffix or name.endswith("." + suffix) for suffix in suffixes)


def windows_java_disabled_test_contract_is_gradable(
    instance: dict,
    platform: str = "windows",
) -> tuple[bool, list[str]]:
    """Check that published metadata represents a Windows-re-enabled JUnit test.

    The guard is intentionally fail-closed and narrow:

    * non-Windows rows are unchanged;
    * Java patches without this exact removed Windows-disable annotation are
      unchanged;
    * a row is unavailable only when none of the re-enabled ``Class#method``
      suffixes appears in ``FAIL_TO_PASS``.

    It does not rewrite test metadata, choose substitute identities, or decide
    whether a candidate patch solved the underlying task.
    """
    if str(platform).lower() != "windows":
        return True, []
    suffixes = reenabled_windows_junit_test_suffixes(
        str(instance.get("test_patch", "") or "")
    )
    if not suffixes:
        return True, []
    expected = list(instance.get("FAIL_TO_PASS", []) or [])
    if any(_matches_reenabled_suffix(name, suffixes) for name in expected):
        return True, []
    return False, suffixes