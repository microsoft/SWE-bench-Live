"""Recognize a narrow Windows Go CGO prerequisite failure before scoring.

Some Windows task images can start ``go test`` but still lack the native CGO
prerequisites required by a repository.  In the Teleport case this leaves the
Windows-only ``tncon`` implementation and SQLite-backed package without their
CGO implementations, so the published command stops during compilation.

This module deliberately recognizes only the joint compiler signature observed
for that failure.  It does not install a compiler, change the published command,
or reinterpret ordinary test failures.
"""

from __future__ import annotations

import re

_BUILD_CONSTRAINTS_EXCLUDED = re.compile(
    r"(?im)\bbuild constraints exclude all Go files in\s+.+$"
)
_SQLITE_UNDEFINED = re.compile(r"(?im)\bundefined:\s+sqlite3\.")
_TNCON_UNDEFINED = re.compile(r"(?im)\bundefined:\s+tncon\.")


def windows_go_cgo_prerequisite_failure(
    output: str,
    platform: str = "windows",
) -> dict[str, object] | None:
    """Return evidence for the known Windows CGO prerequisite signature.

    The guard is intentionally a conjunction: Go must report excluded source
    files *and* missing symbols from both the SQLite and Windows terminal CGO
    dependency paths.  A single ordinary compilation error, a Linux result, or
    a test assertion failure remains outside this contract.
    """
    if str(platform).lower() != "windows":
        return None

    text = str(output or "")
    excluded = _BUILD_CONSTRAINTS_EXCLUDED.findall(text)
    if not excluded or not _SQLITE_UNDEFINED.search(text) or not _TNCON_UNDEFINED.search(text):
        return None

    return {
        "build_constraints_excluded": excluded,
        "missing_cgo_symbol_families": ["sqlite3", "tncon"],
    }