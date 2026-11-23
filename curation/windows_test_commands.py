"""Windows-specific normalization for curated test commands."""

from __future__ import annotations

import re
from collections.abc import Sequence

_GO_FILE_GLOB = re.compile(
    r"(?<!\S)(?P<directory>(?:\.?[\\/])?(?:[A-Za-z0-9_.@+~-]+[\\/])+)\*\.go(?=\s|$)",
    flags=re.IGNORECASE,
)
_COMMAND_SEPARATOR = re.compile(r"[;|&]")
_GO_TEST = re.compile(r"\bgo(?:\.exe)?\s+test\b", flags=re.IGNORECASE)


def _is_go_test_argument(command: str, offset: int) -> bool:
    """Return whether a token belongs to the current shell segment's go test."""
    separators = list(_COMMAND_SEPARATOR.finditer(command, 0, offset))
    segment_start = separators[-1].end() if separators else 0
    return _GO_TEST.search(command[segment_start:offset]) is not None


def normalize_windows_go_test_file_globs(command: str) -> str:
    """Convert local ``*.go`` file globs to Windows-safe Go package arguments.

    PowerShell does not expand native-command wildcards.  A generated command
    such as ``go test pkg/client/*.go`` therefore passes the literal asterisk
    to Go and fails before its tests run.  For relative local directories,
    ``go test ./pkg/client`` is the package-mode equivalent and works on both
    Windows and Unix shells.  The helper deliberately leaves absolute paths,
    quoted arguments, and non-``go test`` command segments untouched.
    """

    def replace(match: re.Match[str]) -> str:
        if not _is_go_test_argument(command, match.start()):
            return match.group(0)
        directory = match.group("directory").rstrip("\\/")
        if not directory or ":" in directory:
            return match.group(0)
        if directory.startswith(("./", ".\\", "../", "..\\")):
            return directory
        return f"./{directory}"

    return _GO_FILE_GLOB.sub(replace, command)


def normalize_windows_test_commands(commands: Sequence[str]) -> list[str]:
    """Normalize each published Windows test command without mutating input."""
    return [normalize_windows_go_test_file_globs(command) for command in commands]