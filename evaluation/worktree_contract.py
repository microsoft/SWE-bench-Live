from __future__ import annotations

import re
from typing import Iterable, Literal, Protocol


class CommandMetadata(Protocol):
    exit_code: int


class CommandResult(Protocol):
    metadata: CommandMetadata


class CommandRunner(Protocol):
    def send_command(self, command: str) -> CommandResult:
        ...


WINDOWS_PATCH_ROOT = r"C:\testbed"

# Only absolute Windows directory changes are relevant here. A command that
# stays in C:\testbed (including a project subdirectory) still sees patches
# applied at the repository root. An external checkout does not.
_WINDOWS_DIRECTORY_CHANGE = re.compile(
    r"(?:Set-Location|cd)\s+(?:-LiteralPath\s+)?(?:['\"])?"
    r"(?P<path>[A-Za-z]:\\[^;'\"\s]+)",
    re.IGNORECASE,
)


def _normalize_windows_path(path: str) -> str:
    return path.replace("/", "\\").rstrip("\\").lower()


def _is_under_patch_root(path: str) -> bool:
    normalized = _normalize_windows_path(path)
    root = _normalize_windows_path(WINDOWS_PATCH_ROOT)
    return normalized == root or normalized.startswith(root + "\\")


def get_windows_evaluation_worktree(commands: Iterable[str]) -> str:
    """Return the checkout that Windows patch/build/test phases must share.

    Windows launch images normally evaluate directly from ``C:\\testbed``.
    Some images also contain a separate source checkout and their published
    command metadata enters it explicitly. Applying test and solution patches
    at ``C:\\testbed`` while commands execute in that other checkout makes a
    candidate invisible to the evaluator.

    Commands under ``C:\\testbed`` retain the normal root because patches are
    repository-relative. Exactly one external absolute checkout is supported;
    multiple distinct roots are ambiguous and fail clearly instead of silently
    grading a different checkout.
    """
    external_roots: list[str] = []
    seen: set[str] = set()
    for command in commands:
        for match in _WINDOWS_DIRECTORY_CHANGE.finditer(str(command or "")):
            path = match.group("path")
            normalized = _normalize_windows_path(path)
            if _is_under_patch_root(path) or normalized in seen:
                continue
            seen.add(normalized)
            external_roots.append(path.rstrip("\\/"))
    if not external_roots:
        return WINDOWS_PATCH_ROOT
    if len(external_roots) == 1:
        return external_roots[0]
    raise ValueError(
        "Windows evaluation commands select multiple external worktrees: "
        + ", ".join(external_roots)
    )


def set_windows_evaluation_worktree(
    container: CommandRunner,
    platform: Literal["windows", "linux"],
    worktree: str,
) -> None:
    """Set the command and patch context to the resolved Windows checkout."""
    if platform != "windows":
        return
    escaped_worktree = worktree.replace("'", "''")
    result = container.send_command(
        f"Set-Location -LiteralPath '{escaped_worktree}'"
    )
    if int(result.metadata.exit_code) != 0:
        raise RuntimeError(
            "Could not set the Windows evaluator working directory to "
            f"{worktree}."
        )
