"""Windows-only test-name matching helpers for SWE-bench evaluation."""

import re
from typing import Literal

TestStatus = Literal["pass", "fail", "skip"]


def collapse_adjacent_repeated_chars(test_name: str) -> str:
    """Remove adjacent duplicate characters from a Windows test-name artifact.

    Windows console capture can duplicate a character at a wrapping boundary
    without inserting whitespace, for example ``github.coom`` instead of
    ``github.com`` or ``TestLLoad`` instead of ``TestLoad``.  This function is
    only used as a fallback after exact matching and never changes the parsed
    status keys in place.
    """
    if not test_name:
        return test_name
    collapsed = [test_name[0]]
    for char in test_name[1:]:
        if char != collapsed[-1]:
            collapsed.append(char)
    return "".join(collapsed)


def build_windows_repeated_char_index(
    status: dict[str, TestStatus],
) -> dict[str, str]:
    """Build a unique, fail-closed lookup for repeated-character artifacts."""
    index: dict[str, str] = {}
    ambiguous: set[str] = set()
    for actual_name in status:
        normalized_name = collapse_adjacent_repeated_chars(actual_name)
        if normalized_name in ambiguous:
            continue
        existing_name = index.get(normalized_name)
        if existing_name is None:
            index[normalized_name] = actual_name
        elif existing_name != actual_name:
            ambiguous.add(normalized_name)
            index.pop(normalized_name, None)
    return index


def _decode_go_escaped_test_name(name: str) -> str:
    """Decode the JSON-style unicode escapes used in Go subtest metadata."""
    try:
        return re.sub(
            r"\\u([0-9a-fA-F]{4})",
            lambda match: chr(int(match.group(1), 16)),
            name,
        )
    except (TypeError, ValueError):
        return name


def resolve_expected_test_name(
    expected_name: str,
    status: dict[str, TestStatus],
    platform: str,
    repeated_char_index: dict[str, str] | None = None,
) -> str | None:
    """Resolve an expected name to a parsed test name, if it is unambiguous.

    Exact matching is always preferred.  The duplicate-character fallback is
    restricted to Windows and accepts only one actual name for a normalized
    key; collisions remain unmatched rather than being guessed.
    """
    if expected_name in status:
        return expected_name
    if platform != "windows":
        return None
    decoded_expected = _decode_go_escaped_test_name(expected_name)
    if decoded_expected in status:
        return decoded_expected
    if repeated_char_index is None:
        repeated_char_index = build_windows_repeated_char_index(status)
    normalized_expected = collapse_adjacent_repeated_chars(decoded_expected)
    if normalized_expected == expected_name:
        return None
    return repeated_char_index.get(normalized_expected)


def classify_expected_tests(
    expected_tests: list[str],
    status: dict[str, TestStatus],
    platform: str,
) -> dict[str, list[str]]:
    """Classify expected names against parsed statuses without fuzzy guessing."""
    repeated_char_index = (
        build_windows_repeated_char_index(status) if platform == "windows" else None
    )
    classified = {"success": [], "failure": []}
    for expected_name in expected_tests:
        actual_name = resolve_expected_test_name(
            expected_name,
            status,
            platform,
            repeated_char_index,
        )
        if actual_name is None:
            continue
        actual_status = status[actual_name].lower()
        if "pass" in actual_status:
            classified["success"].append(expected_name)
        elif "fail" in actual_status:
            classified["failure"].append(expected_name)
    return classified