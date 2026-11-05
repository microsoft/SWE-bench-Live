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


def _edit_distance(left: str, right: str) -> int:
    """Return Levenshtein distance for choosing among artifact candidates."""
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def _windows_expected_variants(name: str) -> list[str]:
    """Return conservative cleanup variants for Windows metadata artifacts."""
    decoded = _decode_go_escaped_test_name(name)
    variants = [decoded]
    stripped = decoded.removeprefix(": ")
    if stripped != decoded:
        variants.append(stripped)
    if stripped.endswith("x"):
        variants.append(stripped[:-1])
    return variants


def resolve_expected_test_name(
    expected_name: str,
    status: dict[str, TestStatus],
    platform: str,
    repeated_char_index: dict[str, str] | None = None,
) -> str | None:
    """Resolve an expected name to a parsed test name, if it is unambiguous.

    Exact matching is always preferred. The duplicate-character fallback is
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


def _assign_ambiguous_windows_names(
    expected_names: list[str],
    status: dict[str, TestStatus],
    used_actual: set[str],
) -> dict[str, str]:
    """Assign same-normalization artifacts with a unique minimum-cost matching."""
    groups: dict[str, list[str]] = {}
    for expected_name in expected_names:
        keys = {
            collapse_adjacent_repeated_chars(variant)
            for variant in _windows_expected_variants(expected_name)
        }
        for key in keys:
            groups.setdefault(key, []).append(expected_name)

    assignments: dict[str, str] = {}
    for key, group_expected in groups.items():
        group_expected = list(dict.fromkeys(group_expected))
        candidates = [
            actual_name for actual_name in status
            if actual_name not in used_actual
            and collapse_adjacent_repeated_chars(
                _decode_go_escaped_test_name(actual_name)
            ) == key
        ]
        if not candidates or len(candidates) < len(group_expected):
            continue
        best_cost: int | None = None
        best: list[tuple[str, str]] = []

        def search(index: int, remaining: list[str], cost: int, pairs: list[tuple[str, str]]):
            nonlocal best_cost, best
            if index == len(group_expected):
                if best_cost is None or cost < best_cost:
                    best_cost, best = cost, list(pairs)
                elif cost == best_cost:
                    best = []  # tie: fail closed for this group
                return
            expected_name = group_expected[index]
            for actual_name in remaining:
                distance = min(
                    _edit_distance(variant, actual_name)
                    for variant in _windows_expected_variants(expected_name)
                )
                if best_cost is not None and cost + distance > best_cost:
                    continue
                search(
                    index + 1,
                    [item for item in remaining if item != actual_name],
                    cost + distance,
                    pairs + [(expected_name, actual_name)],
                )

        search(0, candidates, 0, [])
        if best:
            for expected_name, actual_name in best:
                assignments[expected_name] = actual_name
                used_actual.add(actual_name)
    return assignments


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
    unresolved: list[str] = []
    used_actual: set[str] = set()
    for expected_name in expected_tests:
        actual_name = resolve_expected_test_name(
            expected_name,
            status,
            platform,
            repeated_char_index,
        )
        if actual_name is None:
            unresolved.append(expected_name)
            continue
        used_actual.add(actual_name)
        actual_status = status[actual_name].lower()
        if "pass" in actual_status:
            classified["success"].append(expected_name)
        elif "fail" in actual_status:
            classified["failure"].append(expected_name)

    if platform == "windows" and unresolved:
        assignments = _assign_ambiguous_windows_names(unresolved, status, used_actual)
        for expected_name in unresolved:
            actual_name = assignments.get(expected_name)
            if actual_name is None:
                continue
            actual_status = status[actual_name].lower()
            if "pass" in actual_status:
                classified["success"].append(expected_name)
            elif "fail" in actual_status:
                classified["failure"].append(expected_name)
    return classified