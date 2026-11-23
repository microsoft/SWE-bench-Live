"""Integrity checks for numbered TAP captures produced by Windows tasks."""

from __future__ import annotations

import re

_NUMBERED_PLAN = re.compile(
    r"^\s*1\.\.(?P<count>[1-9]\d*)(?:\s*(?:#.*)?)?$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_NUMBERED_RESULT = re.compile(
    r"^\s*(?:ok|not\s+ok)\s+(?P<identifier>[1-9]\d*)(?=\s|$)",
    flags=re.IGNORECASE | re.MULTILINE,
)


def assert_complete_numbered_tap_capture(log: str) -> bool:
    """Reject a single-plan TAP capture if physical assertion records are missing.

    A Windows test command can produce a TAP stream whose writer output is
    interleaved by concurrent process logging. When a log has one top-level
    ``1..N`` plan, every integer assertion label from one through ``N`` must
    occur exactly once in physical ``ok``/``not ok`` lines. The check is a
    no-op for ordinary logs and for nested TAP streams with multiple plans.
    """
    plans = [int(match.group("count")) for match in _NUMBERED_PLAN.finditer(log)]
    if len(plans) != 1:
        return False

    planned_count = plans[0]
    seen = bytearray(planned_count + 1)
    result_count = 0
    duplicate_ids: list[int] = []
    out_of_range_ids: list[int] = []
    for match in _NUMBERED_RESULT.finditer(log):
        identifier = int(match.group("identifier"))
        if identifier > planned_count:
            if len(out_of_range_ids) < 5:
                out_of_range_ids.append(identifier)
            continue
        if seen[identifier]:
            if len(duplicate_ids) < 5:
                duplicate_ids.append(identifier)
            continue
        seen[identifier] = 1
        result_count += 1

    if result_count == planned_count and not duplicate_ids and not out_of_range_ids:
        return True

    missing_ids: list[int] = []
    for identifier in range(1, planned_count + 1):
        if not seen[identifier]:
            missing_ids.append(identifier)
            if len(missing_ids) == 5:
                break
    raise ValueError(
        "Incomplete numbered TAP capture: "
        f"plan=1..{planned_count}, unique_records={result_count}, "
        f"missing={missing_ids}, duplicates={duplicate_ids}, "
        f"out_of_range={out_of_range_ids}"
    )