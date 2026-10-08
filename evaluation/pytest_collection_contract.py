"""Fail closed when pytest ends during collection without executing tests."""

from __future__ import annotations

import json
from typing import Any


class PytestCollectionUnavailable(RuntimeError):
    """The published command could not execute any pytest test case."""

    def __init__(self, evidence: dict[str, int]) -> None:
        self.evidence = evidence
        super().__init__(
            "pytest exited during collection before executing any test cases; "
            "candidate grading is unavailable"
        )


def pytest_collection_failed_without_test_execution(log: str) -> dict[str, int] | None:
    """Return evidence for a pytest-json-report collection-only failure.

    The predicate is intentionally narrow: pytest must report collection exit
    code 2, a positive collected count, zero executed tests, and at least one
    failed collector. Ordinary test failures remain eligible for normal grading.
    """
    text = str(log or "").strip()
    if not text.startswith("{"):
        return None
    try:
        report: Any = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(report, dict):
        return None
    try:
        exitcode = int(report.get("exitcode"))
    except (TypeError, ValueError):
        return None
    summary = report.get("summary")
    tests = report.get("tests")
    collectors = report.get("collectors")
    if not isinstance(summary, dict) or not isinstance(tests, list) or not isinstance(collectors, list):
        return None
    try:
        total = int(summary.get("total", -1))
        collected = int(summary.get("collected", 0))
    except (TypeError, ValueError):
        return None
    failed_collectors = sum(
        1
        for collector in collectors
        if isinstance(collector, dict) and str(collector.get("outcome", "")).lower() == "failed"
    )
    if exitcode != 2 or total != 0 or tests or collected <= 0 or failed_collectors <= 0:
        return None
    return {
        "exitcode": exitcode,
        "collected": collected,
        "failed_collectors": failed_collectors,
    }


def require_pytest_test_execution(log: str) -> None:
    """Raise instead of converting a collection-only run into a failed patch."""
    evidence = pytest_collection_failed_without_test_execution(log)
    if evidence is not None:
        raise PytestCollectionUnavailable(evidence)