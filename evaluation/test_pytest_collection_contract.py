from __future__ import annotations

import json
import unittest

from evaluation.pytest_collection_contract import (
    PytestCollectionUnavailable,
    pytest_collection_failed_without_test_execution,
    require_pytest_test_execution,
)


class PytestCollectionContractTests(unittest.TestCase):
    def test_collection_only_failure_is_reported_as_unavailable(self) -> None:
        log = json.dumps(
            {
                "exitcode": 2,
                "summary": {"total": 0, "collected": 3237},
                "tests": [],
                "collectors": [
                    {"nodeid": "sdk/unrelated/test_missing_dep.py", "outcome": "failed"},
                    {"nodeid": "sdk/identity", "outcome": "passed"},
                ],
            }
        )
        self.assertEqual(
            pytest_collection_failed_without_test_execution(log),
            {"exitcode": 2, "collected": 3237, "failed_collectors": 1},
        )
        with self.assertRaises(PytestCollectionUnavailable) as raised:
            require_pytest_test_execution(log)
        self.assertEqual(raised.exception.evidence["collected"], 3237)

    def test_executed_pytest_failure_remains_gradable(self) -> None:
        log = json.dumps(
            {
                "exitcode": 1,
                "summary": {"total": 2, "collected": 2, "failed": 1, "passed": 1},
                "tests": [
                    {"nodeid": "sdk/identity/test_broker.py::test_default", "outcome": "passed"},
                    {"nodeid": "sdk/identity/test_broker.py::test_fallback", "outcome": "failed"},
                ],
                "collectors": [],
            }
        )
        self.assertIsNone(pytest_collection_failed_without_test_execution(log))
        require_pytest_test_execution(log)

    def test_non_json_and_incomplete_reports_are_not_reclassified(self) -> None:
        self.assertIsNone(pytest_collection_failed_without_test_execution("pytest output"))
        self.assertIsNone(
            pytest_collection_failed_without_test_execution(
                json.dumps({"exitcode": 2, "summary": {"total": 0, "collected": 2}, "tests": []})
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)