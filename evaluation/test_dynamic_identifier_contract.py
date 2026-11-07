import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

try:
    from evaluation import evaluation
except ImportError:  # unittest discovery invoked from evaluation/ itself
    import evaluation as evaluation


class DynamicIdentifierContractTests(unittest.TestCase):
    def test_runtime_volatile_identifiers_are_reconciled_one_to_one(self):
        instance = {
            "instance_id": "synthetic__dynamic-identifiers",
            "docker_image": "unused",
            "rebuild_cmds": [],
            "test_cmds": [],
            "print_cmds": [],
            "test_patch": "test patch",
            "pred_patch": "candidate patch",
            "log_parser": "custom",
            "PASS_TO_PASS": [
                "pkg.TestHTTP/http://127.0.0.1:31001",
                "pkg.TestHTTP/http://127.0.0.1:31002",
            ],
            "FAIL_TO_PASS": [
                "pkg.TestHTTP/multiaddr:/ip4/127.0.0.1/tcp/31003/http",
            ],
        }
        observed = {
            "pkg.TestHTTP/http://127.0.0.1:41001": "pass",
            "pkg.TestHTTP/http://127.0.0.1:41002": "pass",
            "pkg.TestHTTP/multiaddr:/ip4/127.0.0.1/tcp/41003/http": "pass",
            "pkg.TestHTTP/unrelated": "fail",
        }
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(evaluation, "evaluate_instance", return_value=observed):
                report = evaluation.run_instance(instance, "linux", output_dir, True)

        self.assertTrue(report["resolved"])
        self.assertTrue(report["evaluation_contract"]["valid"])
        self.assertEqual(report["evaluation_contract"]["matched_expected_tests"], 3)
        self.assertEqual(report["evaluation_contract"]["dynamic_identifier_matches"], 3)

    def test_unrelated_observed_failure_is_not_reclassified(self):
        instance = {
            "instance_id": "synthetic__dynamic-identifiers-unrelated",
            "docker_image": "unused",
            "rebuild_cmds": [],
            "test_cmds": [],
            "print_cmds": [],
            "test_patch": "test patch",
            "pred_patch": "candidate patch",
            "log_parser": "custom",
            "PASS_TO_PASS": ["pkg.TestHTTP/http://127.0.0.1:31001"],
            "FAIL_TO_PASS": [],
        }
        observed = {
            "pkg.TestHTTP/http://127.0.0.1:41001": "pass",
            "pkg.TestHTTP/unrelated": "fail",
        }
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(evaluation, "evaluate_instance", return_value=observed):
                report = evaluation.run_instance(instance, "linux", output_dir, True)

        self.assertTrue(report["resolved"])
        self.assertEqual(report["evaluation_contract"]["matched_expected_tests"], 1)

    def test_dynamic_group_count_mismatch_is_unmatched(self):
        expected = {
            "pkg.TestHTTP/http://127.0.0.1:31001",
            "pkg.TestHTTP/http://127.0.0.1:31002",
        }
        reconciled, dynamic_matches, ambiguous_expected = (
            evaluation.reconcile_runtime_identifier_statuses(
                expected,
                {"pkg.TestHTTP/http://127.0.0.1:41001": "pass"},
            )
        )

        self.assertEqual(reconciled, {})
        self.assertEqual(dynamic_matches, 0)
        self.assertEqual(ambiguous_expected, 2)

    def test_exact_dynamic_identifier_is_not_reused(self):
        exact = "pkg.TestHTTP/http://127.0.0.1:31001"
        expected = {exact, "pkg.TestHTTP/http://127.0.0.1:31002"}
        reconciled, dynamic_matches, ambiguous_expected = (
            evaluation.reconcile_runtime_identifier_statuses(expected, {exact: "pass"})
        )

        self.assertEqual(reconciled, {exact: "pass"})
        self.assertEqual(dynamic_matches, 0)
        self.assertEqual(ambiguous_expected, 1)

    def test_mixed_dynamic_outcomes_are_not_aggregated(self):
        expected = {
            "pkg.TestHTTP/http://127.0.0.1:31001",
            "pkg.TestHTTP/http://127.0.0.1:31002",
        }
        observed = {
            "pkg.TestHTTP/http://127.0.0.1:41001": "pass",
            "pkg.TestHTTP/http://127.0.0.1:41002": "fail",
        }
        reconciled, dynamic_matches, ambiguous_expected = (
            evaluation.reconcile_runtime_identifier_statuses(expected, observed)
        )

        self.assertEqual(reconciled, {})
        self.assertEqual(dynamic_matches, 0)
        self.assertEqual(ambiguous_expected, 2)


if __name__ == "__main__":
    unittest.main()