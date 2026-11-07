import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import evaluation


class RandomGoSubtestIdentifierTests(unittest.TestCase):
    def test_one_random_subtest_reconciles(self):
        expected = {
            "TestImageTagInvalidSourceImageName/invalidTag/"
            "WeZSTWsYMDFmvGtDtajPqxTVeJdPCEEmnlDmPBRQrNLOdvmRFClcFWwyHhccVfxqskLDluEQZgewKneZrQcRxbalfxdpVarKdBZbqsczHAnXMfdJVHyyKmtfZ"
        }
        observed = {
            "TestImageTagInvalidSourceImageName/invalidTag/"
            "DlhPzjKnjxiutrBQTzIlBtNKfLggztnGyWMdJDLDqnCtvNXJabyWnQsQPHgRcttyvmpULdlFotDnMuFQrRnqsDLdBCfkjoRHGtseeDglUCPWetKddkRiXulKu": "pass"
        }
        reconciled, matches, ambiguous = evaluation.reconcile_random_go_subtest_statuses(expected, observed)
        self.assertEqual(reconciled, {next(iter(expected)): "pass"})
        self.assertEqual(matches, 1)
        self.assertEqual(ambiguous, 0)

    def test_two_random_subtests_remain_ambiguous(self):
        expected = {
            "TestImageTagInvalidSourceImageName/invalidTag/" + "A" * 121,
            "TestImageTagInvalidSourceImageName/invalidTag/" + "B" * 121,
        }
        observed = {
            "TestImageTagInvalidSourceImageName/invalidTag/" + "C" * 121: "pass",
        }
        reconciled, matches, ambiguous = evaluation.reconcile_random_go_subtest_statuses(expected, observed)
        self.assertEqual(reconciled, {})
        self.assertEqual(matches, 0)
        self.assertEqual(ambiguous, 2)

    def test_short_alphabetic_subtest_is_exact_only(self):
        expected = {"pkg.Test/short-name"}
        observed = {"pkg.Test/other-name": "pass"}
        reconciled, matches, ambiguous = evaluation.reconcile_random_go_subtest_statuses(expected, observed)
        self.assertEqual(reconciled, {})
        self.assertEqual(matches, 0)
        self.assertEqual(ambiguous, 0)

    def test_failure_remains_failure(self):
        instance = {
            "instance_id": "synthetic__random-go-failure",
            "docker_image": "unused",
            "rebuild_cmds": [],
            "test_cmds": [],
            "print_cmds": [],
            "test_patch": "test patch",
            "pred_patch": "candidate patch",
            "log_parser": "custom",
            "PASS_TO_PASS": [],
            "FAIL_TO_PASS": [
                "TestImageTagInvalidSourceImageName/invalidTag/" + "A" * 121
            ],
        }
        observed = {
            "TestImageTagInvalidSourceImageName/invalidTag/" + "B" * 121: "fail"
        }
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(evaluation, "evaluate_instance", return_value=observed):
                report = evaluation.run_instance(instance, "linux", output_dir, True)
        self.assertFalse(report["resolved"])
        self.assertEqual(report["FAIL_TO_PASS"]["failure"], instance["FAIL_TO_PASS"])


if __name__ == "__main__":
    unittest.main()