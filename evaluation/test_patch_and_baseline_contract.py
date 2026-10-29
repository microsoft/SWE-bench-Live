import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import evaluation


class FakeRuntime:
    def __init__(self, baseline_output=""):
        self.baseline_output = baseline_output
        self.commands = []
        self.cleaned_up = False

    def send_command(self, command):
        self.commands.append(command)
        if "git status --porcelain" in command:
            return SimpleNamespace(
                output=self.baseline_output,
                metadata=SimpleNamespace(exit_code=0),
            )
        return SimpleNamespace(output="", metadata=SimpleNamespace(exit_code=0))

    def cleanup(self):
        self.cleaned_up = True


def _instance():
    return {
        "instance_id": "synthetic__patch-contract",
        "docker_image": "unused",
        "rebuild_cmds": ["rebuild-command"],
        "test_cmds": ["test-command"],
        "print_cmds": ["print-command"],
        "test_patch": "test patch",
        "pred_patch": "candidate patch",
        "log_parser": "custom",
        "PASS_TO_PASS": [],
        "FAIL_TO_PASS": [],
    }


class PatchAndBaselineContractTests(unittest.TestCase):
    def _run_instance(self, runtime):
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(
                evaluation.SetupRuntime,
                "from_launch_image",
                return_value=runtime,
            ), patch.object(
                evaluation,
                "evaluate_instance",
                side_effect=evaluation.EvaluationInfrastructureError(
                    "dirty_official_image_baseline", "synthetic dirty baseline"
                ),
            ):
                report = evaluation.run_instance(
                    _instance(), "linux", output_dir, overwrite=True
                )
                saved = json.loads(
                    (Path(output_dir) / "synthetic__patch-contract" / "report.json").read_text(
                        encoding="utf-8"
                    )
                )
        return report, saved

    def test_dirty_official_baseline_is_not_a_candidate_failure(self):
        runtime = FakeRuntime(baseline_output=" M generated.txt\n")
        with self.assertRaises(evaluation.EvaluationInfrastructureError) as ctx:
            evaluation._assert_clean_baseline(
                runtime, "linux", "synthetic__patch-contract"
            )
        self.assertEqual(ctx.exception.failure_class, "dirty_official_image_baseline")

    def test_dirty_baseline_report_is_not_scored_as_task_failure(self):
        runtime = FakeRuntime()
        report, saved = self._run_instance(runtime)
        self.assertIsNone(report["resolved"])
        self.assertEqual(report["grading_status"], "not_evaluated")
        self.assertEqual(report["failure_class"], "dirty_official_image_baseline")
        self.assertEqual(saved, report)

    def test_dirty_baseline_cleans_up_and_stops_before_patch_commands(self):
        runtime = FakeRuntime(baseline_output=" M generated.txt\n")
        with self.assertRaises(evaluation.EvaluationInfrastructureError) as ctx:
            evaluation._assert_clean_baseline(
                runtime, "linux", "synthetic__patch-contract"
            )
        self.assertEqual(ctx.exception.failure_class, "dirty_official_image_baseline")
        self.assertTrue(runtime.cleaned_up)
        self.assertEqual(len(runtime.commands), 1)
        self.assertIn("git status --porcelain --untracked-files=no", runtime.commands[0])

    def test_infrastructure_results_are_counted_separately(self):
        report = {
            "instance_id": "synthetic__patch-contract",
            "resolved": None,
            "grading_status": "not_evaluated",
            "failure_class": "dirty_official_image_baseline",
        }
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(evaluation, "run_instance", return_value=report):
                results = evaluation.run_instances(
                    [_instance()], "linux", 1, output_dir, overwrite=True
                )
        self.assertEqual(results["success"], 0)
        self.assertEqual(results["failure"], 0)
        self.assertEqual(results["error"], 0)
        self.assertEqual(results["infrastructure_error"], 1)
        self.assertEqual(
            results["infrastructure_error_ids"], ["synthetic__patch-contract"]
        )


if __name__ == "__main__":
    unittest.main()