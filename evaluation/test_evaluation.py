import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# The unit tests exercise evaluator control flow only; avoid requiring the
# optional Hugging Face client used exclusively by dataset-backed CLI paths.
if "datasets" not in sys.modules:
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *_args, **_kwargs: None
    sys.modules["datasets"] = datasets

from evaluation import evaluation as evaluator


class _Container:
    def __init__(self, test_output: str):
        self.test_output = test_output
        self.commands: list[str] = []
        self.cleaned = False

    def apply_patch(self, *_args, **_kwargs):
        return True

    def send_command(self, command: str):
        self.commands.append(command)
        return SimpleNamespace(output=self.test_output)

    def cleanup(self):
        self.cleaned = True


class PublishedTestCommandUnavailableTests(unittest.TestCase):
    def test_extracts_unique_powershell_command_not_found_names(self):
        output = """
        yarn : The term 'yarn' is not recognized as the name of a cmdlet, function,
        script file, or operable program.
        node : The term 'node' is not recognized as the name of a cmdlet, function,
        script file, or operable program.
        yarn : The term 'yarn' is not recognized as the name of a cmdlet, function,
        script file, or operable program.
        """
        self.assertEqual(
            evaluator.missing_windows_command_executables(output),
            ["yarn", "node"],
        )

    def test_does_not_match_unrelated_nonzero_test_output(self):
        output = "FAILED src/example.test.ts\nAssertionError: expected true to be false"
        self.assertEqual(evaluator.missing_windows_command_executables(output), [])

    def test_windows_missing_executable_skips_parser_follow_up(self):
        container = _Container(
            "yarn : The term 'yarn' is not recognized as the name of a cmdlet, function,\n"
            "script file, or operable program."
        )
        class _Runtime:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                return container

        with patch.object(evaluator, "SetupRuntime", _Runtime):
            with self.assertRaises(evaluator.PublishedTestCommandUnavailableError) as caught:
                evaluator.evaluate_instance(
                    "fixture__missing_yarn",
                    "fixture:image",
                    "",
                    "yarn vitest run",
                    "print-test-log",
                    "test patch",
                    "solution patch",
                    "vitest",
                    "windows",
                    tempfile.mkdtemp(),
                )
        self.assertEqual(caught.exception.executables, ["yarn"])
        self.assertTrue(container.cleaned)
        self.assertNotIn("print-test-log", container.commands)

    def test_unavailable_command_writes_unresolved_report(self):
        instance = {
            "instance_id": "fixture__missing_yarn",
            "docker_image": "fixture:image",
            "rebuild_cmds": [],
            "test_cmds": ["yarn vitest run"],
            "print_cmds": ["print-test-log"],
            "test_patch": "test patch",
            "pred_patch": "solution patch",
            "PASS_TO_PASS": [],
            "FAIL_TO_PASS": ["target test"],
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                evaluator,
                "evaluate_instance",
                side_effect=evaluator.PublishedTestCommandUnavailableError(["yarn", "node"]),
            ):
                report = evaluator.run_instance(instance, "windows", directory, True)
            on_disk = json.loads(
                (Path(directory) / "fixture__missing_yarn" / "report.json").read_text(encoding="utf-8")
            )
        self.assertEqual(report, on_disk)
        self.assertIsNone(report["resolved"])
        self.assertEqual(report["grading_status"], "unavailable")
        self.assertEqual(report["failure_class"], "published_test_command_executable_missing")
        self.assertEqual(report["missing_command_executables"], ["yarn", "node"])


if __name__ == "__main__":
    unittest.main()