from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


class EvaluationPytestCollectionTests(unittest.TestCase):
    def test_run_instance_returns_incomplete_for_collection_only_pytest(self) -> None:
        launch = types.ModuleType("launch")
        launch_core = types.ModuleType("launch.core")
        launch_runtime = types.ModuleType("launch.core.runtime")
        launch_runtime.SetupRuntime = object
        launch_scripts = types.ModuleType("launch.scripts")
        launch_parser = types.ModuleType("launch.scripts.parser")
        launch_parser.run_parser = lambda *_args, **_kwargs: {}
        datasets = types.ModuleType("datasets")
        datasets.load_dataset = lambda *_args, **_kwargs: []
        saved_modules = {
            name: sys.modules.get(name)
            for name in ("launch", "launch.core", "launch.core.runtime", "launch.scripts", "launch.scripts.parser", "datasets")
        }
        sys.modules.update(
            {
                "launch": launch,
                "launch.core": launch_core,
                "launch.core.runtime": launch_runtime,
                "launch.scripts": launch_scripts,
                "launch.scripts.parser": launch_parser,
                "datasets": datasets,
            }
        )
        sys.modules.pop("evaluation.evaluation", None)
        evaluation = importlib.import_module("evaluation.evaluation")
        log = json.dumps(
            {
                "exitcode": 2,
                "summary": {"total": 0, "collected": 11},
                "tests": [],
                "collectors": [{"nodeid": "unrelated/test_missing_dep.py", "outcome": "failed"}],
            }
        )

        class _Result:
            def __init__(self, output: str = "") -> None:
                self.output = output

        class _Container:
            def apply_patch(self, *_args, **_kwargs):
                return True

            def send_command(self, command: str):
                if command == "print-command":
                    return _Result(log)
                return _Result()

            def cleanup(self):
                return None

        class _Runtime:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                return _Container()

        original_runtime = evaluation.SetupRuntime
        evaluation.SetupRuntime = _Runtime
        instance = {
            "instance_id": "fixture__collection_only",
            "docker_image": "fixture:image",
            "rebuild_cmds": [],
            "test_cmds": ["test-command"],
            "print_cmds": ["print-command"],
            "test_patch": "test patch",
            "pred_patch": "solution patch",
            "log_parser": "pytest",
            "PASS_TO_PASS": [],
            "FAIL_TO_PASS": ["sdk/identity/test_broker.py::test_default"],
        }
        try:
            with tempfile.TemporaryDirectory() as directory:
                report = evaluation.run_instance(instance, "windows", directory, True)
                on_disk = json.loads(
                    (Path(directory) / instance["instance_id"] / "report.json").read_text(encoding="utf-8")
                )
        finally:
            evaluation.SetupRuntime = original_runtime
            sys.modules.pop("evaluation.evaluation", None)
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        self.assertEqual(report, on_disk)
        self.assertIsNone(report["resolved"])
        self.assertEqual(report["grading_status"], "unavailable")
        self.assertEqual(report["failure_class"], "pytest_collection_failed_without_test_execution")
        self.assertEqual(report["pytest_collection_failure"], {"exitcode": 2, "collected": 11, "failed_collectors": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)