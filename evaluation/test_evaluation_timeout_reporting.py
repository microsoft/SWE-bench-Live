import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


launch = types.ModuleType("launch")
launch_core = types.ModuleType("launch.core")
launch_runtime = types.ModuleType("launch.core.runtime")
launch_runtime.SetupRuntime = object
launch_scripts = types.ModuleType("launch.scripts")
launch_parser = types.ModuleType("launch.scripts.parser")
launch_parser.run_parser = lambda _parser, _log: {}
datasets = types.ModuleType("datasets")
datasets.load_dataset = lambda *_args, **_kwargs: None
sys.modules.setdefault("launch", launch)
sys.modules.setdefault("launch.core", launch_core)
sys.modules.setdefault("launch.core.runtime", launch_runtime)
sys.modules.setdefault("launch.scripts", launch_scripts)
sys.modules.setdefault("launch.scripts.parser", launch_parser)
sys.modules.setdefault("datasets", datasets)

MODULE_PATH = Path(__file__).with_name("evaluation.py")
SPEC = importlib.util.spec_from_file_location("timeout_evaluation", MODULE_PATH)
assert SPEC and SPEC.loader
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


class _Result:
    def __init__(self, exit_code=0, output=""):
        self.output = output
        self.metadata = types.SimpleNamespace(exit_code=exit_code)


class _Container:
    def __init__(self):
        self.commands = []
        self.cleaned = False

    def apply_patch(self, *_args, **_kwargs):
        return True

    def send_command(self, command):
        self.commands.append(command)
        if command == "test-command":
            return _Result(124, "**Exited due to timeout; container removed**")
        if command == "print-command":
            raise AssertionError("print command must not run after a terminal timeout")
        return _Result()

    def cleanup(self):
        self.cleaned = True


class TimeoutReportingTests(unittest.TestCase):
    def setUp(self):
        self.original_runtime = EVALUATION.SetupRuntime
        self.container = _Container()
        container = self.container

        class _Runtime:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                return container

        EVALUATION.SetupRuntime = _Runtime

    def tearDown(self):
        EVALUATION.SetupRuntime = self.original_runtime

    def test_timeout_exit_code_is_detected(self):
        self.assertTrue(EVALUATION.command_timed_out(_Result(124)))
        self.assertFalse(EVALUATION.command_timed_out(_Result(1)))

    def test_timeout_skips_follow_up_command_and_writes_unresolved_report(self):
        instance = {
            "instance_id": "example__timeout",
            "docker_image": "example-image",
            "rebuild_cmds": [],
            "test_cmds": ["test-command"],
            "print_cmds": ["print-command"],
            "test_patch": "test patch",
            "pred_patch": "solution patch",
            "PASS_TO_PASS": ["TestStable"],
            "FAIL_TO_PASS": ["TestRegression"],
        }
        with tempfile.TemporaryDirectory() as output_dir:
            report = EVALUATION.run_instance(instance, "windows", output_dir, overwrite=True)
            report_path = Path(output_dir) / instance["instance_id"] / "report.json"
            on_disk = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertFalse(report["resolved"])
        self.assertEqual(report["failure_class"], "timeout")
        self.assertEqual(on_disk["failure_class"], "timeout")
        self.assertEqual(self.container.commands, ["test-command"])
        self.assertTrue(self.container.cleaned)


if __name__ == "__main__":
    unittest.main()
