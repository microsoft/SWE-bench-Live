import sys
import tempfile
import types
import unittest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *_args, **_kwargs: None))
if "launch.core.runtime" not in sys.modules:
    launch_module = types.ModuleType("launch")
    core_module = types.ModuleType("launch.core")
    runtime_module = types.ModuleType("launch.core.runtime")
    runtime_module.SetupRuntime = object
    sys.modules["launch"] = launch_module
    sys.modules["launch.core"] = core_module
    sys.modules["launch.core.runtime"] = runtime_module
if "launch.scripts.parser" not in sys.modules:
    scripts_module = types.ModuleType("launch.scripts")
    parser_module = types.ModuleType("launch.scripts.parser")
    parser_module.run_parser = lambda *_args, **_kwargs: {}
    sys.modules["launch.scripts"] = scripts_module
    sys.modules["launch.scripts.parser"] = parser_module

from . import validation


class _Result:
    def __init__(self, output="", exit_code=0):
        self.output = output
        self.metadata = types.SimpleNamespace(exit_code=exit_code)


class _Container:
    def __init__(self, roots):
        self.roots = list(roots)
        self.commands = []
        self.cleaned = False

    def apply_patch(self, *_args, **_kwargs):
        return True

    def send_command(self, command):
        self.commands.append(command)
        if command == "git rev-parse --show-toplevel":
            return _Result(self.roots.pop(0) + "\n")
        if command == "print-results":
            return _Result("TestExample pass\n")
        return _Result()

    def cleanup(self):
        self.cleaned = True


def _runtime_for(containers):
    class _Runtime:
        @classmethod
        def from_launch_image(cls, *_args, **_kwargs):
            return containers.pop(0)
    return _Runtime


def _validate_with(containers, platform="windows"):
    original_runtime = validation.SetupRuntime
    original_parser = validation.run_parser
    validation.SetupRuntime = _runtime_for(containers)
    validation.run_parser = lambda *_args: {"TestExample": "pass"}
    try:
        with tempfile.TemporaryDirectory() as output_dir:
            return validation.validate_instance(
                instance_id="moby__moby-49938",
                image="example-image",
                rebuild_cmd="rebuild",
                test_cmd="run-tests",
                print_cmd="print-results",
                test_patch="test patch",
                solution_patch="solution patch",
                parser="example",
                platform=platform,
                output_dir=output_dir,
            )
    finally:
        validation.SetupRuntime = original_runtime
        validation.run_parser = original_parser


class ValidationRootIdentityTests(unittest.TestCase):
    def test_validation_rejects_changed_git_root_before_test_or_print(self):
        container = _Container(
            roots=[r"C:\\testbed", r"C:\\go\\src\\github.com\\docker\\docker"],
        )
        with self.assertRaisesRegex(validation.ValidationInfrastructureError, "not patched root"):
            _validate_with([container])
        self.assertTrue(container.cleaned)
        self.assertNotIn("run-tests", container.commands)
        self.assertNotIn("print-results", container.commands)

    def test_validation_accepts_equivalent_windows_root_casing(self):
        containers = [
            _Container(roots=[r"C:\\TestBed", r"c:\\testbed"]),
            _Container(roots=[r"C:\\TestBed", r"c:\\testbed"]),
            _Container(roots=[r"C:\\TestBed", r"c:\\testbed"]),
            _Container(roots=[r"C:\\TestBed", r"c:\\testbed"]),
        ]
        result = _validate_with(containers)
        self.assertEqual(result["FAIL_TO_PASS"], [])
        self.assertTrue(all(container.cleaned for container in containers))
        self.assertTrue(all("run-tests" in container.commands for container in containers))


if __name__ == "__main__":
    unittest.main()
