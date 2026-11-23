import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _Metadata:
    def __init__(self, exit_code: int = 0):
        self.exit_code = exit_code


class _Result:
    def __init__(self, output: str = "", exit_code: int = 0):
        self.output = output
        self.metadata = _Metadata(exit_code)


class _Container:
    def __init__(self):
        self.commands: list[str] = []
        self.patches: list[str] = []
        self.cleaned = False

    def send_command(self, command: str):
        self.commands.append(command)
        return _Result()

    def apply_patch(self, patch: str, verbose: bool = False):
        self.patches.append(patch)
        return True

    def cleanup(self):
        self.cleaned = True


class _Runtime:
    container: _Container | None = None

    @classmethod
    def from_launch_image(cls, *_args, **_kwargs):
        cls.container = _Container()
        return cls.container


class WindowsWorktreeEvaluatorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.setdefault("launch", types.ModuleType("launch"))
        sys.modules.setdefault("launch.core", types.ModuleType("launch.core"))
        runtime = types.ModuleType("launch.core.runtime")
        runtime.SetupRuntime = _Runtime
        sys.modules["launch.core.runtime"] = runtime
        sys.modules.setdefault("launch.scripts", types.ModuleType("launch.scripts"))
        parser = types.ModuleType("launch.scripts.parser")
        parser.run_parser = lambda *_args, **_kwargs: {}
        sys.modules["launch.scripts.parser"] = parser
        datasets = types.ModuleType("datasets")
        datasets.load_dataset = lambda *_args, **_kwargs: None
        sys.modules["datasets"] = datasets
        cls.evaluation = importlib.import_module("evaluation.evaluation")

    def test_moby_secondary_checkout_is_selected_before_patch_and_test(self):
        with tempfile.TemporaryDirectory() as output_dir:
            self.evaluation.evaluate_instance(
                instance_id="moby__moby-49938",
                image="example-image",
                rebuild_cmd=(
                    r"Set-Location C:\go\src\github.com\docker\docker; "
                    r"go build ./cmd/..."
                ),
                test_cmd=r"go test -json -v ./... > reports\go-test-results.json",
                print_cmd=r"cat reports\go-test-results.json",
                test_patch="test-patch",
                solution_patch="solution-patch",
                parser="go_json",
                platform="windows",
                output_dir=output_dir,
            )
            container = _Runtime.container
            self.assertIsNotNone(container)
            assert container is not None
            root = r"Set-Location -LiteralPath 'C:\go\src\github.com\docker\docker'"
            self.assertEqual(container.commands[0], root)
            self.assertEqual(container.patches, ["test-patch", "solution-patch"])
            self.assertEqual(container.commands[1], r"Set-Location C:\go\src\github.com\docker\docker; go build ./cmd/...")
            self.assertEqual(container.commands[2], root)
            self.assertEqual(container.commands[3], r"Set-Location -LiteralPath 'C:\go\src\github.com\docker\docker'")
            self.assertEqual(container.commands[4], r"go test -json -v ./... > reports\go-test-results.json")
            self.assertEqual(container.commands[5], r"cat reports\go-test-results.json")
            self.assertTrue(container.cleaned)
            self.assertEqual(
                json.loads((Path(output_dir) / "status.json").read_text(encoding="utf-8")),
                {},
            )


if __name__ == "__main__":
    unittest.main()