"""Direct regression tests for evaluator command/capture contracts.

Run with: python evaluation/test_evaluation_command_contract.py
"""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import evaluation as evaluator


class _Metadata:
    def __init__(self, exit_code: int):
        self.exit_code = exit_code


class _Result:
    def __init__(self, output: str = "", exit_code: int = 0):
        self.output = output
        self.metadata = _Metadata(exit_code)


class _Container:
    def __init__(self, *, test_exit_code: int, print_output: str):
        self.test_exit_code = test_exit_code
        self.print_output = print_output
        self.calls: list[str] = []
        self.cleaned = False

    def apply_patch(self, patch: str, verbose: bool = False) -> bool:
        return True

    def send_command(self, command: str) -> _Result:
        self.calls.append(command)
        if command.startswith("Test-Path -LiteralPath"):
            return _Result("False\n")
        if command == "go test ./...":
            return _Result("partial test output", self.test_exit_code)
        if command == "Get-Content -Raw reports\\go-test-results.json":
            return _Result(self.print_output)
        return _Result()

    def cleanup(self) -> None:
        self.cleaned = True


class _RuntimeFactory:
    def __init__(self, container: _Container):
        self.container = container

    def from_launch_image(self, *args, **kwargs) -> _Container:
        return self.container


class EvaluationCommandContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_runtime = evaluator.SetupRuntime
        self.original_parser = evaluator.run_parser

    def tearDown(self) -> None:
        evaluator.SetupRuntime = self.original_runtime
        evaluator.run_parser = self.original_parser

    def _run(self, container: _Container, output_dir: Path):
        evaluator.SetupRuntime = _RuntimeFactory(container)
        evaluator.run_parser = lambda parser, log: {"target": "pass"}
        return evaluator.evaluate_instance(
            instance_id="fixture__windows-1",
            image="fixture:image",
            rebuild_cmd=r"Set-Location C:\go\src\github.com\docker\docker; go build ./cmd/...",
            test_cmd="go test ./...",
            print_cmd=r"Get-Content -Raw reports\go-test-results.json",
            test_patch="diff --git a/a b/a\n",
            solution_patch="diff --git a/b b/b\n",
            parser="fixture parser",
            platform="windows",
            output_dir=str(output_dir),
        )

    def test_empty_capture_is_an_infrastructure_error(self) -> None:
        container = _Container(test_exit_code=0, print_output="")
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            with self.assertRaises(evaluator.EvaluationInfrastructureError):
                self._run(container, output_dir)
            evidence = json.loads((output_dir / "evaluator_error.json").read_text())
            self.assertEqual(evidence["error"]["phase"], "capture")
            self.assertTrue(evidence["patches"]["test_patch"]["applied"])
            self.assertTrue(evidence["patches"]["solution_patch"]["applied"])
            self.assertEqual(evidence["commands"]["test"]["exit_code"], 0)
            self.assertEqual(evidence["commands"]["print"]["output_chars"], 0)
        self.assertTrue(container.cleaned)

    def test_failed_test_command_cannot_contribute_passes(self) -> None:
        container = _Container(test_exit_code=1, print_output="captured test output\n")
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            status = self._run(container, output_dir)
            self.assertEqual(status, {"target": "fail"})
            command_status = json.loads((output_dir / "command_status.json").read_text())
            self.assertTrue(command_status["test_command_failed"])
            self.assertEqual(command_status["commands"]["test"]["exit_code"], 1)
            self.assertEqual(
                command_status["command_rewrites"]["rebuild"],
                [{"from": r"C:\go\src\github.com\docker\docker", "to": r"C:\testbed"}],
            )
            self.assertTrue(any(r"Set-Location C:\testbed" in call for call in container.calls))
        self.assertTrue(container.cleaned)


if __name__ == "__main__":
    unittest.main(verbosity=2)