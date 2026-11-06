import tempfile
import types
import unittest

from . import evaluation


class _Result:
    def __init__(self, output: str = "", exit_code: int = 0):
        self.output = output
        self.metadata = types.SimpleNamespace(exit_code=exit_code)


class _Container:
    def __init__(self, exit_code: int):
        self.exit_code = exit_code
        self.cleaned = False
        self.commands: list[str] = []

    def apply_patch(self, *_args, **_kwargs):
        return True

    def send_command(self, command: str):
        self.commands.append(command)
        if command == "run-tests":
            return _Result(exit_code=self.exit_code)
        if command == "print-results":
            return _Result("partial test output")
        return _Result()

    def cleanup(self):
        self.cleaned = True


class EvaluationCommandStatusTests(unittest.TestCase):
    def setUp(self):
        self.containers: list[_Container] = []
        self.original_runtime = evaluation.SetupRuntime
        self.original_parser = evaluation.run_parser

        containers = self.containers

        class _Runtime:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                container = _Container(exit_code=1)
                containers.append(container)
                return container

        evaluation.SetupRuntime = _Runtime
        evaluation.run_parser = lambda *_args: {
            "TestPartial": "pass",
            "TestSkipped": "skip",
        }

    def tearDown(self):
        evaluation.SetupRuntime = self.original_runtime
        evaluation.run_parser = self.original_parser

    def test_evaluation_keeps_terminal_statuses_and_fails_missing_expected_tests(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = evaluation.evaluate_instance(
                instance_id="example__partial-command",
                image="example-image",
                rebuild_cmd="rebuild",
                test_cmd="run-tests",
                print_cmd="print-results",
                test_patch="",
                solution_patch="",
                parser="example",
                platform="windows",
                output_dir=output_dir,
                expected_tests=["TestPartial", "TestSkipped", "TestNotReported"],
            )

        self.assertEqual(
            result,
            {
                "TestPartial": "pass",
                "TestSkipped": "skip",
                "TestNotReported": "fail",
            },
        )
        self.assertEqual(len(self.containers), 1)
        self.assertTrue(self.containers[0].cleaned)

    def test_nonzero_exit_preserves_parsed_pass_fail_and_skip_statuses(self):
        status = {
            "TestPassed": "pass",
            "TestFailed": "fail",
            "TestSkipped": "skip",
        }
        self.assertEqual(
            evaluation._mark_missing_expected_tests_after_failed_test_command(
                status,
                1,
                ["TestPassed", "TestFailed", "TestSkipped"],
            ),
            status,
        )

    def test_successful_command_does_not_invent_missing_failures(self):
        status = {"TestPassing": "pass", "TestSkipped": "skip"}
        self.assertEqual(
            evaluation._mark_missing_expected_tests_after_failed_test_command(
                status,
                0,
                ["TestPassing", "TestSkipped", "TestNotReported"],
            ),
            status,
        )


if __name__ == "__main__":
    unittest.main()