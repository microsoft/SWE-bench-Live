import tempfile
import types
import unittest

from . import evaluation


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
            return _Result("PASSED example::test\n")
        return _Result()

    def cleanup(self):
        self.cleaned = True


class PatchTestRootContractTests(unittest.TestCase):
    def setUp(self):
        self.original_runtime = evaluation.SetupRuntime

    def tearDown(self):
        evaluation.SetupRuntime = self.original_runtime

    def _install_runtime(self, container):
        class _Runtime:
            @classmethod
            def from_launch_image(cls, *_args, **_kwargs):
                return container

        evaluation.SetupRuntime = _Runtime

    def test_windows_root_change_after_rebuild_stops_before_test(self):
        container = _Container([
            r"C:\testbed",
            r"C:\go\src\github.com\docker\docker",
        ])
        self._install_runtime(container)

        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(
                evaluation.EvaluationInfrastructureError,
                "not patched root",
            ):
                evaluation.evaluate_instance(
                    instance_id="moby__moby-49938",
                    image="example-image",
                    rebuild_cmd=r"Set-Location C:\go\src\github.com\docker\docker",
                    test_cmd="run-tests",
                    print_cmd="print-results",
                    test_patch="test-patch",
                    solution_patch="solution-patch",
                    parser="pytest",
                    platform="windows",
                    output_dir=output_dir,
                )

        self.assertEqual(
            container.commands,
            [
                "git rev-parse --show-toplevel",
                r"Set-Location C:\go\src\github.com\docker\docker",
                "git rev-parse --show-toplevel",
            ],
        )
        self.assertTrue(container.cleaned)

    def test_windows_same_root_allows_test_to_run(self):
        container = _Container([r"C:\testbed", r"c:\TESTBED"])
        self._install_runtime(container)

        with tempfile.TemporaryDirectory() as output_dir:
            result = evaluation.evaluate_instance(
                instance_id="example__same-root",
                image="example-image",
                rebuild_cmd="rebuild",
                test_cmd="run-tests",
                print_cmd="print-results",
                test_patch="test-patch",
                solution_patch="solution-patch",
                parser="pytest",
                platform="windows",
                output_dir=output_dir,
            )

        self.assertEqual(result, {"example::test": "pass"})
        self.assertEqual(container.commands[-2:], ["run-tests", "print-results"])
        self.assertTrue(container.cleaned)


if __name__ == "__main__":
    unittest.main()