import sys
import tempfile
import types

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *_args, **_kwargs: None))

from . import validation


class _Result:
    def __init__(self, output="", exit_code=0):
        self.output = output
        self.metadata = types.SimpleNamespace(exit_code=exit_code)


class _Container:
    def __init__(self, exit_code):
        self.exit_code = exit_code
        self.cleaned = False

    def apply_patch(self, *_args, **_kwargs):
        return None

    def send_command(self, command):
        if command == "run-tests":
            return _Result(exit_code=self.exit_code)
        if command == "print-results":
            return _Result("partial test output")
        return _Result()

    def cleanup(self):
        self.cleaned = True


def test_validation_demotes_partial_passes_after_failed_commands():
    containers = []

    class _Runtime:
        @classmethod
        def from_launch_image(cls, *_args, **_kwargs):
            container = _Container(exit_code=1)
            containers.append(container)
            return container

    original_runtime = validation.SetupRuntime
    original_parser = validation.run_parser
    validation.SetupRuntime = _Runtime
    validation.run_parser = lambda *_args: {
        "TestPartial": "pass",
        "TestSkipped": "skip",
    }
    try:
        with tempfile.TemporaryDirectory() as output_dir:
            result = validation.validate_instance(
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
            )
    finally:
        validation.SetupRuntime = original_runtime
        validation.run_parser = original_parser

    assert result["pre_patch_status"] == {
        "TestPartial": "fail",
        "TestSkipped": "skip",
    }
    assert result["post_patch_status"] == {
        "TestPartial": "fail",
        "TestSkipped": "skip",
    }
    assert result["FAIL_TO_PASS"] == []
    assert len(containers) == 4
    assert all(container.cleaned for container in containers)


def test_validation_keeps_statuses_when_command_succeeds():
    assert validation._demote_passes_after_failed_test_command(
        {"TestPassing": "pass", "TestSkipped": "skip"}, 0
    ) == {"TestPassing": "pass", "TestSkipped": "skip"}


if __name__ == "__main__":
    test_validation_demotes_partial_passes_after_failed_commands()
    test_validation_keeps_statuses_when_command_succeeds()
    print("VALIDATION_COMMAND_STATUS_DIRECT_TESTS=PASS")