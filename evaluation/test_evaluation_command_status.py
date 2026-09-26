import sys
import types

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *_args, **_kwargs: None))

from . import evaluation


class _Result:
    def __init__(self, output: str = "", exit_code: int = 0):
        self.output = output
        self.metadata = types.SimpleNamespace(exit_code=exit_code)


class _Container:
    def __init__(self, test_exit_code: int, log: str):
        self.test_exit_code = test_exit_code
        self.log = log
        self.cleaned = False

    def apply_patch(self, *_args, **_kwargs):
        return True

    def send_command(self, command: str):
        if command == "run-tests":
            return _Result(exit_code=self.test_exit_code)
        if command == "print-results":
            return _Result(output=self.log)
        return _Result()

    def cleanup(self):
        self.cleaned = True


def test_evaluator_keeps_real_test_status_when_test_command_succeeds(monkeypatch, tmp_path):
    tmp_path.mkdir()
    container = _Container(0, "log")
    monkeypatch.setattr(
        evaluation.SetupRuntime,
        "from_launch_image",
        lambda *_args, **_kwargs: container,
    )
    monkeypatch.setattr(evaluation, "run_parser", lambda *_args: {"TestTarget": "pass"})

    status = evaluation.evaluate_instance(
        "example__success", "image", "", "run-tests", "print-results", "", "", "parser", "windows", str(tmp_path)
    )

    assert status == {"TestTarget": "pass"}
    assert container.cleaned


def test_evaluator_marks_terminal_test_status_failed_when_test_command_fails(monkeypatch, tmp_path):
    tmp_path.mkdir()
    container = _Container(1, "log")
    monkeypatch.setattr(
        evaluation.SetupRuntime,
        "from_launch_image",
        lambda *_args, **_kwargs: container,
    )
    monkeypatch.setattr(evaluation, "run_parser", lambda *_args: {"TestTarget": "pass", "Skipped": "skip"})

    status = evaluation.evaluate_instance(
        "example__failed-command", "image", "", "run-tests", "print-results", "", "", "parser", "windows", str(tmp_path)
    )

    assert status == {"TestTarget": "fail", "Skipped": "skip"}
    assert container.cleaned