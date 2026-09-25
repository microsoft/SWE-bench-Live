import json
import sys
import types

# The unit test exercises validation's parser integration without requiring its
# optional CLI dependency in a minimal test environment.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *_args, **_kwargs: None))

from . import validation


class _Result:
    def __init__(self, output: str = ""):
        self.output = output


class _Container:
    def __init__(self, log: str):
        self.log = log
        self.cleaned = False

    def apply_patch(self, *_args, **_kwargs):
        return None

    def send_command(self, command: str):
        if command == "print-results":
            return _Result(self.log)
        return _Result()

    def cleanup(self):
        self.cleaned = True


def test_validation_uses_windows_go_json_for_pre_and_all_post_runs(monkeypatch, tmp_path):
    log = json.dumps(
        {
            "Time": "2026-01-01T00:00:00Z",
            "Action": "pass",
            "Package": "example/module",
            "Test": "TestTarget",
        }
    )
    containers = []

    class _Runtime:
        @classmethod
        def from_launch_image(cls, *_args, **_kwargs):
            container = _Container(log)
            containers.append(container)
            return container

    monkeypatch.setattr(validation, "SetupRuntime", _Runtime)
    monkeypatch.setattr(validation, "run_parser", lambda *_args: {"Ghost": "fail"})

    result = validation.validate_instance(
        instance_id="example__windows-validation",
        image="example-image",
        rebuild_cmd="rebuild",
        test_cmd="run-tests",
        print_cmd="print-results",
        test_patch="",
        solution_patch="",
        parser="go-json",
        platform="windows",
        output_dir=str(tmp_path),
    )

    assert result["pre_patch_status"] == {"TestTarget": "pass"}
    assert result["post_patch_status"] == {"TestTarget": "pass"}
    assert len(containers) == 4
    assert all(container.cleaned for container in containers)