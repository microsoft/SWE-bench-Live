import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import evaluation


class FakeRuntime:
    def __init__(self):
        self.commands = []
        self.patches = []
        self.cleaned_up = False

    def apply_patch(self, patch, verbose=False):
        self.patches.append((patch, verbose))
        return True

    def send_command(self, command):
        self.commands.append(command)
        return SimpleNamespace(output="")

    def cleanup(self):
        self.cleaned_up = True


def test_linux_clock_is_scoped_to_test_and_log_commands(tmp_path):
    runtime = FakeRuntime()
    with patch.object(evaluation.SetupRuntime, "from_launch_image", return_value=runtime), patch.object(
        evaluation, "run_parser", return_value={}
    ):
        evaluation.evaluate_instance(
            "example__project-1",
            "example-image",
            "rebuild-command",
            "run-tests --flag",
            "collect-log",
            "test-patch",
            "candidate-patch",
            "custom",
            "linux",
            str(tmp_path),
        )

    assert runtime.patches == [("test-patch", False), ("candidate-patch", True)]
    assert runtime.commands[0] == "rebuild-command"
    assert "declare -F swe_bench_live_with_clock" in runtime.commands[1]
    assert "swe_bench_live_with_clock bash -lc" in runtime.commands[1]
    assert "run-tests --flag" in runtime.commands[1]
    assert "else bash -lc" in runtime.commands[1]
    assert "declare -F swe_bench_live_with_clock" in runtime.commands[2]
    assert "swe_bench_live_with_clock bash -lc" in runtime.commands[2]
    assert "collect-log" in runtime.commands[2]
    assert runtime.cleaned_up