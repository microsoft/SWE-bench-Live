import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import evaluation


class FakeRuntime:
    def __init__(self, mount_path):
        self.commands = []
        self.patches = []
        self.cleaned_up = False
        self.mnt_host = str(mount_path)
        self.mnt_container = "/mnt_tmp"
        self.test_log = "test execution output\n"
        self.collected_log = "collected test output\n"

    def apply_patch(self, patch, verbose=False):
        self.patches.append((patch, verbose))
        return True

    def send_command(self, command):
        self.commands.append(command)
        if command == "git rev-parse --show-toplevel":
            return SimpleNamespace(output="/testbed\n", metadata=SimpleNamespace(exit_code=0))
        match = re.search(r"> /mnt_tmp/(swe_bench_live_eval_[0-9a-f]+\.log) 2>&1", command)
        if match:
            content = self.collected_log if "collect-log" in command else self.test_log
            (Path(self.mnt_host) / match.group(1)).write_text(content, encoding="utf-8")
        return SimpleNamespace(output="", metadata=SimpleNamespace(exit_code=0))

    def cleanup(self):
        self.cleaned_up = True


def test_linux_clock_is_scoped_to_test_and_log_commands(tmp_path):
    runtime = FakeRuntime(tmp_path)
    with patch.object(evaluation.SetupRuntime, "from_launch_image", return_value=runtime), patch.object(
        evaluation, "run_parser", return_value={}
    ) as parser:
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

    assert runtime.patches == [("test-patch", True), ("candidate-patch", True)]
    assert runtime.commands[0] == "git rev-parse --show-toplevel"
    assert runtime.commands[1] == "rebuild-command"
    assert runtime.commands[2] == "git rev-parse --show-toplevel"
    assert "declare -F swe_bench_live_with_clock" in runtime.commands[3]
    assert "swe_bench_live_with_clock bash -c" in runtime.commands[3]
    assert "bash -lc" not in runtime.commands[3]
    assert "run-tests --flag" in runtime.commands[3]
    assert "else bash -c" in runtime.commands[3]
    assert "> /mnt_tmp/swe_bench_live_eval_" in runtime.commands[3]
    assert "declare -F swe_bench_live_with_clock" in runtime.commands[4]
    assert "swe_bench_live_with_clock bash -c" in runtime.commands[4]
    assert "bash -lc" not in runtime.commands[4]
    assert "collect-log" in runtime.commands[4]
    assert "> /mnt_tmp/swe_bench_live_eval_" in runtime.commands[4]
    parser.assert_called_once_with(
        "custom", "test execution output\n\ncollected test output\n"
    )
    assert list(tmp_path.glob("swe_bench_live_eval_*.log")) == []
    assert runtime.cleaned_up