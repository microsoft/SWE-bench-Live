import unittest

from evaluation.worktree_contract import (
    WINDOWS_PATCH_ROOT,
    get_windows_evaluation_worktree,
    set_windows_evaluation_worktree,
)


class _Metadata:
    def __init__(self, exit_code: int):
        self.exit_code = exit_code


class _Result:
    def __init__(self, exit_code: int):
        self.metadata = _Metadata(exit_code)


class _Runner:
    def __init__(self, exit_code: int = 0):
        self.exit_code = exit_code
        self.commands: list[str] = []

    def send_command(self, command: str):
        self.commands.append(command)
        return _Result(self.exit_code)


class WindowsWorktreeContractTests(unittest.TestCase):
    def test_uses_secondary_checkout_selected_by_published_command(self):
        worktree = get_windows_evaluation_worktree(
            (
                r"Set-Location C:\go\src\github.com\docker\docker; go build ./cmd/...",
                r"go test -json -v ./... > reports\go-test-results.json",
                r"cat reports\go-test-results.json",
            )
        )
        self.assertEqual(worktree, r"C:\go\src\github.com\docker\docker")

    def test_keeps_patch_root_for_relative_and_testbed_subdirectory_commands(self):
        worktree = get_windows_evaluation_worktree(
            (
                r"Set-Location C:\testbed\frontend; npm run build",
                r"go test ./...",
                r"Get-Content reports\result.json",
            )
        )
        self.assertEqual(worktree, WINDOWS_PATCH_ROOT)

    def test_rejects_ambiguous_multiple_external_checkouts(self):
        with self.assertRaisesRegex(ValueError, "multiple external worktrees"):
            get_windows_evaluation_worktree(
                (
                    r"Set-Location C:\src\one; go build",
                    r"Set-Location C:\src\two; go test",
                )
            )

    def test_sets_resolved_windows_worktree_before_patch_or_command_phases(self):
        runner = _Runner()
        set_windows_evaluation_worktree(
            runner,
            "windows",
            r"C:\go\src\github.com\docker\docker",
        )
        self.assertEqual(
            runner.commands,
            [r"Set-Location -LiteralPath 'C:\go\src\github.com\docker\docker'"],
        )

    def test_is_noop_for_linux_and_fails_on_unusable_windows_root(self):
        linux_runner = _Runner()
        set_windows_evaluation_worktree(linux_runner, "linux", r"C:\unused")
        self.assertEqual(linux_runner.commands, [])

        failed_runner = _Runner(exit_code=1)
        with self.assertRaisesRegex(RuntimeError, "Could not set"):
            set_windows_evaluation_worktree(failed_runner, "windows", r"C:\missing")


if __name__ == "__main__":
    unittest.main()