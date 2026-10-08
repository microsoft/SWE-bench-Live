import sys
import types
import unittest

sys.modules.setdefault("launch", types.ModuleType("launch"))
sys.modules.setdefault("launch.core", types.ModuleType("launch.core"))
runtime = types.ModuleType("launch.core.runtime")
runtime.SetupRuntime = object
sys.modules.setdefault("launch.core.runtime", runtime)
sys.modules.setdefault("launch.scripts", types.ModuleType("launch.scripts"))
parser = types.ModuleType("launch.scripts.parser")
parser.run_parser = lambda *_args, **_kwargs: {}
sys.modules.setdefault("launch.scripts.parser", parser)
datasets = types.ModuleType("datasets")
datasets.load_dataset = lambda *_args, **_kwargs: []
sys.modules.setdefault("datasets", datasets)
fire = types.ModuleType("fire")
fire.Fire = lambda *_args, **_kwargs: None
sys.modules.setdefault("fire", fire)

from evaluation.command_normalization import normalize_published_shell_command
import evaluation.evaluation as evaluation
import evaluation.validation as validation


class CommandNormalizationTests(unittest.TestCase):
    def test_decodes_html_escaped_powershell_redirection(self):
        self.assertEqual(
            normalize_published_shell_command("mkdir reports -Force 2&gt;$null"),
            "mkdir reports -Force 2>$null",
        )

    def test_leaves_ordinary_shell_text_unchanged(self):
        command = "yarn vitest run --reporter=verbose 2>$null"
        self.assertEqual(normalize_published_shell_command(command), command)

    def test_evaluation_normalizes_all_published_command_lists(self):
        calls = []
        original = evaluation.evaluate_instance
        evaluation.evaluate_instance = lambda *_args: calls.append(_args) or {}
        try:
            instance = {
                "instance_id": "fixture__html_commands",
                "docker_image": "fixture:image",
                "rebuild_cmds": ["setup 2&gt;$null"],
                "test_cmds": ["test 1&gt;out"],
                "print_cmds": ["type out 2&gt;&amp;1"],
                "test_patch": "test patch",
                "pred_patch": "solution patch",
                "PASS_TO_PASS": [],
                "FAIL_TO_PASS": [],
            }
            import tempfile
            with tempfile.TemporaryDirectory() as output_dir:
                evaluation.run_instance(instance, "linux", output_dir, overwrite=True)
        finally:
            evaluation.evaluate_instance = original
        self.assertEqual(calls[0][2:5], ("setup 2>$null", "test 1>out", "type out 2>&1"))

    def test_validation_normalizes_all_published_command_lists(self):
        calls = []
        original = validation.validate_instance
        validation.validate_instance = lambda *_args: calls.append(_args) or {}
        try:
            instance = {
                "instance_id": "fixture__html_commands",
                "docker_image": "fixture:image",
                "rebuild_cmds": ["setup 2&gt;$null"],
                "test_cmds": ["test 1&gt;out"],
                "print_cmds": ["type out 2&gt;&amp;1"],
                "test_patch": "test patch",
                "patch": "solution patch",
                "PASS_TO_PASS": [],
                "FAIL_TO_PASS": [],
            }
            import tempfile
            with tempfile.TemporaryDirectory() as output_dir:
                validation.run_instance(instance, "linux", output_dir, overwrite=True)
        finally:
            validation.validate_instance = original
        self.assertEqual(calls[0][2:5], ("setup 2>$null", "test 1>out", "type out 2>&1"))


if __name__ == "__main__":
    unittest.main()