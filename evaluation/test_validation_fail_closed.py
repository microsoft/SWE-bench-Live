import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The focused helper test does not invoke the CLI, so avoid making its import
# depend on the optional Fire command-line package.
if "fire" not in sys.modules:
    fire_stub = types.ModuleType("fire")
    fire_stub.Fire = lambda *args, **kwargs: None
    sys.modules["fire"] = fire_stub

from evaluation.validation import apply_required_patch


class FakeRuntime:
    def __init__(self, patch_result):
        self.patch_result = patch_result
        self.calls = []
        self.cleaned_up = False

    def apply_patch(self, patch, verbose=False):
        self.calls.append((patch, verbose))
        return self.patch_result

    def cleanup(self):
        self.cleaned_up = True


class ValidationFailClosedTests(unittest.TestCase):
    def test_rejected_patch_cleans_up_and_raises(self):
        runtime = FakeRuntime(False)
        with self.assertRaisesRegex(RuntimeError, "Benchmark test patch did not apply cleanly"):
            apply_required_patch(runtime, "patch", "Benchmark test", "example__project-1")
        self.assertEqual(runtime.calls, [("patch", True)])
        self.assertTrue(runtime.cleaned_up)

    def test_clean_patch_returns_without_cleanup(self):
        runtime = FakeRuntime(True)
        apply_required_patch(runtime, "patch", "Benchmark test", "example__project-1")
        self.assertEqual(runtime.calls, [("patch", True)])
        self.assertFalse(runtime.cleaned_up)


if __name__ == "__main__":
    unittest.main()