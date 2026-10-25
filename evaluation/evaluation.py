import sys, os
import hashlib
import subprocess
import tempfile
sys.path.insert(0, os.path.join(os.getcwd(), "launch"))
from launch.core.runtime import SetupRuntime
from launch.scripts.parser import run_parser
import json
import argparse
import traceback
import re
import xml.etree.ElementTree as ET
import ntpath
import shlex
import uuid
from typing import Literal, TypedDict
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

# SetupRuntime.command_timeout is expressed in minutes (RepoLaunch converts it to seconds).
# Keep the evaluator limit at 150 minutes; passing 150 * 60 here would become 9,000 minutes.
COMMAND_TIMEOUT_MINUTES = 150


class EvaluationContractError(RuntimeError):
    """The evaluator could not observe a scoreable test contract."""


class EvaluationInfrastructureError(RuntimeError):
    """The evaluator could not establish a trustworthy patched worktree."""


def _windows_path_equal(left: str, right: str) -> bool:
    return ntpath.normcase(ntpath.normpath(left)) == ntpath.normcase(ntpath.normpath(right))


def _raise_root_identity_error(container, message: str) -> None:
    try:
        container.cleanup()
    except Exception:
        pass
    raise EvaluationInfrastructureError(message)


def _effective_repository_root(container) -> str:
    result = container.send_command("git rev-parse --show-toplevel")
    if int(getattr(getattr(result, "metadata", None), "exit_code", 1)) != 0 or not (result.output or "").strip():
        _raise_root_identity_error(container, "cannot determine repository root after patch application")
    return (result.output or "").strip().splitlines()[-1].strip()


def _assert_grading_root(container, expected_root: str, platform: Literal["windows", "linux"], instance_id: str) -> None:
    """Fail closed if rebuild commands move grading to another Git worktree."""
    actual_root = _effective_repository_root(container)
    same_root = _windows_path_equal(actual_root, expected_root) if platform == "windows" else actual_root == expected_root
    if not same_root:
        _raise_root_identity_error(
            container,
            f"{instance_id} grading commands would run from repository root {actual_root!r}, not patched root {expected_root!r}",
        )


def validate_solution_patch_format(solution_patch: str) -> dict | None:
    """Reject malformed local candidate artifacts before starting a task container.

    This is strict syntax validation of the local model-patch artifact only. It
    does not inspect a benchmark checkout, hidden test material, or score a task.
    """
    raw = str(solution_patch or "").encode("utf-8")
    evidence = {
        "validator": "git apply --stat --no-summary",
        "candidate_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_bytes": len(raw),
        "terminal_newline": raw.endswith(b"\n"),
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".patch", delete=False) as handle:
            handle.write(raw)
            temporary_path = handle.name
        completed = subprocess.run(
            ["git", "apply", "--stat", "--no-summary", temporary_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {**evidence, "reasons": ["candidate_patch_preflight_unavailable"], "error": str(exc)}
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
    reasons = []
    if not raw.endswith(b"\n"):
        reasons.append("missing_terminal_newline")
    if completed.returncode != 0:
        reasons.append("git_apply_stat_rejected_patch")
    if not reasons:
        return None
    return {
        **evidence,
        "exit_code": completed.returncode,
        "reasons": reasons,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def normalize_month(month: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError(f"Invalid month format: {month}. Expected YYYY-MM.")
    return month


def get_instance_month(instance: dict) -> str | None:
    created_at = instance.get("created_at")
    if created_at is None:
        return None
    created_at_str = str(created_at).strip()
    if not created_at_str:
        return None
    match = re.match(r"^(\d{4}-\d{2})", created_at_str)
    if match is None:
        return None
    return match.group(1)


def parse_log_pytest(log: str, test_spec: "TestSpec") -> dict[str, str]:
    """
    Copied from SWE-bench/swebench/harness/constants/__init__.py
    """
    class TestStatus(Enum):
        FAILED = "FAILED"
        PASSED = "PASSED"
        SKIPPED = "SKIPPED"
        ERROR = "ERROR"
        XFAIL = "XFAIL"
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def default_pytest_parser(log: str) -> dict[str, str]:
    mapping = parse_log_pytest(log, None)
    for test in mapping.keys():
        # Pytest XFAIL means an expected failure remained expected. SWE-bench
        # grading treats XFAIL identically to PASSED for FAIL_TO_PASS and as
        # maintained for PASS_TO_PASS, so do not collapse it into fail here.
        if mapping[test].upper() == 'XFAIL':
            mapping[test] = 'pass'
        elif 'pass' in mapping[test].lower():
            mapping[test] = 'pass'
        elif 'skip' in mapping[test].lower():
            mapping[test] = 'skip'
        else:
            mapping[test] = 'fail'
    return mapping


def _go_terminal_passes(log: str) -> set[str]:
    """Collect Go tests with an explicit terminal PASS event in the log."""
    terminal_passes: set[str] = set()
    for line in log.splitlines():
        output = line
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = None
        if isinstance(event, dict):
            test = event.get("Test")
            if event.get("Action") == "pass" and isinstance(test, str) and test.strip():
                terminal_passes.add(test.strip())
            output = str(event.get("Output", ""))
        for match in re.finditer(r"--- PASS: ([^\r\n]+?)(?: \([^)]*\))?(?:\r?\n|$)", output):
            terminal_passes.add(match.group(1).strip())
    return terminal_passes


def _matches_go_terminal_pass(test_name: str, terminal_pass: str) -> bool:
    """Match evaluator package-qualified Go names to go test's short names."""
    return (
        test_name == terminal_pass
        or test_name.endswith("/" + terminal_pass)
        or test_name.endswith("." + terminal_pass)
        or test_name.endswith("::" + terminal_pass)
    )


def _structured_terminal_statuses(log: str) -> dict[str, Literal['pass', 'fail', 'skip']]:
    """Read complete JUnit-style terminal testcase statuses from concatenated XML."""
    statuses: dict[str, Literal['pass', 'fail', 'skip']] = {}

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].lower()

    for match in re.finditer(r"(<\?xml[^>]*\?>.*?)(?=(<\?xml|\Z))", log, re.DOTALL):
        try:
            root = ET.fromstring(match.group(1).strip())
        except ET.ParseError:
            continue
        root_tag = local(root.tag)
        suites = [root] if root_tag == "testsuite" else [child for child in root if local(child.tag) == "testsuite"] if root_tag == "testsuites" else []
        for suite in suites:
            suite_name = suite.attrib.get("name", "")
            for testcase in suite:
                if local(testcase.tag) != "testcase":
                    continue
                name = testcase.attrib.get("name", "")
                if not name:
                    continue
                classname = testcase.attrib.get("classname", "")
                identity = f"{classname}::{name}" if classname else (f"{suite_name}::{name}" if suite_name else name)
                status: Literal['pass', 'fail', 'skip'] = "pass"
                for child in testcase:
                    child_tag = local(child.tag)
                    if child_tag in {"failure", "error"}:
                        status = "fail"
                        break
                    if child_tag == "skipped":
                        status = "skip"
                        break
                statuses[identity] = status
    return statuses


def _demote_unproven_passes_after_failed_command(
    statuses: dict[str, Literal['pass', 'fail', 'skip']],
    log: str,
) -> dict[str, Literal['pass', 'fail', 'skip']]:
    """Reject partial passes but retain explicit terminal structured-test passes."""
    terminal_passes = _go_terminal_passes(log)
    structured_statuses = _structured_terminal_statuses(log)
    return {
        test: (
            status
            if status != "pass"
            or any(_matches_go_terminal_pass(test, passed) for passed in terminal_passes)
            or structured_statuses.get(test) == "pass"
            else "fail"
        )
        for test, status in statuses.items()
    }


def get_default_image_name(instance_id: str, platform: Literal["windows", "linux"]) -> str:
    if platform == "linux":
        med = "x86_64"
    else:
        med = "win"
    name = instance_id.replace("__", "_1776_").lower()
    image = f"starryzhang/sweb.eval.{med}.{name}"
    return image

def apply_solution_patch_best_effort(solution_patch: str, 
                                    container: SetupRuntime,
                                    platform: Literal["windows", "linux"]) -> None:
    '''
    For some 1% corner cases where /testbed/repo_name is the actual project
    '''
    if platform == "linux":
        container.send_command("cd /testbed")
        container.send_command("""[ -d .git ] || { g=$(find . -maxdepth 2 -mindepth 2 -type d -name .git -print -quit); [ -n "$g" ] && cd "${g%/.git}"; } ;""")
        container.apply_patch(solution_patch, verbose=True)
        container.send_command("cd /testbed")
    else:
        container.send_command(r"cd C:\testbed")
        container.send_command(r"""if (-not (Test-Path .git)) { $g = Get-ChildItem -Directory -Recurse -Depth 2 -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq '.git' } | Select-Object -First 1; if ($g) { Set-Location $g.Parent.FullName } };""")
        container.apply_patch(solution_patch, verbose=True)
        container.send_command(r"cd C:\testbed")
    return

def evaluate_instance(  
                    instance_id: str,
                    image: str, 
                    rebuild_cmd: str, 
                    test_cmd: str, 
                    print_cmd: str,
                    test_patch: str, 
                    solution_patch: str,
                    parser: str,
                    platform: Literal["windows", "linux"],
                    output_dir: str,
                    ) -> dict[str, Literal['pass', 'fail', 'skip']]:
    container: SetupRuntime = SetupRuntime.from_launch_image(
        image, instance_id, platform, command_timeout=COMMAND_TIMEOUT_MINUTES
    )
    if not container.apply_patch(test_patch, verbose=True):
        container.cleanup()
        raise RuntimeError(f"Benchmark test patch did not apply cleanly for {instance_id}")
    if not container.apply_patch(solution_patch, verbose=True):
        container.cleanup()
        raise RuntimeError(f"Candidate patch did not apply cleanly for {instance_id}")
    patched_root = _effective_repository_root(container)
    # Rebuild commands may intentionally change directory. The test and log
    # commands must still execute in the Git worktree that received both patches.
    if rebuild_cmd.strip():
        container.send_command(rebuild_cmd)
    _assert_grading_root(container, patched_root, platform, instance_id)
    # RepoLaunch exposes a task-specific faketime environment only when an
    # evaluator explicitly requested a deterministic snapshot clock. Keep patch
    # application and rebuilds on real filesystem time, then scope faketime to
    # the test and log-collection commands.
    if not print_cmd.strip():
        # for backward compatibility with SWE-bench-Live/SWE-bench-Live (Python)
        container.send_command(f"cat > run_test.sh <<'CC_PROMPT'\n{test_cmd}\nCC_PROMPT\n")
        test_cmd = "bash run_test.sh > testlog.out 2>&1"
        print_cmd = "cat testlog.out"

    def run_test_command(command: str):
        if platform != "linux":
            return container.send_command(command)
        # RepoLaunch defines this shell function only for an explicitly requested
        # evaluation clock. Do not replace the function with a fallback: that
        # would disable faketime in the very runs that requested it.
        shell_command = shlex.quote(command)
        return container.send_command(
            "if declare -F swe_bench_live_with_clock >/dev/null; then "
            f"swe_bench_live_with_clock bash -c {shell_command}; "
            f"else bash -c {shell_command}; fi"
        )

    def run_test_command_with_mounted_output(command: str):
        """Run a Linux command without streaming a large output through the PTY."""
        if platform != "linux":
            return run_test_command(command), None
        log_name = f"swe_bench_live_eval_{uuid.uuid4().hex}.log"
        host_log_path = os.path.join(container.mnt_host, log_name)
        container_log_path = f"{container.mnt_container.rstrip('/')}/{log_name}"
        redirected_command = f"{{ {command}; }} > {shlex.quote(container_log_path)} 2>&1"
        return run_test_command(redirected_command), host_log_path

    def collect_test_log(command: str) -> tuple[str, int]:
        """Collect Linux command output and preserve the command's exit status."""
        result, host_log_path = run_test_command_with_mounted_output(command)
        exit_code = int(getattr(getattr(result, "metadata", None), "exit_code", 0))
        if host_log_path is None:
            return result.output, exit_code
        try:
            with open(host_log_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(), exit_code
        except FileNotFoundError:
            # Preserve the command transcript when a task-specific command fails
            # before it creates the redirected log file.
            return result.output, exit_code
        finally:
            try:
                os.remove(host_log_path)
            except FileNotFoundError:
                pass

    # The test command itself is the primary evidence. Earlier PTY-avoidance
    # code discarded that mounted output and relied only on print_cmd; many Go
    # tasks therefore produced an empty parser input. Keep both sources off the
    # PTY and append print_cmd output only when it contributes additional text.
    post_patch_log, test_exit_code = collect_test_log(test_cmd)
    if print_cmd.strip():
        printed_log, _ = collect_test_log(print_cmd)
        if printed_log:
            post_patch_log += printed_log if not post_patch_log else "\n" + printed_log
    with open(os.path.join(output_dir, "post_patch_log.txt"), "w", encoding="utf-8") as f:
        f.write(post_patch_log)
    if parser.lower().strip() == "pytest":
        # for backward compatibility with SWE-bench-Live/SWE-bench-Live (Python)
        post_patch_status: dict[str, Literal['pass', 'fail', 'skip']] = default_pytest_parser(post_patch_log)
    else:
        post_patch_status: dict[str, Literal['pass', 'fail', 'skip']] = run_parser(parser, post_patch_log)
    if test_exit_code != 0:
        # A failed aggregate command may contain partial parser output. Those
        # partial passes are not sufficient evidence for FAIL_TO_PASS or
        # PASS_TO_PASS success; preserve explicit skips for reporting.
        post_patch_status = _demote_unproven_passes_after_failed_command(
            post_patch_status,
            post_patch_log,
        )
    container.cleanup()
    with open(os.path.join(output_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(post_patch_status, f, indent = True)
    return post_patch_status

def normalize_test_identifier(test_name: str) -> str:
    """Remove Jest's runtime-only duration suffix from a parsed test name."""
    return re.sub(r" \(\d+ ms\)$", "", test_name)


def evaluate_baseline_instance(
                    instance_id: str,
                    image: str,
                    rebuild_cmd: str,
                    test_cmd: str,
                    print_cmd: str,
                    test_patch: str,
                    parser: str,
                    platform: Literal["windows", "linux"],
                    output_dir: str,
                    ) -> dict[str, Literal['pass', 'fail', 'skip']]:
    """Replay the published test contract without the candidate patch.

    This conditional control runs only after every FAIL_TO_PASS target passed
    and candidate PASS_TO_PASS failures remain.  It uses the same fail-closed
    patch, patched-worktree, clock, and mounted-log rules as candidate grading.
    """
    container: SetupRuntime = SetupRuntime.from_launch_image(
        image, instance_id, platform, command_timeout=COMMAND_TIMEOUT_MINUTES
    )
    try:
        if not container.apply_patch(test_patch, verbose=True):
            raise RuntimeError(f"Benchmark test patch did not apply cleanly for baseline control {instance_id}")
        patched_root = _effective_repository_root(container)
        if rebuild_cmd.strip():
            container.send_command(rebuild_cmd)
        _assert_grading_root(container, patched_root, platform, instance_id)
        if not print_cmd.strip():
            container.send_command(f"cat > run_test.sh <<'CC_PROMPT'\n{test_cmd}\nCC_PROMPT\n")
            test_cmd = "bash run_test.sh > testlog.out 2>&1"
            print_cmd = "cat testlog.out"

        def run_clock_scoped(command: str):
            if platform != "linux":
                return container.send_command(command)
            shell_command = shlex.quote(command)
            return container.send_command(
                "if declare -F swe_bench_live_with_clock >/dev/null; then "
                f"swe_bench_live_with_clock bash -c {shell_command}; "
                f"else bash -c {shell_command}; fi"
            )

        def collect_log(command: str) -> tuple[str, int]:
            if platform != "linux":
                result = run_clock_scoped(command)
                return result.output, int(getattr(getattr(result, "metadata", None), "exit_code", 0))
            log_name = f"swe_bench_live_baseline_{uuid.uuid4().hex}.log"
            host_log_path = os.path.join(container.mnt_host, log_name)
            container_log_path = f"{container.mnt_container.rstrip('/')}/{log_name}"
            result = run_clock_scoped(f"{{ {command}; }} > {shlex.quote(container_log_path)} 2>&1")
            exit_code = int(getattr(getattr(result, "metadata", None), "exit_code", 0))
            try:
                with open(host_log_path, "r", encoding="utf-8", errors="replace") as handle:
                    return handle.read(), exit_code
            except FileNotFoundError:
                return result.output, exit_code
            finally:
                try:
                    os.remove(host_log_path)
                except FileNotFoundError:
                    pass

        baseline_log, test_exit_code = collect_log(test_cmd)
        if print_cmd.strip():
            printed_log, _ = collect_log(print_cmd)
            if printed_log:
                baseline_log += printed_log if not baseline_log else "\n" + printed_log
        with open(os.path.join(output_dir, "baseline_post_patch_log.txt"), "w", encoding="utf-8") as handle:
            handle.write(baseline_log)
        if parser.lower().strip() == "pytest":
            baseline_status: dict[str, Literal['pass', 'fail', 'skip']] = default_pytest_parser(baseline_log)
        else:
            baseline_status = run_parser(parser, baseline_log)
        if test_exit_code != 0:
            baseline_status = _demote_unproven_passes_after_failed_command(
                baseline_status,
                baseline_log,
            )
        with open(os.path.join(output_dir, "baseline_status.json"), "w", encoding="utf-8") as handle:
            json.dump(baseline_status, handle, indent=True)
        return baseline_status
    finally:
        container.cleanup()


def run_instance(
                    instance: dict,
                    platform: Literal["windows","linux"],
                    output_dir: str,
                    overwrite: bool,
                    collect_only: bool = False,
                ):
    instance_output_dir = os.path.join(output_dir, instance["instance_id"])
    report_dir = os.path.join(instance_output_dir, "report.json")
    if (collect_only or not overwrite) and os.path.exists(report_dir):
        try:
            with open(report_dir, encoding="utf-8") as f:
                report = json.load(f)
                if report.get("resolved", None) is not None:
                    suc = "Success!" if report["resolved"] else "Failed..."
                    print(suc, "(Skipped)", instance["instance_id"], flush=True)
                    return report
        except:
            pass
    if collect_only:
        print("Incomplete...", instance["instance_id"], flush=True)
        return {"instance_id": instance["instance_id"], "resolved": None}
    os.makedirs(instance_output_dir, exist_ok=True)
    patch_preflight = validate_solution_patch_format(instance["pred_patch"])
    if patch_preflight is not None:
        report = {
            "instance_id": instance["instance_id"],
            "resolved": None,
            "grading_status": "not_evaluated",
            "failure_class": "local_candidate_patch_preflight_failed",
            "candidate_patch_preflight": patch_preflight,
        }
        with open(report_dir, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=True)
        print("Invalid candidate artifact...", instance["instance_id"], flush=True)
        return report
    res: dict[str, Literal['pass', 'fail', 'skip']] = evaluate_instance(
            instance["instance_id"],
            instance.get("docker_image", get_default_image_name(instance["instance_id"], platform)),
            " ; ".join(instance.get("rebuild_cmds", [])),
            " ; ".join(instance.get("test_cmds", [])),
            " ; ".join(instance.get("print_cmds", [])),
            instance["test_patch"],
            instance["pred_patch"],
            instance.get("log_parser", instance.get("parser", "")),
            platform,
            instance_output_dir
    )
    suc = {normalize_test_identifier(test) for test, status in res.items() if 'pass' in status.lower()}
    fail = {normalize_test_identifier(test) for test, status in res.items() if 'fail' in status.lower()}
    pass_to_pass = {normalize_test_identifier(test) for test in instance["PASS_TO_PASS"]}
    fail_to_pass = {normalize_test_identifier(test) for test in instance["FAIL_TO_PASS"]}
    expected_tests = pass_to_pass | fail_to_pass
    observed_tests = {normalize_test_identifier(test) for test in res}
    matched_tests = observed_tests & expected_tests
    contract = {
        "valid": bool(expected_tests) and bool(observed_tests) and bool(matched_tests),
        "expected_tests": len(expected_tests),
        "observed_tests": len(observed_tests),
        "matched_expected_tests": len(matched_tests),
    }
    report = {
        "instance_id": instance["instance_id"],
        "resolved": False,
        "evaluation_contract": contract,
        "PASS_TO_PASS": {
            "success": list(suc & pass_to_pass),
            "failure": list(fail & pass_to_pass),
        },
        "FAIL_TO_PASS": {
            "success": list(suc & fail_to_pass),
            "failure": list(fail & fail_to_pass),
        },
    }
    if not contract["valid"]:
        with open(report_dir, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=True)
        raise EvaluationContractError(
            f"Unscoreable evaluator output for {instance['instance_id']}: "
            f"expected={contract['expected_tests']}, "
            f"observed={contract['observed_tests']}, "
            f"matched={contract['matched_expected_tests']} (matched none)"
        )

    f2p = fail_to_pass.issubset(set(report["FAIL_TO_PASS"]["success"])) \
        or (len(report["FAIL_TO_PASS"]["success"]) == len(fail_to_pass))
    if (len(report["PASS_TO_PASS"]["failure"]) == 0) \
        and (len(report["FAIL_TO_PASS"]["failure"]) == 0) \
        and f2p:
        report["resolved"] = True
        print("Success!", instance["instance_id"], flush=True)
    elif f2p and report["PASS_TO_PASS"]["failure"]:
        baseline_status = evaluate_baseline_instance(
            instance["instance_id"],
            instance.get("docker_image", get_default_image_name(instance["instance_id"], platform)),
            " ; ".join(instance.get("rebuild_cmds", [])),
            " ; ".join(instance.get("test_cmds", [])),
            " ; ".join(instance.get("print_cmds", [])),
            instance["test_patch"],
            instance.get("log_parser", instance.get("parser", "")),
            platform,
            instance_output_dir,
        )
        baseline_failures = {
            test for test, status in baseline_status.items() if "fail" in status.lower()
        }
        candidate_failures = set(report["PASS_TO_PASS"]["failure"])
        unchanged_baseline_failures = sorted(candidate_failures & baseline_failures)
        candidate_only_failures = sorted(candidate_failures - baseline_failures)
        report["baseline_control"] = {
            "performed": True,
            "matching_baseline_failures": unchanged_baseline_failures,
            "candidate_only_failures": candidate_only_failures,
        }
        if not candidate_only_failures:
            report["resolved"] = True
            print("Success!", instance["instance_id"], "(baseline PASS_TO_PASS failures retained)", flush=True)
        else:
            print("Failed...", instance["instance_id"], flush=True)
    else:
        print("Failed...", instance["instance_id"], flush=True)

    with open(report_dir, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=True)
    return report

def run_instances(instances: list[dict[str, str]],
                    platform: Literal["windows", "linux"],
                    workers: int,
                    output_dir: str,
                    overwrite: bool,
                    collect_only: bool = False):
    todos = []
    empty_instance_ids = []
    for i in instances:
        if i["pred_patch"].strip():
            todos.append(i)
        else:
            empty_instance_ids.append(i["instance_id"])
            print("Empty patch...", i["instance_id"], flush=True)
    results = {
        "submitted": len(instances),
        "submitted_ids": [i["instance_id"] for i in instances],
        "empty_patch": len(empty_instance_ids),
        "empty_patch_ids": empty_instance_ids,
        "success_ids": [],
        "failure_ids": [],
        "error_ids": [],
        "incomplete_ids": [],
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit tasks to the executor
        future_to_instance = {
            executor.submit(run_instance, instance, platform, output_dir, overwrite, collect_only): instance
            for instance in todos
        }

        # Collect results as they complete
        for future in as_completed(future_to_instance):
            instance = future_to_instance[future]
            try:
                result = future.result()
                if result["resolved"] is None:
                    results["incomplete_ids"].append(instance["instance_id"])
                elif result["resolved"]:
                    results["success_ids"].append(instance["instance_id"])
                else:
                    results["failure_ids"].append(instance["instance_id"])
            except Exception as e:
                print(f"Error processing instance {instance['instance_id']}: {e} \n{traceback.format_exc()}")
                results["error_ids"].append(instance["instance_id"])
    results["success"] = len(results["success_ids"])
    results["failure"] = len(results["failure_ids"])
    results["error"] = len(results["error_ids"])
    results["incomplete"] = len(results["incomplete_ids"])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent = True)
    return results

def main(
            dataset: str,
            patch_dir: str, 
            platform: Literal["windows", "linux"], 
            workers: int, 
            output_dir: str, 
            overwrite: int,
            split: str|None = None,
            instance_ids: list[str] | None = None,
            start_month: str | None = None,
            end_month: str | None = None,
            collect_only: bool = False,
        ):
    if patch_dir.strip() != "gold":
        with open(patch_dir, encoding="utf-8") as f:
            preds = json.load(f)
        print(f"Loaded {len(preds)} predictions.")
    else:
        print("Running Ground Truth Patches...")
    if os.path.exists(dataset) and dataset.endswith(".jsonl"):
        with open(dataset, encoding="utf-8") as f:
            instances = [json.loads(i) for i in f]
    elif split is not None:
        instances = load_dataset(dataset, split=split)
    else:
        instances = []
        ds = load_dataset(dataset)
        if hasattr(ds, "keys"):
            for key in ds.keys():
                instances.extend(ds[key])
        else:
            instances = ds
    all_months = []
    for instance in instances:
        month = get_instance_month(instance)
        if month is not None:
            all_months.append(month)
    if all_months:
        oldest_month = min(all_months)
        newest_month = max(all_months)
        start_month = normalize_month(start_month) if start_month is not None else oldest_month
        end_month = normalize_month(end_month) if end_month is not None else newest_month
        if start_month > end_month:
            raise ValueError(f"Invalid month range: start-month ({start_month}) is after end-month ({end_month}).")
        print(f"Month filter: {start_month} to {end_month} (dataset range: {oldest_month} to {newest_month})")
    elif start_month is not None or end_month is not None:
        raise ValueError("Month filtering requires 'created_at' field, but no parseable 'created_at' values were found.")
    if instance_ids is not None:
        print(f"Evaluating {instance_ids} ......")
    todos = []
    for idx in range(len(instances)):
        if instance_ids is not None and instances[idx]["instance_id"] not in instance_ids:
            continue
        # Create a proper dict copy since HuggingFace Dataset rows don't support in-place modification
        instance = dict(instances[idx])
        month = get_instance_month(instance)
        if all_months and month is not None and (month < start_month or month > end_month):
            continue
        if patch_dir.strip() != "gold" and instance["instance_id"] in preds.keys():
            instance["pred_patch"] = preds[instance["instance_id"]]["model_patch"]
            todos.append(instance)
        elif patch_dir.strip() == "gold":
            instance["pred_patch"] = instance["patch"]
            todos.append(instance)
    results = run_instances(todos, platform, workers, output_dir, overwrite != 0, collect_only)
    print("Submitted:", results["submitted"])
    print("Success:", results["success"])
    print("Failure:", results["failure"])
    print("Empty:", results["empty_patch"])
    print("Error:", results["error"])
    print("Incomplete:", results["incomplete"])
    if results["incomplete_ids"]:
        print("Incomplete IDs:", ", ".join(results["incomplete_ids"]))
    print("Evaluation ended successfully.")
    if patch_dir.strip().lower() == "gold":
        with open(os.path.join(output_dir, "gold_patch_evaluated_instances.jsonl"), "w", encoding="utf-8") as f:
            for instance in instances:
                if instance["instance_id"] in results["success_ids"]:
                    f.write(json.dumps(instance)+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SWE-bench instances")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--patch_dir", type=str, required=True, help="Path to patch file or 'gold' for ground truth")
    parser.add_argument("--platform", type=str, choices=["windows", "linux"], required=True, help="Platform to run on")
    parser.add_argument("--workers", type=int, required=True, help="Number of worker threads")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--overwrite", type=int, required=True, help="Overwrite existing results (0 or 1)")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to use")
    parser.add_argument("--instance_ids", type=str, nargs="+", default=None, help="Specific instance IDs to evaluate")
    parser.add_argument("--start-month", type=str, default=None, help="Start month (inclusive) in YYYY-MM format")
    parser.add_argument("--end-month", type=str, default=None, help="End month (inclusive) in YYYY-MM format")
    parser.add_argument("--collect-only", action="store_true", help="Only collect existing results without running unfinished instances")

    args = parser.parse_args()
    
    main(
        dataset=args.dataset,
        patch_dir=args.patch_dir,
        platform=args.platform,
        workers=args.workers,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        split=args.split,
        instance_ids=args.instance_ids,
        start_month=args.start_month,
        end_month=args.end_month,
        collect_only=args.collect_only,
    )
