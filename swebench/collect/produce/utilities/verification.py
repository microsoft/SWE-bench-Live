from swebench.collect.produce.utilities.llm import LLMProvider
from datasets import load_dataset
import time
import ast
import pathlib
import subprocess
from typing import List, Tuple, Set
import shlex
import re, textwrap
from itertools import dropwhile, takewhile
from tqdm import tqdm  # shows progress bar, optional
import shutil

REPO_ROOT = pathlib.Path("repos")

class Verifier:
    def __init__(self, llm_provider, model):
        self.llm = LLMProvider(llm_provider, model)
    
    def prompt(self, description: str, test_case: list[tuple[str, str]], gold_patch: str) -> str:
        test_case_prompt = "\n===================================\n".join([f"{i[0]}:\n{i[1]}" for i in test_case])
        if len(test_case_prompt) > 150000:
            print("Warning! Exceeding context window! Chunking!", flush = True)
            test_case_prompt = test_case_prompt[:150000]
        return f"""
    You are a coding agent to debug git repositories.
    Your task is to rigorously evaluate the quality of a single issue description from the swe-bench dataset to dertermine whether it is actually solvable.

    Context:
    - Issue Description:
    {description}

    - Gold Patch (the ground truth fix):
    {gold_patch}

    - Fail-to-Pass Test Cases (fail initially, should pass after applying the gold patch):
    {test_case_prompt}


    Classify the instance into one of the following categories, based strictly on whether the issue description alone is sufficient to enable an agent to fix the issue and pass the test cases after trials and errors:

    1. The issue description has minor vagueness or missing details, so it is hard to understand, reproduce and solve the bug.
    2. The issue description is very vague, unclear, or incomplete, making it impossible to reproduce and then solve the bug.
    3. The issue description includes one or more solutions, but at least one is misleading or incorrect given the gold patch.
    4. The issue description is sufficient, but the provided test cases are too broad, missing or under-specifying required outputs or error formats described in the issue, so passing the test cases does not mean fixing the issue.
    5. The gold patch or the test cases require specific outputs, error messages, or formats NOT described in the issue, or have unnecessarily narrow / wrong restrictions for solving the described issue, causing correct solutions to fail.
    6. The ground truth fix, whether in diff patch or natual language, is directly stated in the issue description, making the fix trivial.
    7. The issue description is complete, precise, and useful, neither too hard nor too easy. This is a good instance.
    8. The instance has other flaws that make it unsolvable or misleading, such as environmental constraints, flaky behavior, or licensing problems that make solving or evaluating the bug impossible.

    Be skeptical and cautious before you classify the instance into category 7 (good instance). Try your best to find any flaws in the instance and classify into other categories.

    Output format:
    Return exactly 2 lines, separated by "\n". The first line, which begins with "Reasoning:", is your reasoning summary; and the second line, which begins with "Category:", is your category decision. No extra commentary.

    Output format example:
    Reasoning: Summarize the issue description and solution. Reason about which category the instance should fit into. 
    Category: Output only a single number from 1 to 8, no description.
    """

    def answer(self, description: str, test_case: list[tuple[str, str]], gold_patch: str) -> tuple[str, str]:
        messages = [{"role": "user", "content": self.prompt(description, test_case, gold_patch)}]
        response = []
        for i in range(3):
            try:
                response = self.llm.invoke(messages)
                response = response.content.strip().splitlines()
                break
            except Exception as e:
                print(e, flush = True)
                time.sleep(60)
        if len(response) != 2:
            return "","Error"
        try:
            reasoning = response[0].split(":")[1]
            category = str(int(response[1].split(":")[1]))
        except Exception as e:
            print(e)
        return category, reasoning
    
    def analyse_one_case(self, row):
        instance_id = row["instance_id"]
        repo_id = row["repo"]
        commit = row["base_commit"]
        gold_patch = row["patch"]
        test_patch = row["test_patch"]
        description = row['problem_statement'] # currently row['hints_text'] should not be seen by agents
        test_case_path = row["FAIL_TO_PASS"]
        test_case = get_testcase(instance_id, repo_id, commit, test_case_path, test_patch)
        [print(i[0], i[1][:2000], i[1][-500:], sep = "\n") for i in test_case]
        category, reasoning = self.answer(description, test_case, gold_patch)
        return {
            "instance_id": instance_id,
            "category": category,
            "reason": reasoning
        }


    def analyse_all(self, df: dict) -> list:
        """
        Run analyse_one_case on every row in the dataframe and save results to CSV.
        """
        REPO_ROOT.mkdir(exist_ok=True)
        records = []
        for row in tqdm(df, total=len(df)):
            result = self.analyse_one_case(row)
            print(result)
            records.append(result)
        shutil.rmtree(REPO_ROOT)
        
        return records


def _clone_if_needed(repo_id: str) -> pathlib.Path:
    repo_name = repo_id.split("/")[-1]
    repo_path = REPO_ROOT / repo_name
    if not repo_path.exists():
        url = f"https://github.com/{repo_id}.git"
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", url, str(repo_path)],
            check=True
        )
    return repo_path


def _checkout_commit(repo_path: pathlib.Path, commit: str) -> None:
    subprocess.run(["git", "fetch", "--quiet", "origin", commit],
                   cwd=repo_path, check=True)
    subprocess.run(["git", "checkout", "--quiet", commit],
                   cwd=repo_path, check=True)

GIT_APPLY_CMDS = [
    "git apply -v",                     # strict
    "git apply -v --reject",            # leave *.rej if partial
    "patch --batch --fuzz=5 -p1 -i",    # GNU patch, fuzzy match
]

def _apply_patch(repo_path: pathlib.Path, patch: str) -> None:
    """
    Try to apply `patch` with the same 3-stage strategy the SWE-bench
    harness uses.  Raises CalledProcessError only if **all** strategies
    fail.
    """
    if not patch.strip():
        return

    tmp_name = ".tmp_swebench.patch"
    patch_file = repo_path / ".tmp_swebench.patch"
    patch_file.write_text(patch, encoding="utf-8")

    last_exc = None
    for cmd in GIT_APPLY_CMDS:
        try:
            subprocess.run(
                shlex.split(f"{cmd} {tmp_name}"),
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            patch_file.unlink(missing_ok=True)
            return                       # ✅ applied
        except subprocess.CalledProcessError as exc:
            last_exc = exc               # remember why it failed
            continue                     # try next strategy

    # All strategies failed → re-raise with context
    stderr = (last_exc.stderr or "").strip()
    raise RuntimeError(
        f"❌ Failed to apply patch after {len(GIT_APPLY_CMDS)} attempts:\n{stderr}"
    ) from last_exc


def _parse_from_test_patch(test_patch: str, func_name: str) -> str:
        patch_lines = test_patch.splitlines()
        start_iter = dropwhile(
            lambda l: not re.search(rf'\bdef\s+{re.escape(func_name)}\b', l), patch_lines
        )
        snippet = list(
            takewhile(lambda l: not l.startswith('diff --git'), start_iter)
        )
        if snippet:
            return "\n".join(l for l in snippet)
        else:
            return ""

def _extract_test_context(source: str, selector: str, test_patch: str) -> str:
    """
    Trim `source` so that it keeps

    • all import statements
    • the selected test function (sync or async)
    • its containing class (if any)
    • any top-level helpers / fixtures referenced by that test
    """
    if not source:
        return ""
    if not selector:
        return source
    try:
        module = ast.parse(source)
    except Exception as e:
        print(e)
        print("Cannot parse, returning...")
        return ""

    # ── 1. split selector ──────────────────────────────────────────────
    parts = [p.split("[", 1)[0] for p in selector.split("::")]
    func_name = parts[-1]                 # final component = test function
    class_chain = parts[:-1]              # zero-to-many enclosing classes
    # assert func_name in source

    # ── 2. locate target function robustly ────────────────────────────
    target_func = None
    container_cls = None

    def walk(node, todo, parents):
        nonlocal target_func, container_cls
        if not todo:
            return
        head, *rest = todo
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef) and child.name == head:
                if rest:                             # need to go deeper
                    walk(child, rest, parents + [child])
                else:                                # selector ended on class?
                    return
            elif (not rest and isinstance(child, (ast.FunctionDef,
                                                  ast.AsyncFunctionDef))
                  and child.name == head):
                target_func = child
                container_cls = parents[-1] if parents else None
                return
        # keep searching siblings if not found
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                walk(child, todo, parents + [child])

    walk(module, parts, [])

    if target_func is None:
        # fallback: any function/async func with matching name anywhere
        for n in ast.walk(module):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func_name:
                target_func = n
                break

    if target_func is None:
        # ── Fallback-1: slice function by indentation ────────────────────
        lines = source.splitlines()
        pattern = re.compile(rf'^([ \t]*)((async\s+)?def)\s+{re.escape(func_name)}\b')

        match = None
        for idx, line in enumerate(lines):
            if pattern.match(line):
                match = idx
                indent = len(pattern.match(line).group(1).expandtabs(4))
                break

        if match is not None:
            # include decorators immediately above
            start = match
            while start > 0 and lines[start - 1].lstrip().startswith("@"):
                start -= 1
            start = max(0, start - 3)            # three lines of context

            end = len(lines) - 1
            for j in range(match + 1, len(lines)):
                l = lines[j]
                if l.strip() and (len(l) - len(l.lstrip())) <= indent:
                    end = j - 1
                    break
            snippet = "\n".join(lines[start:end + 1])
            return snippet

    if target_func is None:
        return ""
        # If we cannot find, just do not return anything!

    # ── 3. collect helper definitions referenced by the test — unchanged ──
    needed_names = set()
    for n in ast.walk(target_func):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            needed_names.add(n.id)            # bare names
        elif isinstance(n, ast.Attribute):    #  utils.helper() etc.
            needed_names.add(n.attr)
    helper_nodes = []
    for n in module.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in needed_names:
            helper_nodes.append(n)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in needed_names:
                    helper_nodes.append(n)
                    break
        if isinstance(n, ast.FunctionDef) and any(
                getattr(d, "id", "") == "fixture"
                for d in [getattr(d, "attr", "") for d in n.decorator_list]):
            helper_nodes.append(n)

    import_nodes = [n for n in module.body
                    if isinstance(n, (ast.Import, ast.ImportFrom))]

    # ── 4. build the line mask & emit ──────────────────────────────────
    def span(node):  # 0-based inclusive range
        return range(node.lineno - 1,
                     (node.end_lineno or node.lineno))

    mask = set()
    for n in (*import_nodes, *helper_nodes, target_func):
        mask.update(span(n))
    if container_cls:
        mask.update(span(container_cls))

    lines = source.splitlines()
    return "\n".join(lines[i] for i in sorted(mask))


def _wipe_worktree(repo_path: pathlib.Path) -> None:
    """Discard uncommitted changes and untracked files."""
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_path,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "clean", "-fd"], cwd=repo_path,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _parse_node_id(nid: str) -> tuple[str, str]:
    """Return (relative_file_path, selector) for both pytest and unittest ids."""
    if "::" in nid:                           # pytest-style
        file_part, selector = nid.split("::", 1)
        if "/" not in file_part:
            # Possibly missing file path — fallback to selector only
            return "", nid
        return file_part, selector

    # unittest-style:  <func> (<module>.<Class>)
    m = re.match(r"""(?P<func>[^\s(]+)           # test func
                    \s+\((?P<modcls>[^)]+)\)    # (module.Class)
                    (?:\s+\([^)]*\))?           # optional subTest/phase
                """, nid, re.X)

    if not m:
        print(f"Unrecognised node-id format: {nid}", flush = True)
        return "" , ""

    func      = m.group("func")
    modcls    = m.group("modcls")
    *mods, cls = modcls.split(".")
    tests_root = REPO_ROOT / "dummy" / "tests"  # dummy avoids needing repo_path here
    candidate_path = pathlib.Path(*mods).with_suffix(".py")
    if (tests_root / candidate_path).exists():
        path_parts = ["tests", *mods]
    else:
        path_parts = mods
    file_part = pathlib.Path(*path_parts).with_suffix(".py").as_posix()
    selector  = f"{cls}::{func}"
    return file_part, selector

# ---------- public API --------------------------------------------------------

def get_testcase(instance_id: str,
                 repo_id: str,
                 base_commit: str,
                 test_case_path: List[str],
                 test_patch: str) -> List[Tuple[str, str]]:
    """
    Parameters
    ----------
    repo_id          e.g. "pandas-dev/pandas"
    base_commit      SHA before the PR
    test_case_path   list[node-id] like ".../test_foo.py::Class::test_bar"
    test_patch       patch string that introduces / tweaks tests
    """
    repo_path = _clone_if_needed(repo_id)
    _wipe_worktree(repo_path)
    _checkout_commit(repo_path, base_commit)
    _apply_patch(repo_path, test_patch)           # <- now tests exist

    results = []
    for node_id in test_case_path:
        file_path, selector = _parse_node_id(node_id)
        func_name = selector.split("::")[-1]
        if not file_path:
            print(f"\033[91mWarning! {instance_id} -- {node_id} not found!\033[0m", flush = True)
            results.append((node_id, _parse_from_test_patch(test_patch, func_name))) 
            continue

        # try plain path, then tests/ prefix
        cand1 = repo_path / file_path
        cand2 = repo_path / "tests" / file_path
        cand3 = repo_path / "src" / file_path
        cand4 = repo_path / file_path.lstrip("tests/")  # repo root fallback
        abs_path = next((p for p in (cand1, cand2, cand3, cand4) if p.exists()),
                        None)
        if abs_path is None:
            print(f"\033[91mWarning! {instance_id} -- {node_id} not found!\033[0m", flush = True)
            results.append((node_id, _parse_from_test_patch(test_patch, func_name)))     # empty snippet
            continue

        with open(abs_path, encoding="utf-8") as f:
            src = f.read()
        snippet = _extract_test_context(src, selector, test_patch)
        if not snippet:
            print(f"\033[91mWarning! {instance_id} -- {selector} not found!\033[0m", flush = True)
            snippet = _parse_from_test_patch(test_patch, func_name)
        results.append((node_id, snippet))

    _wipe_worktree(repo_path)
    return results



