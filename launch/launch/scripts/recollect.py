
import json
from pathlib import Path
from typing import Literal
from launch.core.runtime import SetupRuntime
from fire import Fire
from launch.scripts.parser import run_get_pertest_cmd, run_parser

def main(workspace: str, platform: Literal["linux", "windows"]):
    '''
    workspace: the place that stores setup.jsonl, organize.jsonl, playground...
    '''
    workspace = Path(workspace)
    playground = workspace / "playground"
    output_jsonl = workspace / f"organize.jsonl"
    swe_instances = []
    max_len = 5000_0000
    all_len = 0
    for subfolder in playground.iterdir():
        if not subfolder.is_dir():
            continue

        instance_path = subfolder / "instance.json"
        result_path = subfolder / "result.json"

        if not instance_path.exists() or not result_path.exists():
            continue
        
        instance = json.loads(instance_path.read_text())
        result = json.loads(result_path.read_text())

        if not result.get("organize_completed", False):
            continue
        
        if not result.get("unittest_generator", ""):
            continue

        try:
            container = SetupRuntime.from_launch_image(result["docker_image"], result["instance_id"], platform)
        except:
            print("pulling image timeout, skipping")
            continue
        test_output = container.send_command(";".join(result["test_commands"])).output # unstripped / full result
        test_status: dict[str, str] = run_parser(result["log_parser"], test_output)
        test_list = [i for i in test_status.keys() if test_status[i] == "pass"]
        pertest_cmd: dict[str, str] = run_get_pertest_cmd(result["unittest_generator"], test_list)
        # for debug
        print(test_status)
        print(pertest_cmd, flush = True)
        container.cleanup()

        if not pertest_cmd:
            continue

        swe_instance = {
            **instance,
            "setup_cmds": result.get("setup_commands", []),
            "test_cmds": result["test_commands"],
            "log_parser": result.get("log_parser", "pytest"),
            "docker_image": result.get("docker_image", f"karinali20011210/migbench:{instance["instance_id"]}_{platform}"),
        }
        swe_instance["rebuild_cmds"] = result["rebuild_commands"]
        swe_instance["test_status"] = test_status
        swe_instance["pertest_command"] = pertest_cmd
        swe_instance["log_parser"] = result["log_parser"]
        swe_instance["per_test_command_generator"] = result["unittest_generator"]

        swe_instances.append(swe_instance)

        all_len += len(str(pertest_cmd))
        if all_len > max_len:
            break

    with open(output_jsonl, "w") as f:
        for i in swe_instances:
            json.dump(i, f)
            f.write("\n")

if __name__ == "__main__":
    Fire(main)