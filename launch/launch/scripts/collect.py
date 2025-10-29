import json
from pathlib import Path
from typing import Literal, Optional
from fire import Fire

def main(
    workspace: str,
    platform: Literal["linux", "windows"] = "linux",
    step: Literal["setup", "organize"] = "setup",
    instance_ids: Optional[list[str]] = None
):
    workspace = Path(workspace)
    playground = workspace / "playground"
    output_jsonl = workspace / f"{step}.jsonl"
    swe_instances = []
    for subfolder in playground.iterdir():
        if not subfolder.is_dir():
            continue

        instance_path = subfolder / "instance.json"
        result_path = subfolder / "result.json"

        if not instance_path.exists() or not result_path.exists():
            continue

        instance = json.loads(instance_path.read_text())
        result = result_path.read_text()
        if not result.strip():
            continue
        result = json.loads(result_path.read_text())

        if (instance_ids is not None) and (result["instance_id"] not in instance_ids):
            continue

        if step == "setup" and (not result.get("completed", False)):
            continue
        if step == "organize" and (not result.get("organize_completed", False)):
            continue
        
        swe_instance = {
            **instance,
            "setup_cmds": result.get("setup_commands", []),
            "test_cmds": result["test_commands"],
            "log_parser": result.get("log_parser", "pytest"),
            "docker_image": result.get("docker_image", f"karinali20011210/migbench:{instance["instance_id"]}_{platform}"),
        }
        if result.get("rebuild_commands", ""):
            swe_instance["rebuild_cmds"] = result["rebuild_commands"]
        if result.get("test_status", ""):
            swe_instance["test_status"] = result["test_status"]
        if result.get("pertest_command", ""):
            swe_instance["pertest_command"] = result["pertest_command"]
        if result.get("log_parser", ""):
            swe_instance["log_parser"] = result["log_parser"]
        if result.get("unittest_generator", ""):
            swe_instance["per_test_command_generator"] = result["unittest_generator"]

        swe_instances.append(swe_instance)

    with open(output_jsonl, "w") as f:
        for instance in swe_instances:
            f.write(json.dumps(instance) + "\n")
    print(f"Saved {len(swe_instances)} instances to {output_jsonl}")

if __name__ == "__main__":
    Fire(main)