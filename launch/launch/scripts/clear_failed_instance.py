import os
import json
import stat
import shutil
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--base_dir", type = str)
args = parser.parse_args()

def on_rm_error(func, path, exc_info):
    # Change the file to be writable and try again
    os.chmod(path, stat.S_IWRITE)
    func(path)

for instance_id in os.listdir(args.base_dir):
    instance_path = os.path.join(args.base_dir, instance_id)
    result_path = os.path.join(instance_path, "result.json")
    if os.path.isdir(instance_path) and not os.path.isfile(result_path):
        shutil.rmtree(instance_path, onexc=on_rm_error)
    if os.path.isdir(instance_path) and os.path.isfile(result_path):
        with open(result_path, "r") as f:
            d = json.load(f)
        if not d.get("completed", False):
            # Check exception before removing
            exception = d.get("exception", "")
            if "launch failed" in exception.lower():
                print("skipping", instance_path, "- launch failed")
                continue
            # Remove the directory if "completed" is False and not launch failed
            print("deleting", instance_path)
            shutil.rmtree(instance_path, onexc=on_rm_error)
    

