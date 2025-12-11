from datasets import load_dataset, DatasetDict, Features, Value, Sequence, Dataset
import json

ds = {}

# 1. 加载原始数据集
with open("logs/c/verified_instances.jsonl") as f:
    ds["c"] = [json.loads(i) for i in f]
with open("logs/cpp/verified_instances.jsonl") as f:
    ds["cpp"] = [json.loads(i) for i in f]

print("data loaded")

all_fields = ["repo", "pull_number", "instance_id", "issue_numbers", "base_commit", 
              "patch", "test_patch", "problem_statement", "hints_text", "all_hints_text",
              "commit_urls", "created_at", "commit_url", "rebuild_cmds","test_cmds", "print_cmds",
              "log_parser", "FAIL_TO_PASS", "PASS_TO_PASS", "docker_image"]
 

def get_default_image_name(instance_id: str, platform) -> str:
    if platform == "linux":
        med = "x86_64"
    else:
        med = "win"
    name = instance_id.replace("__", "_1776_").lower()
    image = f"starryzhang/sweb.eval.{med}.{name}"
    return image

# only retain all_fields in dataset
for key in ds.keys():
    for idx in range(len(ds[key])):
        ds[key][idx] = {
            field: ds[key][idx][field] for field in all_fields
        }
        ds[key][idx]["docker_image"] = get_default_image_name(ds[key][idx]["instance_id"], "linux")
    ds[key] = Dataset.from_list(ds[key])

# 2. 转换字段：pull_number 为 string，其余为 string 的数组类型
# 你可以使用 map + features 来精确控制字段类型
 
# 定义新的字段类型
new_features = Features({
    **ds["c"].features,  # 保留其他字段
    "pull_number": Value("string"),
    "docker_image": Value("string"),
    "issue_numbers": Sequence(Value("string")),
    "commit_urls": Sequence(Value("string")),
    "rebuild_cmds": Sequence(Value("string")),
    "test_cmds": Sequence(Value("string")),
    "print_cmds": Sequence(Value("string")),
    "FAIL_TO_PASS": Sequence(Value("string")),
    "PASS_TO_PASS": Sequence(Value("string")),
})
 
# 使用 map 转换字段内容的类型
def convert_types(example):
    example["pull_number"] = str(example["pull_number"])
 
    # 确保是字符串数组
    for field in ["issue_numbers", "commit_urls", "rebuild_cmds","test_cmds", "print_cmds", "FAIL_TO_PASS", "PASS_TO_PASS"]:
        if field in example and example[field] is not None:
            example[field] = [str(x) for x in example[field]]
        else:
            example[field] = []  # 或者保留为 None
    return example
 
for key in ds.keys():

    ds[key] = ds[key].map(convert_types)
    
    # 设置新的 features
    ds[key] = ds[key].cast(new_features)
 

print(len(ds["c"]))
 
dataset_dict = DatasetDict(ds)
 
dataset_dict.push_to_hub("SWE-bench-Live/MultiLang")