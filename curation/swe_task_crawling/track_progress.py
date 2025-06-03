import os

# Define the path to the directory
directory_path = "../swe_data_typescript"

# Initialize counters for the instances
instances_count = 0
instances_all_count = 0
pr_count = 0
repo_count = 0
processed_repo_count = 0

# Loop through the files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):
        if filename.endswith("instances.jsonl"):
            with open(file_path, 'r') as file:
                instances_count += sum(1 for line in file)
        elif filename.endswith("instances.jsonl.all"):
            processed_repo_count += 1
            with open(file_path, 'r') as file:
                instances_all_count += sum(1 for line in file)
        elif "pull2issue" in filename:
            repo_count += 1
            with open(file_path, 'r') as file:
                pr_count += sum(1 for line in file)
            
        

print(f"Total instances.jsonl line count: {instances_count}")
print(f"Total instances.jsonl.all line count: {instances_all_count}")
print(f"Total pull2issue line count: {pr_count}")
print(f"Total repo count: {repo_count}")
print(f"Total processed repo count: {processed_repo_count}")
