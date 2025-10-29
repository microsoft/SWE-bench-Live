"""
Script to merge all .jsonl files from a specified folder into a single output file.
"""

import argparse
import json
from pathlib import Path


def merge_jsonl_files(input_folder: str, output_file: str, input_repo: str):
    """
    Merge all .jsonl files from input_folder into a single output file.
    
    Args:
        input_folder (str): Path to the folder containing .jsonl files
        output_file (str): Path to the output merged file. If None, will be created in the input folder.
    """
    input_path: Path = Path(input_folder)

    input_repo: Path = Path(input_repo)
    
    # Check if input folder exists
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return False
    
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a directory.")
        return False
    
    if not input_repo.exists():
        print(f"Error: '{input_repo}' not found!")
        return False
    
    repo_list = [json.loads(i) for i in input_repo.read_text().splitlines()]
    repo_dict = {i["full_name"]: i for i in repo_list}
    
    # Find all .jsonl files in the input folder
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No .jsonl files found in '{input_folder}'")
        return False
    
    # Set default output file name if not provided
    output_file = Path(output_file)
    
    # Merge all files
    try:
        with open(output_file, 'w', encoding='utf-8') as outf:
            for jsonl_file in sorted(jsonl_files):
                with open(jsonl_file, 'r', encoding='utf-8') as inf:
                    for line in inf:
                        line = line.strip()
                        if line:  # Skip empty lines
                            # Validate JSON format
                            try:
                                l = json.loads(line)
                                l["language"] = repo_dict[l["repo"]]["language"]
                                outf.write(json.dumps(l) + '\n')
                            except json.JSONDecodeError as e:
                                print(f"  Warning: Invalid JSON in {jsonl_file.name}: {e}")
                                continue
        return True
    except Exception as e:
        print(f"Error during merge: {e}")
        return False


def main():
    """Main function to handle command line arguments and execute merge."""
    parser = argparse.ArgumentParser(
        description="Merge all .jsonl files from a folder into a single output file",
    )
    
    parser.add_argument(
        '--input_folder',
        dest='input_folder',
        default='output/tasks',
        help='Folder containing .jsonl files to merge (default: output/tasks)'
    )
    
    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        help='Output file path (default: merged_tasks.jsonl in input folder)'
    )

    parser.add_argument(
        '--input_repos',
        dest='input_repos',
        help='The original filtered repo list'
    )
    
    args = parser.parse_args()
    
    # Execute merge
    success = merge_jsonl_files(args.input_folder, args.output_file, args.input_repos)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
