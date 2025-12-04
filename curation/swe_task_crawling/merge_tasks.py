"""
Script to merge all .jsonl files from a specified folder into a single output file.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

REQUIRED_FIELDS = ("instance_id", "repo", "pull_number")


def merge_jsonl_files(input_folder: str, output_file: str = None, validate: bool = False):
    """
    Merge all .jsonl files from input_folder into a single output file.
    
    Args:
        input_folder (str): Path to the folder containing .jsonl files
        output_file (str): Path to the output merged file. If None, will be created in the input folder.
    """
    input_path = Path(input_folder)
    
    # Check if input folder exists
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return False
    
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a directory.")
        return False
    
    # Find all .jsonl files in the input folder
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No .jsonl files found in '{input_folder}'")
        return False
    
    # Set default output file name if not provided
    if output_file is None:
        output_file = input_path / "merged_tasks.jsonl"
    output_file = Path(output_file)
    
    # Merge all files
    try:
        seen_ids: Set[str] = set()
        with open(output_file, 'w', encoding='utf-8') as outf:
            for jsonl_file in sorted(jsonl_files):
                with open(jsonl_file, 'r', encoding='utf-8') as inf:
                    for line in inf:
                        line = line.strip()
                        if line:  # Skip empty lines
                            # Validate JSON format
                            try:
                                record = json.loads(line)
                                if validate:
                                    _validate_record(record, jsonl_file.name, seen_ids)
                                outf.write(json.dumps(record, ensure_ascii=False) + '\n')
                            except json.JSONDecodeError as e:
                                print(f"  Warning: Invalid JSON in {jsonl_file.name}: {e}")
                                continue
        return True
    except Exception as e:
        print(f"Error during merge: {e}")
        return False


def _validate_record(record: Dict, source_name: str, seen_ids: Set[str]) -> None:
    """Ensure a merged record has required fields and a unique instance id."""
    missing: List[str] = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(
            f"{source_name}: missing required fields {missing} for instance {record.get('instance_id')}"
        )
    instance_id = record["instance_id"]
    if instance_id in seen_ids:
        raise ValueError(f"Duplicate instance_id '{instance_id}' found while merging ({source_name}).")
    seen_ids.add(instance_id)


def main():
    """Main function to handle command line arguments and execute merge."""
    parser = argparse.ArgumentParser(
        description="Merge all .jsonl files from a folder into a single output file",
    )
    
    parser.add_argument(
        'input_folder',
        nargs='?',
        default='output/tasks',
        help='Folder containing .jsonl files to merge (default: output/tasks)'
    )
    
    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        help='Output file path (default: merged_tasks.jsonl in input folder)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate required fields and ensure unique instance_ids while merging.'
    )
    
    args = parser.parse_args()
    
    # Execute merge
    success = merge_jsonl_files(args.input_folder, args.output_file, validate=args.validate)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
