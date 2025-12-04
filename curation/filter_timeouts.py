"""
Filter Stage 2 raw_tasks.jsonl by removing instances that timed out in Stage 3.

Usage:
    python curation/filter_timeouts.py \
        --raw-tasks curation/output/raw_tasks.jsonl \
        --timeouts curation/output/timeouts.txt \
        --output curation/output/raw_tasks.filtered.jsonl

The timeouts file should contain one instance_id per line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove timed-out instance IDs from raw_tasks.jsonl"
    )
    parser.add_argument(
        "--raw-tasks",
        type=Path,
        default=Path("curation/output/raw_tasks.jsonl"),
        help="Path to Stage 2 raw_tasks.jsonl",
    )
    parser.add_argument(
        "--timeouts",
        type=Path,
        default=Path("curation/output/timeouts.txt"),
        help="File containing one instance_id per line to exclude",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("curation/output/raw_tasks.filtered.jsonl"),
        help="Destination JSONL after filtering",
    )
    return parser.parse_args()


def load_timeouts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            ids.add(stripped)
    return ids


def main() -> None:
    args = parse_args()
    timeouts = load_timeouts(args.timeouts)

    kept = []
    dropped = 0
    with args.raw_tasks.open() as src:
        for line in src:
            line = line.rstrip("\n")
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("instance_id") in timeouts:
                dropped += 1
                continue
            kept.append(line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + ("\n" if kept else ""))

    print(
        f"Filtered {dropped} instance(s); wrote {len(kept)} entries to {args.output}"
    )


if __name__ == "__main__":
    main()

