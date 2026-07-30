"""
Validate training data against Replicate's fine-tuning format and print
dataset statistics. Run this before kicking off training.

Usage:
    python src/data_prep.py --file data/train.jsonl
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


def validate(path: Path) -> bool:
    prompt_lens, completion_lens = [], []
    errors = 0

    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  Line {i}: invalid JSON")
                errors += 1
                continue
            if not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str):
                print(f"  Line {i}: missing 'prompt' or 'completion' string fields")
                errors += 1
                continue
            if not row["prompt"].strip() or not row["completion"].strip():
                print(f"  Line {i}: empty prompt or completion")
                errors += 1
                continue
            prompt_lens.append(len(row["prompt"].split()))
            completion_lens.append(len(row["completion"].split()))

    n = len(prompt_lens)
    print(f"\nFile: {path}")
    print(f"Valid rows: {n} | Errors: {errors}")
    if n:
        print(f"Prompt words     -> mean {statistics.mean(prompt_lens):.0f}, "
              f"min {min(prompt_lens)}, max {max(prompt_lens)}")
        print(f"Completion words -> mean {statistics.mean(completion_lens):.0f}, "
              f"min {min(completion_lens)}, max {max(completion_lens)}")
        approx_tokens = (statistics.mean(prompt_lens) + statistics.mean(completion_lens)) * 1.35
        print(f"Approx tokens/example: {approx_tokens:.0f} "
              f"(keep total sequence under model context limit)")
    return errors == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/train.jsonl")
    args = parser.parse_args()
    ok = validate(Path(args.file))
    sys.exit(0 if ok else 1)
