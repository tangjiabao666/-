#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create reproducible train/dev JSONL splits with group-level leakage control."""

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def infer_group(record: Dict[str, object], group_field: str, group_regex: str) -> str:
    if group_field and record.get(group_field) not in {None, ""}:
        return str(record[group_field])
    joined = " ".join(
        str(record.get(key, ""))
        for key in ["speaker", "说话人", "唤醒音频", "识别音频", "id"]
    )
    if group_regex:
        match = re.search(group_regex, joined)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    enroll = str(record.get("唤醒音频", ""))
    mixed = str(record.get("识别音频", ""))
    if enroll or mixed:
        key = "|".join([enroll.rsplit("/", 1)[-1].split("_")[-1], mixed.rsplit("/", 1)[-1].split("_")[-1]])
        return key
    return hashlib.sha1(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a JSONL file by group.")
    parser.add_argument("--input", default="data/raw/test_set_a/labels.jsonl")
    parser.add_argument("--train_out", default="data/splits/train.jsonl")
    parser.add_argument("--dev_out", default="data/splits/dev.jsonl")
    parser.add_argument("--dev_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--group_field", default="")
    parser.add_argument("--group_regex", default="")
    args = parser.parse_args()

    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    with open(args.input, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            groups[infer_group(record, args.group_field, args.group_regex)].append(record)

    group_keys = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(group_keys)
    dev_count = max(1, int(len(group_keys) * args.dev_ratio))
    dev_groups = set(group_keys[:dev_count])

    train_rows: List[Dict[str, object]] = []
    dev_rows: List[Dict[str, object]] = []
    for key in group_keys:
        if key in dev_groups:
            dev_rows.extend(groups[key])
        else:
            train_rows.extend(groups[key])

    for path, rows in [(Path(args.train_out), train_rows), (Path(args.dev_out), dev_rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"groups: total={len(group_keys)} train={len(group_keys) - len(dev_groups)} dev={len(dev_groups)}")
    print(f"rows  : train={len(train_rows)} dev={len(dev_rows)}")
    print(f"train : {args.train_out}")
    print(f"dev   : {args.dev_out}")


if __name__ == "__main__":
    main()
