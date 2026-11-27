from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from worldreforge.io import checked_file, contained


PAYLOAD_FIELDS = ("condition", "actions", "gt", "points", "reference")


def parse_manifest(content, root, *, check_files=()):
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(rows):
        raise ValueError("Empty manifest or duplicate sample IDs")

    for row in rows:
        sample_id = row["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
            raise ValueError(f"Invalid sample ID: {sample_id!r}")
        if row["domain"] not in {"sim", "droid", "real10"}:
            raise ValueError(f"{sample_id}: unknown domain {row['domain']!r}")
        if row["frames"] != 17 or row["score_hw"] != [544, 640]:
            raise ValueError(f"{sample_id}: expected 17 frames at 544 x 640")
        if row["conditioning"] != "measured_robot_state_endpoint_v1" or row["normalized_actions"] is not True:
            raise ValueError(f"{sample_id}: expected normalized measured robot states")
        if any(row.get(key) is None for key in ("condition", "actions", "gt")):
            raise ValueError(f"{sample_id}: condition, actions and gt are required")
        if (row.get("points") is None) != (row.get("reference") is None):
            raise ValueError(f"{sample_id}: provide both point queries and reference tracks")

        for key in PAYLOAD_FIELDS:
            reference = row.get(key)
            if reference is None:
                continue
            if key in check_files:
                checked_file(root, reference)
            else:
                contained(root, reference["path"])
    return rows


def load_manifest(path, data_root=None, verify_files=False):
    path = Path(path)
    if verify_files and data_root is None:
        raise ValueError("--data-root is required for payload checks")
    return parse_manifest(path.read_bytes(), data_root or path.parent,
                          check_files=PAYLOAD_FIELDS if verify_files else ())


def summary(rows):
    return {
        "windows": len(rows),
        "domains": dict(Counter(row["domain"] for row in rows)),
        "panels": dict(Counter(row["panel"] for row in rows)),
        "motion_reference_windows": sum(row.get("points") is not None for row in rows),
    }
