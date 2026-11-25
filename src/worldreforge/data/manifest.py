from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from worldreforge.io import checked_file, contained


def load_manifest(path, data_root=None, verify_files=False):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Empty manifest or duplicate sample IDs")
    for row in rows:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["id"]):
            raise ValueError("Unsafe sample ID")
        if row["domain"] not in {"sim", "droid", "real10"}:
            raise ValueError("Unknown domain")
        if row["frames"] != 17 or row["score_hw"] != [544, 640]:
            raise ValueError("Expected 17 observations, native 544 x 640 scoring")
        if row["conditioning"] != "measured_robot_state_endpoint_v1":
            raise ValueError("Commands cannot substitute for measured-state conditioning")
        if row["normalized_actions"] is not True:
            raise ValueError("Release actions must already be normalized exactly once")
        for field in ("condition", "actions", "gt", "points", "reference"):
            reference = row.get(field)
            if reference is not None:
                contained(data_root or Path(path).parent, reference["path"])
                if verify_files:
                    if data_root is None:
                        raise ValueError("--data-root is required for payload checks")
                    checked_file(data_root, reference)
        if any(row.get(k) is None for k in ("condition", "actions", "gt")):
            raise ValueError("Missing required media/state references")
        if (row.get("points") is None) != (row.get("reference") is None):
            raise ValueError("Queries and reference tracks must be supplied together")
    return rows


def summary(rows):
    return {"windows": len(rows), "domains": dict(Counter(r["domain"] for r in rows)),
            "panels": dict(Counter(r["panel"] for r in rows)),
            "motion_reference_windows": sum(r.get("points") is not None for r in rows)}
