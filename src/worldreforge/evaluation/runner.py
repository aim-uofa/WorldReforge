from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from worldreforge.io import atomic_json, checked_file, contained, output_path, read_json
from worldreforge.evaluation.motion import missing_motion, score_canvas_pixel
from worldreforge.evaluation.pixels import score_pixels, summarize_cases
from worldreforge.evaluation.predictions import load_prediction


def score_one(task):
    row, data_root, prediction_root, track_root, output, pixels_only = task
    cid = row["id"]
    gt_path = checked_file(data_root, row["gt"])
    prediction = load_prediction(prediction_root, row)
    gt = np.load(gt_path, allow_pickle=False, mmap_mode="r")
    pixels = score_pixels(prediction, gt, row.get("boxes"))
    if pixels_only or not row.get("points"):
        motion = missing_motion()
        status = "explicitly_skipped" if pixels_only else "no_qualified_reference"
    else:
        motion = score_motion(row, data_root, track_root)
        status = "scored"

    result = {
        "id": cid, "domain": row["domain"], "panel": row["panel"], "asset": row.get("asset"),
        "pixels": pixels, "motion": motion, "motion_status": status,
    }
    atomic_json(Path(output) / "cases" / f"{cid}.json", result)
    return result


def score_motion(row, data_root, track_root):
    if track_root is None:
        raise ValueError("Supply --track-root for motion evaluation, or use --pixels-only")
    points = read_json(checked_file(data_root, row["points"]))["points"]
    tracks = contained(track_root, f"{row['id']}/tracks.npz")
    if read_json(tracks.parent / "COMPLETE.json")["id"] != row["id"]:
        raise ValueError("Tracking sample ID differs from the requested sample")

    reference = checked_file(data_root, row["reference"])
    with np.load(reference, allow_pickle=False) as gt, np.load(tracks, allow_pickle=False) as pred:
        if "point_ids" in pred and list(map(str, pred["point_ids"])) != [str(p["id"]) for p in points]:
            raise ValueError("Prediction/query point order differs")
        return score_canvas_pixel(points, dict(gt), pred["tracks"], pred["visibility"])


def score(rows, data_root, prediction_root, track_root, output, workers=1, pixels_only=False):
    if workers < 1:
        raise ValueError("workers must be positive")
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    tasks = [(row, data_root, prediction_root, track_root, output, pixels_only) for row in rows]
    if workers == 1:
        results = [score_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(score_one, tasks, chunksize=1))
    report = summarize_cases(results)
    atomic_json(output / "summary.json", report)
    atomic_json(output / "COMPLETE.json", {"status": "passed", "windows": len(results),
                "pixels_only": pixels_only, "ids": [r["id"] for r in results]})
    return report
