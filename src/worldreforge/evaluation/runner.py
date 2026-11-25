from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

from worldreforge.io import atomic_json, checked_file, contained, output_path, read_json
from worldreforge.evaluation.motion import score_canvas_pixel
from worldreforge.evaluation.pixels import score_pixels, aggregate_videos
from worldreforge.evaluation.predictions import load_prediction


def score_one(task):
    row, data_root, prediction_root, track_root, output, pixels_only = task
    cid = row["id"]
    gt_path = checked_file(data_root, row["gt"])
    prediction = load_prediction(prediction_root, row)
    gt = np.load(gt_path, allow_pickle=False, mmap_mode="r")
    pixels = score_pixels(prediction, gt, row.get("boxes"))
    motion = score_canvas_pixel(None, None, None, None)
    if row.get("points") and not pixels_only:
        if track_root is None:
            raise ValueError("Motion reference exists: run track and supply --track-root, or explicitly use --pixels-only")
        points = read_json(checked_file(data_root, row["points"]))["points"]
        with np.load(checked_file(data_root, row["reference"]), allow_pickle=False) as archive:
            reference = {k: archive[k] for k in archive.files}
        track_path = contained(track_root, f"{cid}/tracks.npz")
        receipt = read_json(track_path.parent / "COMPLETE.json")
        if receipt.get("id") != cid:
            raise ValueError("Tracking sample ID differs from the requested sample")
        with np.load(track_path, allow_pickle=False) as archive:
            if "point_ids" in archive and list(map(str, archive["point_ids"])) != [str(p["id"]) for p in points]:
                raise ValueError("Prediction/query point order differs")
            motion = score_canvas_pixel(points, reference, archive["tracks"], archive["visibility"])
    result = dict(id=cid, domain=row["domain"], panel=row["panel"], asset=row.get("asset"),
                  pixels=pixels, motion=motion,
                  motion_status="explicitly_skipped" if pixels_only else ("scored" if row.get("points") else "no_qualified_reference"))
    atomic_json(Path(output) / "cases" / f"{cid}.json", result)
    return result


def score(rows, data_root, prediction_root, track_root, output, workers=1, pixels_only=False):
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    tasks = [(r, str(data_root), str(prediction_root), str(track_root) if track_root else None,
              str(output), pixels_only) for r in rows]
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        results = [score_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(score_one, tasks, chunksize=1))
    groups = {"all": results}
    for key in ("domain", "panel"):
        for value in sorted({r[key] for r in results}):
            groups[f"{key}:{value}"] = [r for r in results if r[key] == value]
    report = {k: aggregate_videos(v) for k, v in groups.items()}
    atomic_json(output / "summary.json", report)
    atomic_json(output / "COMPLETE.json", {"status": "passed", "windows": len(results),
                "pixels_only": pixels_only, "ids": [r["id"] for r in results]})
    return report
