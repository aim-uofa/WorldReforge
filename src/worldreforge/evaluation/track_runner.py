from __future__ import annotations

import numpy as np

from worldreforge.io import atomic_file, atomic_json, checked_file, output_path, read_json
from worldreforge.evaluation.predictions import load_prediction
from worldreforge.runtime import bind_gpu, configure_storage


def track(rows, data_root, prediction_root, repository, checkpoint, output, gpu=0):
    out = output_path(output)
    selected = [r for r in rows if r.get("points") is not None]
    device = bind_gpu(gpu) if selected else None
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "PLAN.json", dict(device=device, ids=[r["id"] for r in selected]))
    ids = []
    try:
        if selected:
            configure_storage(out / "runtime")
            from worldreforge.evaluation.tracking import load_tracker, track_video
            model = load_tracker(repository, checkpoint)
            for row in selected:
                prediction = load_prediction(prediction_root, row)
                points = read_json(checked_file(data_root, row["points"]))["points"]
                xy, vis = track_video(model, prediction, points)
                folder = out / row["id"]
                folder.mkdir()
                with atomic_file(folder / "tracks.npz") as stream:
                    np.savez_compressed(stream, tracks=xy, visibility=vis, point_ids=np.asarray([p["id"] for p in points]))
                atomic_json(folder / "COMPLETE.json", dict(id=row["id"], status="passed"))
                ids.append(row["id"])
        receipt = dict(status="passed", windows=len(ids), ids=ids)
        atomic_json(out / "COMPLETE.json", receipt)
        return receipt
    except BaseException as error:
        atomic_json(out / "FAILED.json", dict(status="failed", error_type=type(error).__name__, error=str(error)))
        raise
