from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

from worldreforge.evaluation.motion import load_policy
from worldreforge.evaluation.pixels import validate_video
from worldreforge.runtime import require_single_gpu


def load_tracker(repository, checkpoint, device="cuda:0"):
    import torch
    require_single_gpu()
    repository, checkpoint = Path(repository).resolve(), Path(checkpoint).resolve()
    if not (repository / "cotracker/predictor.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Explicit CoTracker repository and checkpoint required")
    sys.path.insert(0, str(repository))
    from cotracker.predictor import CoTrackerPredictor
    torch.manual_seed(62117)
    return CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, window_len=60).to(device).eval()


def track_video(model, video, points, device="cuda:0"):
    import torch
    validate_video(video)
    xy = np.full((17, len(points), 2), np.nan, np.float32)
    visible = np.zeros((17, len(points)), bool)
    policy = load_policy()
    for camera, (x0, y0, x1, y1) in policy["camera_bounds_xyxy"].items():
        indices = [i for i, p in enumerate(points) if p["camera"] == camera]
        if not indices:
            continue
        query = np.asarray([[0, points[i]["query_xy"][0] - x0,
                             points[i]["query_xy"][1] - y0] for i in indices], np.float32)
        if not ((query[:, 1:] >= 0).all() and (query[:, 1:] < [x1 - x0, y1 - y0]).all()):
            raise ValueError("Query outside its camera")
        crop = np.ascontiguousarray(video[:, y0:y1, x0:x1])
        tensor = torch.from_numpy(crop).permute(0, 3, 1, 2)[None].float().to(device)
        qt = torch.from_numpy(query)[None].to(device)
        with torch.inference_mode():
            tracks, mask = model(tensor, queries=qt, backward_tracking=False)
        tracks = tracks[0].float().cpu().numpy()
        tracks += np.asarray([x0, y0])
        xy[:, indices] = tracks
        visible[:, indices] = mask[0].bool().cpu().numpy()
    if not np.isfinite(xy).all() or np.max(abs(xy[0] - np.asarray([p["query_xy"] for p in points]))) >= 1e-3:
        raise ValueError("Invalid tracks or initial query mismatch")
    return xy, visible
