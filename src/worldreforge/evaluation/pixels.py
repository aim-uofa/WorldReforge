from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
from skimage.metrics import structural_similarity


def mean(values):
    values = [float(v) for v in values if v is not None]
    return math.fsum(values) / len(values) if values else None


def validate_video(video):
    if video.shape != (17, 544, 640, 3) or video.dtype != np.uint8:
        raise ValueError("Expected uint8 [17,544,640,3]; never pad or stretch native output")


def score_pixels(prediction, gt, boxes=None):
    validate_video(prediction)
    validate_video(gt)
    frames = []
    for frame in range(1, 17):
        p, g = prediction[frame].astype(np.float32) / 255, gt[frame].astype(np.float32) / 255
        error = (p - g) ** 2
        static = ((gt[0].astype(np.float32) - gt[frame].astype(np.float32)) / 255) ** 2
        dp = (prediction[frame].astype(np.float32) - prediction[frame - 1].astype(np.float32)) / 255
        dg = (gt[frame].astype(np.float32) - gt[frame - 1].astype(np.float32)) / 255
        row = dict(mse=float(error.mean()),
                   ssim=float(structural_similarity(p, g, channel_axis=2, data_range=1)),
                   static_mse=float(static.mean()),
                   temporal_difference_mse=float(((dp - dg) ** 2).mean()),
                   roi_mse=None, roi_ssim=None, static_roi_mse=None)
        roi = []
        for sequence in boxes or []:
            bounds = np.asarray(sequence[frame])
            if bounds.shape != (4,) or not np.isfinite(bounds).all():
                raise ValueError("Invalid ROI coordinates")
            x0, y0 = np.floor(bounds[:2]).astype(int)
            x1, y1 = np.ceil(bounds[2:]).astype(int)
            if not (0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 544 and min(x1-x0, y1-y0) >= 2):
                raise ValueError("ROI must already be in native640 scoring coordinates")
            p_roi, g_roi = p[y0:y1, x0:x1], g[y0:y1, x0:x1]
            size = min(y1 - y0, x1 - x0)
            ssim = (float(structural_similarity(p_roi, g_roi, channel_axis=2, data_range=1,
                     win_size=min(7, (size // 2) * 2 - 1))) if size >= 3 else None)
            roi.append(dict(roi_mse=float(((p_roi - g_roi) ** 2).mean()), roi_ssim=ssim,
                            static_roi_mse=float(static[y0:y1, x0:x1].mean())))
        if roi:
            row.update({k: mean(r[k] for r in roi) for k in roi[0]})
        frames.append(row)
    result = {k: mean(r[k] for r in frames) for k in frames[0]}
    for name in ("mse", "roi_mse", "static_mse"):
        result[name.replace("mse", "psnr_db")] = (
            -10 * math.log10(max(result[name], 1e-15)) if result[name] is not None else None)
    result["frames_scored"] = 16
    return result


def aggregate_videos(rows):
    flat = []
    for row in rows:
        item = dict(row["pixels"])
        for entity in ("object", "gripper"):
            for key in ("ade_px", "fde_px", "motion_ade_px", "static_ade_px", "static_fde_px"):
                item[entity + "_" + key] = row["motion"][entity][key]
        flat.append(item)
    keys = sorted(set().union(*(r.keys() for r in flat))) if flat else []
    return {"videos": len(rows), "metrics": {
        k: {"mean": mean(r.get(k) for r in flat),
            "valid_videos": sum(r.get(k) is not None for r in flat)} for k in keys if k != "frames_scored"}}

def summarize_cases(cases):
    groups = defaultdict(list)
    for case in cases:
        for field in ("domain", "panel"):
            groups[f"{field}:{case[field]}"].append(case)
    return {
        "all": aggregate_videos(cases),
        **{name: aggregate_videos(group) for name, group in sorted(groups.items())},
    }
