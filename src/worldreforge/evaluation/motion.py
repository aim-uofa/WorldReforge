import json
import math
from collections import Counter
from importlib.resources import files

import numpy as np

METRICS = ("ade_px", "fde_px", "motion_ade_px", "static_ade_px", "static_fde_px")


def load_policy():
    return json.loads(files("worldreforge").joinpath("resources/motion_policy.json").read_text())


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(points, ref, xy, predicted_visible, policy):
    count = len(points)
    ids = [point["id"] for point in points]
    gt, visible, states = ref["tracks"], ref["visibility"], ref["visibility_state"]
    _require(0 < count <= policy["total_maximum"] and len(set(ids)) == count,
             "Expected unique query IDs within the point quota")
    _require(gt.shape == xy.shape == (17, count, 2), "Expected 17 frames of XY tracks")
    _require(visible.shape == states.shape == predicted_visible.shape == (17, count),
             "Visibility and tracks must cover the same frames and points")
    _require(visible.dtype == bool and predicted_visible.dtype == bool, "Expected boolean visibility")
    _require(np.isfinite(gt).all() and np.isfinite(xy).all(), "Nonfinite point coordinates")
    if "point_ids" in ref:
        _require(list(map(str, ref["point_ids"])) == list(map(str, ids)), "Reference/query point order")

    maxima = policy["query_point_maxima"]
    for (camera, entity), used in Counter((p["camera"], p["entity"]) for p in points).items():
        _require(used <= maxima.get(camera, {}).get(entity, 0), f"Invalid point quota for {camera}/{entity}")

    bounds = np.array([policy["camera_bounds_xyxy"][point["camera"]] for point in points])
    inside = ((gt >= bounds[:, :2]) & (gt < bounds[:, 2:])).all(axis=-1)
    queries = np.array([point["query_xy"] for point in points])
    _require(np.max(np.abs(gt[0] - queries)) < 1e-3, "First-frame reference and query coordinates differ")
    _require(visible[0].all() and np.array_equal(visible, (states == 1) & inside),
             "Reference visibility must match the camera bounds and include every first-frame query")


def _missing():
    return dict(**{key: None for key in METRICS}, query_points=0, reference_pairs=0,
                possible_pairs=0, terminal_pairs=0, valid_future_frames=0,
                windows_with_ade=0, windows_with_fde=0, reference_fraction=None,
                prediction_visible_fraction=None, missing_reason='No valid motion reference available')


def missing_motion():
    return dict(unit="pixel", object=_missing(), gripper=_missing(),
                balanced={key: None for key in METRICS})


def score_entity(visible, predicted_visible, distance, displacement_error, static_distance):
    future = visible[1:]
    valid_frames = np.flatnonzero(future.any(axis=1)) + 1
    last = visible[-1]
    pairs = int(future.sum())

    def frame_average(errors):
        if not len(valid_frames):
            return None
        return math.fsum(errors[t, visible[t]].mean() for t in valid_frames) / len(valid_frames)

    def final_error(errors):
        return float(errors[-1, last].mean()) if last.any() else None

    return {
        "ade_px": frame_average(distance),
        "fde_px": final_error(distance),
        "motion_ade_px": frame_average(displacement_error),
        "static_ade_px": frame_average(static_distance),
        "static_fde_px": final_error(static_distance),
        "query_points": visible.shape[1],
        "reference_pairs": pairs,
        "possible_pairs": future.size,
        "terminal_pairs": int(last.sum()),
        "valid_future_frames": len(valid_frames),
        "windows_with_ade": int(len(valid_frames) > 0),
        "windows_with_fde": int(last.any()),
        "reference_fraction": pairs / future.size if future.size else None,
        "prediction_visible_fraction": float(predicted_visible[1:][future].mean()) if pairs else None,
    }


def score_canvas_pixel(points, reference, xy, predicted_visible, policy=None):
    if points is None:
        return missing_motion()

    prediction = np.asarray(xy)
    validate(points, reference, prediction, predicted_visible, load_policy() if policy is None else policy)
    gt = np.asarray(reference["tracks"], dtype=np.float64)
    prediction = prediction.astype(np.float64, copy=False)
    distance = np.linalg.norm(prediction - gt, axis=-1)
    gt_motion = gt - gt[:1]
    displacement_error = np.linalg.norm((prediction - prediction[:1]) - gt_motion, axis=-1)
    static_distance = np.linalg.norm(gt_motion, axis=-1)

    result = {"unit": "pixel"}
    labels = np.array([point["entity"] for point in points])
    for name, members in (("object", ["object"]), ("gripper", ["left_gripper", "right_gripper"])):
        selected = np.isin(labels, members)
        result[name] = score_entity(
            reference["visibility"][:, selected], predicted_visible[:, selected],
            distance[:, selected], displacement_error[:, selected], static_distance[:, selected],
        )

    balanced = {}
    for metric in METRICS:
        obj, gripper = result["object"][metric], result["gripper"][metric]
        balanced[metric] = (obj + gripper) / 2 if obj is not None and gripper is not None else None
    result["balanced"] = balanced
    return result
