import math
import numpy as np
from importlib.resources import files
import json

METRICS = ("ade_px", "fde_px", "motion_ade_px", "static_ade_px", "static_fde_px")

def load_policy():
    return json.loads(files("worldreforge").joinpath("resources/motion_policy.json").read_text())

def _require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(points, ref, xy, predicted_visible, policy):
    n = len(points)
    _require(0 < n <= policy['total_maximum'], 'Nonempty existing query set, at most20')
    _require(len({p['id'] for p in points}) == n, 'Unique point IDs')
    _require(ref['tracks'].shape == xy.shape == (17, n, 2), 'Full17-frame point trajectories')
    _require(ref['visibility'].shape == predicted_visible.shape == ref['visibility_state'].shape == (17, n), 'Visibility shapes')
    _require(np.isfinite(ref['tracks']).all() and np.isfinite(xy).all(), 'No missing/nonfinite predicted point may be silently excluded')
    _require(np.asarray(ref['visibility']).dtype == np.bool_ and np.asarray(predicted_visible).dtype == np.bool_, 'Boolean visibility arrays')
    if 'point_ids' in ref:
        _require(list(map(str, ref['point_ids'])) == [str(p['id']) for p in points], 'Reference/query point order')
    inside = np.zeros((17, n), dtype=bool)
    counts = {}
    for i, point in enumerate(points):
        camera, entity = point['camera'], point['entity']
        _require(camera in policy['query_point_maxima'] and entity in policy['query_point_maxima'][camera], 'Known camera/entity')
        counts[camera, entity] = counts.get((camera, entity), 0) + 1
        _require(counts[camera, entity] <= policy['query_point_maxima'][camera][entity], 'Frozen query quota')
        bounds = policy['camera_bounds_xyxy'][camera]
        gt = ref['tracks'][:, i]
        inside[:, i] = (gt >= bounds[:2]).all(-1) & (gt < bounds[2:]).all(-1)
        _require(np.max(np.abs(gt[0] - point['query_xy'])) < 1e-3, 'Existing global-canvas query coordinates')
    mask = np.asarray(ref['visibility'])
    _require(np.array_equal(mask, (ref['visibility_state'] == 1) & inside), 'Preserve original GT visibility and in-camera support')
    _require(mask[0].all(), 'Every query must be GT-visible on frame0')


def _mean(values):
    v = [float(x) for x in values if x is not None]
    return math.fsum(v) / len(v) if v else None


def _missing():
    return dict(**{key: None for key in METRICS}, query_points=0, reference_pairs=0,
                possible_pairs=0, terminal_pairs=0, valid_future_frames=0,
                windows_with_ade=0, windows_with_fde=0, reference_fraction=None,
                prediction_visible_fraction=None, missing_reason='No valid motion reference available')


def score_canvas_pixel(points, reference, xy, predicted_visible, policy=None):
    policy = load_policy() if policy is None else policy
    if points is None:
        return dict(unit='pixel', object=_missing(), gripper=_missing(),
                    balanced={key: None for key in METRICS})
    xy = np.asarray(xy)
    validate(points, reference, xy, predicted_visible, policy)
    gt = np.asarray(reference['tracks'], dtype=np.float64)
    prediction = np.asarray(xy, dtype=np.float64)
    err = np.linalg.norm(prediction - gt, axis=-1)
    motion = np.linalg.norm((prediction - prediction[:1]) - (gt - gt[:1]), axis=-1)
    static = np.linalg.norm(gt - gt[:1], axis=-1)
    mask = np.asarray(reference['visibility'])
    result = dict(unit='pixel')
    for entity in ['object', 'gripper']:
        select = np.array([p['entity'] == 'object' if entity == 'object' else p['entity'] in ['left_gripper', 'right_gripper'] for p in points])
        entity_mask = mask & select[None, :]
        frames = []
        for frame in range(1, 17):
            valid = entity_mask[frame]
            if valid.any():
                frames.append(dict(ade_px=float(err[frame, valid].mean()),
                                   motion_ade_px=float(motion[frame, valid].mean()),
                                   static_ade_px=float(static[frame, valid].mean())))
        last = entity_mask[16]
        visible_pairs = entity_mask[1:]
        possible = 16 * int(select.sum())
        result[entity] = dict(
            ade_px=_mean(f['ade_px'] for f in frames),
            fde_px=float(err[16, last].mean()) if last.any() else None,
            motion_ade_px=_mean(f['motion_ade_px'] for f in frames),
            static_ade_px=_mean(f['static_ade_px'] for f in frames),
            static_fde_px=float(static[16, last].mean()) if last.any() else None,
            query_points=int(select.sum()), reference_pairs=int(visible_pairs.sum()), possible_pairs=possible,
            terminal_pairs=int(last.sum()), valid_future_frames=len(frames),
            windows_with_ade=int(bool(frames)), windows_with_fde=int(last.any()),
            reference_fraction=float(visible_pairs.sum()) / possible if possible else None,
            prediction_visible_fraction=float(predicted_visible[1:][visible_pairs].mean()) if visible_pairs.any() else None)
    result['balanced'] = {key: (result['object'][key] + result['gripper'][key]) / 2
                          if result['object'][key] is not None and result['gripper'][key] is not None else None
                          for key in METRICS}
    return result
