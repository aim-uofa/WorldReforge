import copy
import unittest
import numpy as np

from worldreforge.evaluation.motion import score_canvas_pixel
from worldreforge.evaluation.pixels import aggregate_videos


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.points = [dict(id="obj", camera="wrist_cam", entity="object", query_xy=[50., 50.]),
                       dict(id="finger", camera="over_shoulder_left_camera", entity="left_gripper", query_xy=[60., 400.])]
        gt = np.repeat(np.asarray([[50., 50.], [60., 400.]])[None], 17, axis=0)
        gt[:, :, 0] += np.arange(17)[:, None]
        self.ref = dict(tracks=gt, visibility=np.ones((17, 2), bool),
                        visibility_state=np.ones((17, 2), np.int8))
        self.xy = gt.copy()
        self.xy[1:, 0, 0] += 3
        self.xy[1:, 0, 1] += 4
        self.xy[1:, 1, 0] += 2
        self.vis = np.ones((17, 2), bool)

    def test_pixels_without_diagonal_normalization(self):
        result = score_canvas_pixel(self.points, self.ref, self.xy, self.vis)
        self.assertEqual(result["object"]["ade_px"], 5.)
        self.assertEqual(result["object"]["fde_px"], 5.)
        self.assertEqual(result["gripper"]["ade_px"], 2.)
        self.assertEqual(result["balanced"]["ade_px"], 3.5)

    def test_predicted_invisibility_does_not_hide_errors(self):
        self.vis[:] = False
        result = score_canvas_pixel(self.points, self.ref, self.xy, self.vis)
        self.assertEqual(result["object"]["ade_px"], 5.)
        self.assertEqual(result["object"]["prediction_visible_fraction"], 0.)

    def test_occluded_final_frame_is_missing_not_last_visible(self):
        self.ref["visibility"][16, 0] = False
        self.ref["visibility_state"][16, 0] = 2
        self.xy[16, 0] = [-1000, -1000]
        result = score_canvas_pixel(self.points, self.ref, self.xy, self.vis)
        self.assertEqual(result["object"]["ade_px"], 5.)
        self.assertIsNone(result["object"]["fde_px"])
        self.assertEqual(result["object"]["valid_future_frames"], 15)

    def test_initial_occlusion_and_nonfinite_predictions_rejected(self):
        ref = copy.deepcopy(self.ref)
        ref["visibility"][0, 0] = False
        ref["visibility_state"][0, 0] = 2
        with self.assertRaises(ValueError):
            score_canvas_pixel(self.points, ref, self.xy, self.vis)
        self.xy[5, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            score_canvas_pixel(self.points, self.ref, self.xy, self.vis)

    def test_missing_real_gt_is_not_zero(self):
        value = score_canvas_pixel(None, None, None, None)
        self.assertIsNone(value["object"]["ade_px"])
        self.assertEqual(value["object"]["windows_with_ade"], 0)

    def test_video_average_and_denominators(self):
        motion = score_canvas_pixel(self.points, self.ref, self.xy, self.vis)
        missing = score_canvas_pixel(None, None, None, None)
        rows = [dict(pixels={"psnr_db": p}, motion=m) for p, m in [(20, motion), (20, motion), (29, missing)]]
        value = aggregate_videos(rows)["metrics"]
        self.assertEqual(value["psnr_db"]["mean"], 23.)
        self.assertEqual(value["object_ade_px"]["valid_videos"], 2)


if __name__ == "__main__":
    unittest.main()
