import math
import unittest
import numpy as np

from worldreforge.evaluation.pixels import score_pixels


class PixelTests(unittest.TestCase):
    def test_video_mse_then_psnr_condition_excluded(self):
        gt = np.zeros((17, 544, 640, 3), np.uint8)
        prediction = gt.copy()
        prediction[0] = 255
        prediction[1:9] = 10
        prediction[9:] = 20
        result = score_pixels(prediction, gt)
        mse = (10 ** 2 + 20 ** 2) / 2 / 255 ** 2
        self.assertAlmostEqual(result["mse"], mse, places=8)
        self.assertAlmostEqual(result["psnr_db"], -10 * math.log10(mse), places=5)
        self.assertIsNone(result["roi_mse"])
        self.assertEqual(result["frames_scored"], 16)

    def test_padding_cannot_be_silently_scored(self):
        gt = np.zeros((17, 544, 640, 3), np.uint8)
        pred = np.zeros((17, 544, 645, 3), np.uint8)
        with self.assertRaises(ValueError):
            score_pixels(pred, gt)


if __name__ == "__main__":
    unittest.main()
