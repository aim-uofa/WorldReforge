import copy
import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from worldreforge.evaluation.motion import score_canvas_pixel
from worldreforge.evaluation.pixels import aggregate_videos
from worldreforge.evaluation.pipeline import evaluate, run_pipeline
from worldreforge.evaluation.predictions import load_prediction
from worldreforge.evaluation.runner import score_one
from worldreforge.evaluation.track_runner import track
from worldreforge.cli import build_parser
from worldreforge.runtime import gpu_device
from worldreforge.evaluation.validation import validate_results
from worldreforge.io import atomic_array, atomic_json, read_json


class ResultAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metrics = self.root / "metrics"
        self.rows = [dict(id="sim_case", domain="sim", panel="XY", asset="fixture",
                         gt={"path": "sim/gt.npy"}, points={"path": "sim/points.json"},
                         reference={"path": "sim/reference.npz"}),
                     dict(id="real_case", domain="droid", panel="droid_failure", asset="fixture",
                         gt={"path": "real/gt.npy"}, points=None, reference=None)]
        points = [dict(id="object", entity="object", camera="wrist_cam", query_xy=[20., 20.]),
                  dict(id="finger", entity="left_gripper", camera="wrist_cam", query_xy=[40., 40.])]
        gt = np.repeat(np.asarray([p["query_xy"] for p in points])[None], 17, axis=0)
        reference = dict(tracks=gt, visibility=np.ones((17, 2), bool), visibility_state=np.ones((17, 2), np.int8))
        pred = gt.copy()
        pred[1:] += 1
        sim_motion = score_canvas_pixel(points, reference, pred, np.ones((17, 2), bool))
        no_motion = score_canvas_pixel(None, None, None, None)
        self.results = []
        for row, motion in zip(self.rows, (sim_motion, no_motion)):
            pixels = dict(mse=.01, ssim=.8, psnr_db=20., static_mse=.04,
                          static_psnr_db=-10 * math.log10(.04), temporal_difference_mse=.003,
                          roi_mse=None, roi_ssim=None, static_roi_mse=None, roi_psnr_db=None, frames_scored=16)
            self.results.append(dict(id=row["id"], domain=row["domain"], panel=row["panel"], asset=row["asset"],
                pixels=pixels, motion=motion,
                motion_status="scored" if row["points"] else "no_qualified_reference"))
        self.write_metrics()

    def write_metrics(self):
        for row in self.results:
            atomic_json(self.metrics / "cases" / f"{row['id']}.json", row)
        grouped = {"all": self.results}
        for key in ("domain", "panel"):
            for value in {r[key] for r in self.results}:
                grouped[f"{key}:{value}"] = [r for r in self.results if r[key] == value]
        atomic_json(self.metrics / "summary.json", {k: aggregate_videos(v) for k, v in grouped.items()})
        atomic_json(self.metrics / "COMPLETE.json", dict(status="passed", windows=2, pixels_only=False,
                                                         ids=[r["id"] for r in self.rows]))

    def validate(self):
        return validate_results(self.rows, self.metrics, self.root / "validation.json")

    def test_accepts_complete_results_without_claiming_improvement(self):
        report = self.validate()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["motion_reference_windows"], 1)
        self.assertEqual(report["scientific_status"], "not_inferred_by_integrity_checks")

    def test_missing_window_fails(self):
        (self.metrics / "cases/sim_case.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or extra"):
            self.validate()
        self.assertEqual(read_json(self.root / "validation.json")["status"], "failed")

    def test_fabricated_summary_mean_fails(self):
        value = read_json(self.metrics / "summary.json")
        value["all"]["metrics"]["psnr_db"]["mean"] += .1
        atomic_json(self.metrics / "summary.json", value)
        with self.assertRaisesRegex(ValueError, "aggregate differs"):
            self.validate()

    def test_nonfinite_metric_fails(self):
        value = copy.deepcopy(self.results[0])
        value["pixels"]["ssim"] = float("nan")
        (self.metrics / "cases/sim_case.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.validate()

    def test_wrong_psnr_transform_fails_even_with_consistent_summary(self):
        self.results[0]["pixels"]["psnr_db"] += 1
        self.write_metrics()
        with self.assertRaisesRegex(ValueError, "MSE|PSNR"):
            self.validate()

    def test_missing_real_motion_cannot_be_filled_with_zero(self):
        self.results[1]["motion"]["object"]["ade_px"] = 0.
        self.write_metrics()
        with self.assertRaisesRegex(ValueError, "stay null"):
            self.validate()

    def test_occluded_terminal_frame_may_have_no_fde(self):
        entity = self.results[0]["motion"]["object"]
        entity.update(terminal_pairs=0, valid_future_frames=15, reference_pairs=15, reference_fraction=15/16,
                      windows_with_fde=0, fde_px=None, static_fde_px=None)
        self.results[0]["motion"]["balanced"].update(fde_px=None, static_fde_px=None)
        self.write_metrics()
        self.assertEqual(self.validate()["status"], "passed")


class PipelineControlTests(unittest.TestCase):
    def test_combined_command_accepts_model_and_tracking_options(self):
        args = build_parser().parse_args(["run", "--manifest", "manifest.jsonl", "--data-root", "data",
            "--checkpoint", "model", "--tokenizer", "vae", "--tracker-repository", "tracker",
            "--tracker-checkpoint", "weights", "--output", "run"])
        self.assertEqual((args.gpu, args.batch_size, args.cpu_workers), (0, 1, 8))
        self.assertFalse(args.pixels_only)

    def test_gpu_indices_respect_visible_devices(self):
        self.assertEqual(gpu_device(1, {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}), "GPU-b")
        self.assertEqual(gpu_device(3, {}), "3")
        for value in (-1, 2, "0,1", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gpu_device(value, {"CUDA_VISIBLE_DEVICES": "0,1"})
        for visible in ("", "-1"):
            with self.subTest(visible=visible), self.assertRaises(ValueError):
                gpu_device(0, {"CUDA_VISIBLE_DEVICES": visible})

    def test_distributed_launch_is_rejected(self):
        for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "single process"):
                gpu_device(0, {key: "8"})

    def test_cli_separates_generation_from_evaluation(self):
        parser = build_parser()
        common = ["--manifest", "manifest.jsonl", "--data-root", "data", "--output", "run"]
        infer = parser.parse_args(["infer", *common, "--checkpoint", "model", "--tokenizer", "vae"])
        self.assertEqual((infer.gpu, infer.batch_size), (0, 1))
        with contextlib.redirect_stderr(io.StringIO()):
            for command in (["eval", *common],
                            ["eval", *common, "--prediction-root", "predictions", "--checkpoint", "model"],
                            ["infer", *common, "--checkpoint", "model", "--tokenizer", "vae", "--workers", "8"]):
                with self.subTest(command=command), self.assertRaises(SystemExit):
                    parser.parse_args(command)
        evaluate_args = parser.parse_args(["eval", *common, "--prediction-root", "predictions"])
        self.assertFalse(hasattr(evaluate_args, "checkpoint"))

    def test_failed_stage_never_creates_pipeline_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, predictions = root / "data", root / "predictions"
            data.mkdir()
            predictions.mkdir()
            (data / "gt.npy").touch()
            ref = lambda name: dict(path=name)
            row = dict(id="fixture", domain="real10", panel="fixture", asset="fixture", frames=17,
                score_hw=[544, 640], conditioning="measured_robot_state_endpoint_v1", normalized_actions=True,
                condition=ref("condition.png"), actions=ref("actions.json"), gt=ref("gt.npy"), points=None, reference=None)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            with patch("worldreforge.evaluation.runner.score", side_effect=ValueError("fixture scoring failure")), \
                 patch("worldreforge.evaluation.pipeline.run_tracking") as gpu:
                with self.assertRaisesRegex(ValueError, "fixture scoring failure"):
                    evaluate(manifest=manifest, data_root=data, prediction_root=predictions,
                             output=root / "run", cpu_workers=1)
                gpu.assert_not_called()
            self.assertFalse((root / "run/COMPLETE.json").exists())
            self.assertEqual(read_json(root / "run/FAILED.json")["stage"], "scoring")


class CombinedWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        data = self.root / "data"
        self.video = np.full((17, 544, 640, 3), 64, np.uint8)
        atomic_array(data / "gt.npy", self.video)
        (data / "condition.png").touch()
        atomic_json(data / "actions.json", [])
        atomic_json(self.root / "checkpoint/config.json", {})
        (self.root / "vae.pth").touch()
        row = dict(id="sample", domain="real10", panel="fixture", asset="fixture", frames=17,
            score_hw=[544, 640], conditioning="measured_robot_state_endpoint_v1", normalized_actions=True,
            condition={"path": "condition.png"}, actions={"path": "actions.json"}, gt={"path": "gt.npy"},
            points=None, reference=None)
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        self.options = dict(manifest=manifest, data_root=data, checkpoint=self.root / "checkpoint",
                            tokenizer=self.root / "vae.pth", output=self.root / "run", cpu_workers=1)

    def generate(self, command, *, check):
        self.assertTrue(check)
        self.assertEqual(command[1:4], ["-m", "worldreforge", "infer"])
        output = Path(command[command.index("--output") + 1])
        atomic_array(output / "predictions/sample/prediction.npy", self.video + 1)
        atomic_json(output / "predictions/sample/COMPLETE.json", dict(id="sample"))
        atomic_json(output / "COMPLETE.json", dict(status="passed", windows=1, ids=["sample"]))

    def test_new_predictions_are_scored_and_validated_in_one_run(self):
        for name in ("vocab.json", "merges.txt"):
            (self.root / name).touch()
        self.options.update(gpu=1, text_tokenizer=self.root)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"}), \
             patch("worldreforge.evaluation.pipeline.subprocess.run", side_effect=self.generate) as infer:
            report = run_pipeline(**self.options)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-a,GPU-b")
        infer.assert_called_once()
        command = infer.call_args.args[0]
        self.assertEqual(command[command.index("--gpu") + 1], "1")
        self.assertEqual(command[command.index("--text-tokenizer") + 1], str(self.root))
        output = self.options["output"]
        self.assertEqual(read_json(output / "PLAN.json")["device"], "GPU-b")
        self.assertEqual(read_json(output / "COMPLETE.json")["windows"], 1)
        self.assertEqual(read_json(output / "evaluation/checks/results.json")["status"], "passed")
        self.assertFalse((output / "FAILED.json").exists())
        case = read_json(output / "evaluation/metrics/cases/sample.json")
        self.assertAlmostEqual(case["pixels"]["psnr_db"], 20 * math.log10(255), places=3)
        self.assertEqual(report["completion"]["status"], "passed")

    def test_inference_failure_stops_evaluation(self):
        with patch("worldreforge.evaluation.pipeline.subprocess.run",
                   side_effect=subprocess.CalledProcessError(1, "infer")), \
             patch("worldreforge.evaluation.pipeline.evaluate") as evaluate_stage:
            with self.assertRaises(subprocess.CalledProcessError):
                run_pipeline(**self.options)
            evaluate_stage.assert_not_called()
        self.assertFalse((self.options["output"] / "COMPLETE.json").exists())
        self.assertEqual(read_json(self.options["output"] / "FAILED.json")["stage"], "inference")

    def test_successful_exit_without_matching_predictions_stops_evaluation(self):
        def incomplete(command, *, check):
            self.generate(command, check=check)
            atomic_json(self.options["output"] / "inference/COMPLETE.json",
                        dict(status="passed", windows=1, ids=["wrong_sample"]))
        with patch("worldreforge.evaluation.pipeline.subprocess.run", side_effect=incomplete), \
             patch("worldreforge.evaluation.pipeline.evaluate") as evaluate_stage:
            with self.assertRaisesRegex(ValueError, "coverage"):
                run_pipeline(**self.options)
            evaluate_stage.assert_not_called()
        self.assertFalse((self.options["output"] / "COMPLETE.json").exists())

    def test_scoring_failure_prevents_overall_completion(self):
        with patch("worldreforge.evaluation.pipeline.subprocess.run", side_effect=self.generate), \
             patch("worldreforge.evaluation.runner.score", side_effect=ValueError("invalid prediction")):
            with self.assertRaisesRegex(ValueError, "invalid prediction"):
                run_pipeline(**self.options)
        output = self.options["output"]
        self.assertFalse((output / "COMPLETE.json").exists())
        self.assertEqual(read_json(output / "FAILED.json")["stage"], "evaluation")
        self.assertEqual(read_json(output / "evaluation/FAILED.json")["stage"], "scoring")

    def test_missing_tracker_is_caught_before_inference(self):
        row = json.loads(self.options["manifest"].read_text())
        for field in ("points", "reference"):
            row[field] = {"path": field}
            (self.options["data_root"] / field).touch()
        self.options["manifest"].write_text(json.dumps(row) + "\n")
        with patch("worldreforge.evaluation.pipeline.subprocess.run") as infer:
            with self.assertRaisesRegex(ValueError, "tracker-repository"):
                run_pipeline(**self.options)
            infer.assert_not_called()
        self.assertFalse(self.options["output"].exists())

    def test_existing_run_is_preserved(self):
        self.options["output"].mkdir()
        sentinel = self.options["output"] / "saved.txt"
        sentinel.write_text("preserve")
        with patch("worldreforge.evaluation.pipeline.subprocess.run") as infer:
            with self.assertRaises(FileExistsError):
                run_pipeline(**self.options)
            infer.assert_not_called()
        self.assertEqual(sentinel.read_text(), "preserve")


class TrackingPipelineTests(unittest.TestCase):
    def test_tracking_and_scoring_preserve_sample_and_point_correspondence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, predictions, tracking = root / "data", root / "predictions", root / "tracking"
            points = [dict(id="object", entity="object", camera="wrist_cam", query_xy=[20., 20.]),
                      dict(id="finger", entity="left_gripper", camera="wrist_cam", query_xy=[40., 40.])]
            xy = np.repeat(np.asarray([p["query_xy"] for p in points])[None], 17, axis=0)
            visible = np.ones((17, 2), bool)
            video = np.zeros((17, 544, 640, 3), np.uint8)
            atomic_array(data / "gt.npy", video)
            atomic_json(data / "points.json", {"points": points})
            np.savez_compressed(data / "reference.npz", tracks=xy, visibility=visible,
                                visibility_state=np.ones((17, 2), np.int8))
            atomic_array(predictions / "sample/prediction.npy", video)
            atomic_json(predictions / "sample/COMPLETE.json", {"id": "sample"})
            row = dict(id="sample", domain="sim", panel="XY", asset="fixture", gt={"path": "gt.npy"},
                       points={"path": "points.json"}, reference={"path": "reference.npz"})
            with patch("worldreforge.evaluation.track_runner.bind_gpu", return_value="0"), \
                 patch("worldreforge.evaluation.track_runner.configure_storage"), \
                 patch("worldreforge.evaluation.tracking.load_tracker") as loader, \
                 patch("worldreforge.evaluation.tracking.track_video", return_value=(xy, visible)):
                self.assertEqual(track([row], data, predictions, root / "tracker", root / "weights", tracking)["windows"], 1)
                loader.assert_called_once()
            task = (row, data, predictions, tracking, root / "metrics", False)
            result = score_one(task)
            self.assertEqual(result["motion"]["object"]["ade_px"], 0.)
            self.assertEqual(result["pixels"]["mse"], 0.)

            atomic_json(predictions / "sample/COMPLETE.json", {"id": "wrong"})
            with self.assertRaisesRegex(ValueError, "Prediction sample ID"):
                load_prediction(predictions, row)
            atomic_json(predictions / "sample/COMPLETE.json", {"id": "sample"})
            atomic_json(tracking / "sample/COMPLETE.json", {"id": "wrong"})
            with self.assertRaisesRegex(ValueError, "Tracking sample ID"):
                score_one(task)
            atomic_json(tracking / "sample/COMPLETE.json", {"id": "sample"})
            np.savez_compressed(tracking / "sample/tracks.npz", tracks=xy, visibility=visible,
                                point_ids=np.asarray(["finger", "object"]))
            with self.assertRaisesRegex(ValueError, "point order"):
                score_one(task)


if __name__ == "__main__":
    unittest.main()
