import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from worldreforge.cli import build_parser
from worldreforge.inference.nano import infer, load_model_config
from worldreforge.io import atomic_array, atomic_json, read_json
from worldreforge.runtime import bind_gpu, require_single_gpu


class SingleGpuTests(unittest.TestCase):
    def test_original_entrypoint_uses_base_checkpoint_and_accepts_local_override(self):
        script = Path(__file__).resolve().parents[1] / "scripts/infer_original.sh"
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture"
            capture.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            capture.chmod(0o700)
            for extra, expected in (([], "models/Cosmos3-Nano"),
                                    (["--checkpoint", "local original"], "local original")):
                with self.subTest(checkpoint=expected):
                    result = subprocess.run(["bash", str(script), "--manifest", "eval.jsonl",
                        "--data-root", "data", "--output", "original output", "--tokenizer", "vae.pth",
                        "--gpu", "1", "--batch-size", "2", *extra],
                        env=dict(os.environ, PYTHON=str(capture)), text=True,
                        capture_output=True, check=True)
                    argv = result.stdout.splitlines()
                    self.assertEqual(argv[:2], ["-m", "worldreforge"])
                    args = build_parser().parse_args(argv[2:])
                    self.assertEqual(args.command, "infer")
                    self.assertEqual(args.checkpoint, Path(expected))
                    self.assertEqual((args.gpu, args.batch_size), (1, 2))
                    self.assertEqual(args.output, Path("original output"))

    def test_base_and_finetuned_configs_preserve_model_identity_and_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vae, text_tokenizer = root / "vae.pth", root / "text"
            vae.touch()
            text_tokenizer.mkdir()
            for name in ("vocab.json", "merges.txt"):
                (text_tokenizer / name).touch()
            for identity in ("pretrained", "worldreforge"):
                with self.subTest(identity=identity):
                    checkpoint = root / identity
                    original = dict(architectures=["Cosmos3ForConditionalGeneration"],
                        model=dict(config=dict(resolution="720", identity=identity,
                            tokenizer=dict(vae_path="upstream-vae", spatial_compression_factor=16),
                            vlm_config=dict(tokenizer=dict(pretrained_model_name="upstream-text",
                                                           config_variant="hf")))))
                    path = checkpoint / "config.json"
                    atomic_json(path, original)
                    before = path.read_bytes()
                    resolved = load_model_config(checkpoint, vae, text_tokenizer)
                    self.assertEqual(path.read_bytes(), before)
                    self.assertEqual(resolved["architectures"], original["architectures"])
                    self.assertEqual(resolved["model"]["config"]["identity"], identity)
                    model = resolved["model"]["config"]
                    self.assertEqual(model["resolution"], "480")
                    self.assertEqual(model["tokenizer"], dict(vae_path=str(vae), spatial_compression_factor=16))
                    self.assertEqual(model["vlm_config"]["tokenizer"]["pretrained_model_name"], str(text_tokenizer))
                    model["resolution"] = original["model"]["config"]["resolution"]
                    model["tokenizer"]["vae_path"] = "upstream-vae"
                    model["vlm_config"]["tokenizer"]["pretrained_model_name"] = "upstream-text"
                    self.assertEqual(resolved, original)

    def test_bind_exposes_only_requested_gpu(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"}):
            self.assertEqual(bind_gpu(1), "GPU-b")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-b")

    def test_runtime_rejects_missing_or_multiple_gpus(self):
        for count in (0, 1, 2):
            cuda = SimpleNamespace(is_available=lambda: count > 0, device_count=lambda: count)
            with self.subTest(count=count), patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
                if count == 1:
                    require_single_gpu()
                else:
                    with self.assertRaisesRegex(RuntimeError, "Exactly one"):
                        require_single_gpu()

    def test_one_model_handles_every_batch_and_saves_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, output = root / "checkpoint", root / "inference"
            atomic_json(checkpoint / "config.json", {})
            rows = [dict(id=str(i), condition={"path": f"{i}/condition.png"},
                         actions={"path": f"{i}/actions.json"}) for i in range(3)]

            def predict(batch, data_root, batch_name):
                arrays = [np.zeros((1, 1, 1, 3), np.uint8) for _ in batch]
                for row, array in zip(batch, arrays):
                    atomic_array(output / "runtime/calls" / batch_name / row["id"] / "prediction.npy", array)
                return arrays, dict(batch_ids=[r["id"] for r in batch], batch_size=len(batch))

            model = Mock()
            model.predict.side_effect = predict
            with patch("worldreforge.inference.nano.bind_gpu", return_value="GPU-b"), \
                 patch("worldreforge.inference.nano.NanoRuntime", return_value=model) as factory:
                receipt = infer(rows, root, checkpoint, root / "tokenizer", output, batch_size=2, gpu=1,
                                text_tokenizer=root / "text_tokenizer")
            factory.assert_called_once_with(checkpoint, root / "tokenizer", output / "runtime", 2,
                                            text_tokenizer=root / "text_tokenizer")
            self.assertEqual(model.predict.call_count, 2)
            self.assertEqual(read_json(output / "PLAN.json")["batches"], [["0", "1"], ["2"]])
            self.assertEqual(receipt["ids"], ["0", "1", "2"])
            self.assertEqual(read_json(output / "COMPLETE.json"), receipt)
            self.assertFalse((output / "workers").exists())
            for row in rows:
                self.assertTrue((output / "predictions" / row["id"] / "COMPLETE.json").is_file())

    def test_prediction_failure_has_no_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, output = root / "checkpoint", root / "inference"
            atomic_json(checkpoint / "config.json", {})
            model = Mock()
            model.predict.side_effect = RuntimeError("test prediction failure")
            with patch("worldreforge.inference.nano.bind_gpu", return_value="0"), \
                 patch("worldreforge.inference.nano.NanoRuntime", return_value=model), \
                 self.assertRaisesRegex(RuntimeError, "test prediction failure"):
                infer([{"id": "sample"}], root, checkpoint, root / "tokenizer", output)
            self.assertFalse((output / "COMPLETE.json").exists())
            self.assertEqual(read_json(output / "FAILED.json")["stage"], "inference")


if __name__ == "__main__":
    unittest.main()
