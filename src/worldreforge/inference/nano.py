from __future__ import annotations

from contextlib import contextmanager
import functools
import os
from pathlib import Path
import time

import numpy as np

from worldreforge.io import atomic_array, atomic_json, checked_file, read_json, output_path
from worldreforge.runtime import bind_gpu, configure_storage, require_single_gpu


def deterministic_policy():
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.use_deterministic_algorithms(True)


def install_inference_policy(output):
    import torch
    import torch._inductor
    import torch._inductor.config as config
    from cosmos_framework.inference.model import Cosmos3OmniModel

    if getattr(torch.compile, "_worldreforge_policy", False):
        raise RuntimeError("Use one runtime per process")
    original = torch.compile
    config.deterministic = True

    @functools.wraps(original)
    def compiled(*args, **kwargs):
        if kwargs.get("backend", "inductor") != "inductor":
            raise ValueError("Unexpected inference compilation backend")
        mode = kwargs.pop("mode", None)
        options = dict(torch._inductor.list_mode_options(mode or "default", kwargs.get("dynamic")))
        options.update(kwargs.get("options") or {})
        options["deterministic"] = True
        kwargs["options"] = options
        return original(*args, **kwargs)

    compiled._worldreforge_policy = True
    torch.compile = compiled
    before = Cosmos3OmniModel.before_load_model.__func__
    after = Cosmos3OmniModel.after_load_model.__func__

    def before_load(cls):
        before(cls)
        deterministic_policy()

    def after_load(cls, model):
        after(cls, model)
        deterministic_policy()

    Cosmos3OmniModel.before_load_model = classmethod(before_load)
    Cosmos3OmniModel.after_load_model = classmethod(after_load)
    deterministic_policy()
    atomic_json(output / "policy.json", {"eager_model": True, "cuda_graphs": False,
                "deterministic_algorithms": True, "allow_tf32": False})


def load_model_config(checkpoint, tokenizer, text_tokenizer=None):
    checkpoint, tokenizer = Path(checkpoint).resolve(), Path(tokenizer).resolve()
    config = read_json(checkpoint / "config.json")
    if not tokenizer.is_file():
        raise FileNotFoundError(tokenizer)
    model = config["model"]["config"]
    model["resolution"] = "480"
    model["tokenizer"]["vae_path"] = str(tokenizer)
    if text_tokenizer is not None:
        text_tokenizer = Path(text_tokenizer).resolve()
        for name in ("vocab.json", "merges.txt"):
            if not (text_tokenizer / name).is_file():
                raise FileNotFoundError(text_tokenizer / name)
        model["vlm_config"]["tokenizer"].update(
            pretrained_model_name=str(text_tokenizer), config_variant="hf")
    return config


def check_conditioning(data, video_key, gt_path, actions_path):
    import torch

    video = data[video_key][0][0]
    size = data["image_size"][0].int().tolist()
    if video.shape != (3, 17, 544, 736) or size != [544, 736, 544, 645]:
        raise ValueError("Expected a 544 x 645 image padded to width 736")
    if not torch.equal(video[..., 645:], video[..., 553:644].flip(-1)):
        raise ValueError("Expected reflection padding on the right")

    first_frame = video[:, 0, :, :640].permute(1, 2, 0).cpu().numpy()
    gt = np.load(gt_path, mmap_mode="r", allow_pickle=False)
    if not np.array_equal(first_frame, gt[0]):
        raise ValueError(f"Conditioning frame differs from {gt_path}")

    actions = data["action"][0][0].detach().cpu()
    expected = torch.tensor(read_json(actions_path), dtype=actions.dtype)
    if (expected.shape != (16, 10) or not torch.isfinite(expected).all()
            or not torch.equal(actions[:, :10], expected) or torch.count_nonzero(actions[:, 10:])):
        raise ValueError(f"Expected the 16 measured 10D state transitions from {actions_path}")


@contextmanager
def prediction_io(inputs):
    import torch
    import cosmos_framework.inference.action as action
    import cosmos_framework.inference.inference as inference

    read_input = action.get_action_sample_data
    save_video = inference.save_img_or_video
    observed = set()

    def prepare(*args, **kwargs):
        data = read_input(*args, **kwargs)
        path = Path(kwargs["vision_path"]).resolve()
        gt_path, actions_path = inputs[path]
        check_conditioning(data, kwargs["model_config"].input_video_key, gt_path, actions_path)
        observed.add(path)
        return data

    def capture(sample, path, *args, **kwargs):
        if sample.shape != (3, 17, 544, 640) or not torch.isfinite(sample).all():
            raise ValueError("Expected a finite native [3,17,544,640] prediction")
        frames = sample.detach().float().clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy()
        atomic_array(Path(path).parent / "prediction.npy", (frames * 255).astype(np.uint8))
        return save_video(sample, path, *args, **kwargs)

    action.get_action_sample_data = prepare
    inference.save_img_or_video = capture
    try:
        yield
        if observed != inputs.keys():
            raise ValueError("Cosmos did not read every conditioning input in the batch")
    finally:
        action.get_action_sample_data = read_input
        inference.save_img_or_video = save_video


class NanoRuntime:
    def __init__(self, checkpoint, tokenizer, output, batch_size=1, text_tokenizer=None):
        configure_storage(output)
        require_single_gpu()
        from cosmos_framework.inference.common.init import init_script, init_output_dir
        from cosmos_framework.inference.args import OmniSetupOverrides

        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        init_script()
        install_inference_policy(self.output)
        checkpoint = Path(checkpoint).resolve()
        config = load_model_config(checkpoint, tokenizer, text_tokenizer)
        if text_tokenizer is not None:
            os.environ["COSMOS3_LOCAL_TOKENIZER_DIR"] = str(Path(text_tokenizer).resolve())
        atomic_json(self.output / "resolved_model_config.json", config)
        setup = OmniSetupOverrides(checkpoint_path=str(checkpoint),
            config_file=str(self.output / "resolved_model_config.json"),
            output_dir=str(self.output / "pipeline"), parallelism_preset="latency",
            dp_shard_size=1, dp_replicate_size=1, cp_size=1, cfgp_size=1,
            max_num_seqs=batch_size, guardrails=False, keep_going=False,
            skip_invalid_samples=False, use_torch_compile=False, use_cuda_graphs=False).build_setup()
        deterministic_policy()
        init_output_dir(setup.output_dir)
        self.pipe = setup.get_inference_cls().create(setup)
        deterministic_policy()

    def predict(self, rows, data_root, batch_name):
        import torch
        from cosmos_framework.inference.args import OmniSampleOverrides

        inputs = {}
        samples = []
        batch_dir = self.output / "calls" / batch_name
        for row in rows:
            condition = checked_file(data_root, row["condition"])
            actions = checked_file(data_root, row["actions"])
            gt = checked_file(data_root, row["gt"])
            inputs[condition] = (gt, actions)
            folder = batch_dir / row["id"]
            folder.mkdir(parents=True, exist_ok=False)
            spec = OmniSampleOverrides.model_validate({
                "name": "TF_" + row["id"], "prompt": "", "output_dir": str(folder),
                "vision_path": str(condition), "action_path": str(actions),
                "domain_name": "droid_lerobot", "model_mode": "forward_dynamics",
                "view_point": "concat_view", "image_size": 480, "video_save_quality": 10,
                "action_chunk_size": 16, "num_frames": 17, "fps": 15,
                "seed": 62100, "num_steps": 30, "guidance": 1., "shift": 10.,
            })
            samples.append(spec.build_sample(model_config=self.pipe.model_config))
        if len(inputs) != len(rows):
            raise ValueError("Each sample needs a distinct conditioning file")
        deterministic_policy()
        torch.cuda.synchronize()
        tick = time.monotonic()
        with prediction_io(inputs):
            self.pipe.generate(samples)
        torch.cuda.synchronize()
        result = [np.load(batch_dir / row["id"] / "prediction.npy", mmap_mode="r", allow_pickle=False)
                  for row in rows]
        receipt = {"batch_ids": [r["id"] for r in rows], "batch_size": len(rows),
                   "seconds": time.monotonic() - tick, "seed": 62100, "steps": 30,
                   "guidance": 1., "shift": 10.}
        atomic_json(batch_dir / "COMPLETE.json", receipt)
        return result, receipt


def infer(rows, data_root, checkpoint, tokenizer, output, batch_size=1, gpu=0, text_tokenizer=None):
    if batch_size < 1 or not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Positive batch size and nonempty unique members are required")
    device = bind_gpu(gpu)
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    stage = "initialization"
    try:
        batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
        atomic_json(output / "PLAN.json", dict(device=device, batch_size=batch_size,
            batches=[[r["id"] for r in batch] for batch in batches],
            checkpoint=str(Path(checkpoint).resolve()),
            protocol="worldreforge_single_gpu_v1"))
        runtime = NanoRuntime(checkpoint, tokenizer, output / "runtime", batch_size, text_tokenizer=text_tokenizer)
        completed = []
        stage = "inference"
        for index, batch in enumerate(batches):
            batch_name = f"batch{index:05d}"
            _, receipt = runtime.predict(batch, data_root, batch_name)
            for row in batch:
                folder = output / "predictions" / row["id"]
                folder.mkdir(parents=True, exist_ok=False)
                source = output / "runtime/calls" / batch_name / row["id"] / "prediction.npy"
                os.link(source, folder / "prediction.npy")
                atomic_json(folder / "COMPLETE.json", dict(receipt, id=row["id"]))
                completed.append(row["id"])
        result = dict(status="passed", windows=len(completed), ids=completed, device=device)
        atomic_json(output / "COMPLETE.json", result)
        return result
    except BaseException as error:
        atomic_json(output / "FAILED.json", dict(status="failed", stage=stage,
                    error_type=type(error).__name__, error=str(error)))
        raise
