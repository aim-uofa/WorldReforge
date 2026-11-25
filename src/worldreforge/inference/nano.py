from __future__ import annotations

import copy
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
    if not checkpoint.is_dir() or not tokenizer.is_file():
        raise FileNotFoundError("Provide a local Nano checkpoint directory and its VAE file")
    config = copy.deepcopy(read_json(checkpoint / "config.json"))
    config["model"]["config"]["resolution"] = "480"
    config["model"]["config"]["tokenizer"]["vae_path"] = str(tokenizer)
    if text_tokenizer is not None:
        text_tokenizer = Path(text_tokenizer).resolve()
        for name in ("vocab.json", "merges.txt"):
            if not (text_tokenizer / name).is_file():
                raise FileNotFoundError(text_tokenizer / name)
        config["model"]["config"]["vlm_config"]["tokenizer"].update(
            pretrained_model_name=str(text_tokenizer), config_variant="hf")
    return config


class NanoRuntime:
    def __init__(self, checkpoint, tokenizer, output, batch_size=1, text_tokenizer=None):
        configure_storage(output)
        require_single_gpu()
        import torch
        from cosmos_framework.inference.common.init import init_script, init_output_dir
        from cosmos_framework.inference.args import OmniSetupOverrides
        import cosmos_framework.inference.action as action
        import cosmos_framework.inference.inference as inference
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
        self.rows, self.saved = {}, {}
        self.input_ids, self.crops, self.live = [], [], 0
        native_input = action.get_action_sample_data
        def prepare(*args, **kwargs):
            data = native_input(*args, **kwargs)
            row = self.rows[str(Path(kwargs["vision_path"]).resolve())]
            tensor = data[kwargs["model_config"].input_video_key][0][0]
            sizes = data["image_size"][0].int().tolist()
            if tuple(tensor.shape) != (3, 17, 544, 736) or sizes != [544, 736, 544, 645]:
                raise ValueError("Native input geometry changed; expected content645/padded736")
            if not torch.equal(tensor[:, :, :, 645:], tensor[:, :, :, 553:644].flip(-1)):
                raise ValueError("Input reflection-padding mismatch")
            pixels = tensor[:, 0].permute(1, 2, 0).cpu().numpy()
            gt = np.load(row["gt_path"], mmap_mode="r", allow_pickle=False)
            if not np.array_equal(pixels[:, :640], gt[0]):
                raise ValueError("Conditioning frame differs from native640 GT")
            actual = data["action"][0][0].detach().cpu()
            expected = torch.tensor(read_json(row["actions_path"]), dtype=actual.dtype)
            if expected.shape != (16, 10) or not torch.isfinite(expected).all():
                raise ValueError("Expected 16 finite normalized 10D state transitions")
            if not torch.equal(actual[:, :10], expected) or torch.count_nonzero(actual[:, 10:]):
                raise ValueError("Upstream did not consume the exact measured-state inputs")
            self.input_ids.append(row["id"])
            return data
        action.get_action_sample_data = prepare
        native_save = inference.save_img_or_video
        def capture(sample, path, *args, **kwargs):
            if tuple(sample.shape) != (3, 17, 544, 640) or not torch.isfinite(sample).all():
                raise ValueError("Output must be native [3,17,544,640], no padding restoration")
            array = (sample.detach().float().clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
            self.saved[str(Path(path).parent.resolve())] = array

            atomic_array(Path(path).parent / "prediction.npy", array)
            native_save(sample, path, *args, **kwargs)
        inference.save_img_or_video = capture
        native_crop = self.pipe.model._remove_padding_from_latent
        if getattr(native_crop, "__func__", None) is not type(self.pipe.model)._remove_padding_from_latent:
            raise ValueError("Upstream native latent crop was replaced")
        def crop(latents, sizes):
            result = native_crop(latents, sizes)
            if not (all(tuple(x.shape[-2:]) == (34, 46) for x in latents) and
                    all(tuple(x.shape[-2:]) == (34, 40) for x in result)):
                raise ValueError("Expected encoded34x46 -> native34x40")
            self.crops.append(len(result))
            return result
        self.pipe.model._remove_padding_from_latent = crop
        original_pack = self.pipe.model._pack_input_sequence
        def packed(*args, **kwargs):
            result = original_pack(*args, **kwargs)
            shapes = [tuple(map(int, x)) for x in result.vision.token_shapes]
            if not all(x == (5, 17, 20) for x in shapes):
                raise ValueError("Unexpected native DiT token grid")
            self.live = max(self.live, len(shapes))
            return result
        self.pipe.model._pack_input_sequence = packed

    def predict(self, rows, data_root, batch_name):
        import torch
        from cosmos_framework.inference.args import OmniSampleOverrides
        self.rows, self.saved = {}, {}
        self.input_ids, self.crops, self.live = [], [], 0
        samples, folders = [], []
        for row in rows:
            condition = checked_file(data_root, row["condition"])
            actions = checked_file(data_root, row["actions"])
            gt = checked_file(data_root, row["gt"])
            self.rows[str(condition.resolve())] = dict(row, actions_path=actions, gt_path=gt)
            folder = self.output / "calls" / batch_name / row["id"]
            folder.mkdir(parents=True, exist_ok=False)
            folders.append(folder)
            spec = dict(action_chunk_size=16, action_path=str(actions), domain_name="droid_lerobot",
                fps=15, num_frames=17, image_size=480, video_save_quality=10,
                view_point="concat_view", model_mode="forward_dynamics", name="TF_" + row["id"],
                prompt="", seed=62100, vision_path=str(condition), output_dir=str(folder),
                num_steps=30, guidance=1., shift=10.)
            samples.append(OmniSampleOverrides.model_validate(spec).build_sample(model_config=self.pipe.model_config))
        if len(self.rows) != len(rows):
            raise ValueError("Each sample needs a distinct conditioning file")
        deterministic_policy()
        torch.cuda.synchronize()
        tick = time.monotonic()
        self.pipe.generate(samples)
        torch.cuda.synchronize()
        if set(self.input_ids) != {r["id"] for r in rows} or not self.crops or self.live != len(rows):
            raise ValueError("Input, crop or true batch validation failed")
        result = [self.saved[str(p.resolve())] for p in folders]
        receipt = {"batch_ids": [r["id"] for r in rows], "batch_size": len(rows),
                   "seconds": time.monotonic() - tick, "seed": 62100, "steps": 30,
                   "guidance": 1., "shift": 10.}
        atomic_json(self.output / "calls" / batch_name / "COMPLETE.json", receipt)
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
            arrays, receipt = runtime.predict(batch, data_root, f"batch{index:05d}")
            if len(arrays) != len(batch):
                raise ValueError("Prediction count differs from batch size")
            for row in batch:
                folder = output / "predictions" / row["id"]
                folder.mkdir(parents=True, exist_ok=False)
                source = output / "runtime/calls" / f"batch{index:05d}" / row["id"] / "prediction.npy"
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
