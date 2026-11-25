from __future__ import annotations

import os

from worldreforge.io import output_path


def gpu_device(index, environment=None):

    environment = os.environ if environment is None else environment
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("--gpu must be one nonnegative integer")
    if any(int(environment.get(key, "1")) != 1 for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")):
        raise ValueError("Use a single process, without a distributed launcher")
    if "CUDA_VISIBLE_DEVICES" not in environment:
        return str(index)
    visible = [value.strip() for value in environment["CUDA_VISIBLE_DEVICES"].split(",")]
    if index >= len(visible) or not visible[index] or visible[index] == "-1":
        raise ValueError("Requested GPU is outside CUDA_VISIBLE_DEVICES")
    return visible[index]


def bind_gpu(index):
    device = gpu_device(index)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    return device


def require_single_gpu():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")


def configure_storage(output):
    output = output_path(output)
    for key, suffix in {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "HF_HOME": "cache/huggingface",
        "TORCH_HOME": "cache/torch", "XDG_CACHE_HOME": "cache/xdg",
        "TORCHINDUCTOR_CACHE_DIR": "cache/inductor", "TRITON_CACHE_DIR": "cache/triton",
        "CUDA_CACHE_PATH": "cache/cuda", "TORCH_EXTENSIONS_DIR": "cache/extensions",
        "PYTORCH_KERNEL_CACHE_PATH": "cache/kernels",
    }.items():
        path = output / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TORCHINDUCTOR_DETERMINISTIC"] = "1"
