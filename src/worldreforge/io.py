from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contained(root, relative):
    root = Path(root).resolve()
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Expected a contained relative path: {relative}")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes its declared root: {relative}")
    return path


def output_path(path):

    path = Path(path).resolve()
    env = "COSIM_RESULTS_ROOT"
    if os.environ.get(env) and not path.is_relative_to(Path(os.environ[env]).resolve()):
        raise ValueError(f"Output must be within {env}")
    return path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".partial-",
                                     encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def atomic_array(path, array):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".partial-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def checked_file(root, reference):
    path = contained(root, reference["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
