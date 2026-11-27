from __future__ import annotations

from contextlib import contextmanager
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


@contextmanager
def atomic_file(path, mode="wb", **options):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, prefix=".partial-",
                                         delete=False, **options)
    try:
        with stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stream.name, path)
    except BaseException:
        Path(stream.name).unlink(missing_ok=True)
        raise


def atomic_json(path, value):
    with atomic_file(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def atomic_array(path, array):
    import numpy as np

    with atomic_file(path) as stream:
        np.save(stream, array, allow_pickle=False)


def checked_file(root, reference):
    path = contained(root, reference["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
