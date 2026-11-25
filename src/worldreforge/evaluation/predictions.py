import numpy as np
from worldreforge.io import contained, read_json
from worldreforge.evaluation.pixels import validate_video


def load_prediction(root, row):
    path = contained(root, f"{row['id']}/prediction.npy")
    receipt = read_json(path.parent / "COMPLETE.json")
    if receipt.get("id") != row["id"]:
        raise ValueError("Prediction sample ID differs from the requested sample")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    validate_video(array)
    return array
