# Input format

`manifests/eval1200.jsonl` defines 1,200 Reforge-Eval simulation windows
with relative file paths.
Each window has 17 observations and 16 state transitions at a model time base of 15 FPS.
Each `samples/<id>/` directory contains:

| File | Format |
|---|---|
| `condition.png` | RGB three-view mosaic: 1080 x 1280 (height x width) |
| `actions.json` | Normalized measured-state conditions, shape `[16, 10]` |
| `gt.npy` | Lossless reference video, `uint8 [17, 544, 640, 3]` |
| `points.json` | Up to 20 object and gripper queries visible in the first frame |
| `reference.npz` | Fixed-point trajectories and per-frame `visibility` / `visibility_state` |

When a motion reference is unavailable, both `points` and `reference` are `null`.

## State conditioning

Each transition contains the translation and first two rotation columns of
`inv(T_t) @ T_(t+1)`, followed by `1 - measured_gripper_closure[t+1]`.
The provided conditions have already been normalized with the configured quantiles.
Inference consumes them directly, without clipping or a second normalization.

## Spatial and inference settings

Spatial dimensions below are height by width. Reforge-Eval conditioning mosaics
are resized to 544 x 645 with their aspect ratio preserved.
The input is reflection-padded on the right
to 544 x 736. VAE latents are cropped from 34 x 46 to 34 x 40 by the native Cosmos
path, producing 544 x 640 frames. The DiT token grid is 5 x 17 x 20.
Reference images, optional ROIs, and point coordinates use the same output canvas.

Default inference settings: one GPU, batch size 1, seed 62100, 30 sampling steps,
guidance 1, shift 10, and eager model execution.
These settings apply to both the original pretrained Nano checkpoint and
WorldReforge. `scripts/infer_original.sh` selects the original checkpoint;
`scripts/evaluate.sh --checkpoint models/Cosmos3-Nano ...` also runs scoring.
