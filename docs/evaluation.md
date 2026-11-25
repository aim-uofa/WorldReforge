# Evaluation metrics

`scripts/evaluate.sh` generates predictions, runs point tracking, computes metrics,
and validates the results. Nano inference finishes and releases GPU memory before
CoTracker is loaded on the same GPU. Pixel scoring uses CPU workers.

Videos have equal weight in each aggregate. Valid sample counts are reported,
and unavailable references remain `null`.

## Space and time

All three views share a canvas of height 544 and width 640.
Regions use `[x0, y0, x1, y1]` coordinates:

| View | Region |
|---|---|
| Wrist | `[0, 0, 640, 363]` |
| Left external | `[0, 363, 323, 544]` |
| Right external | `[323, 363, 640, 544]` |

Frame 0 is conditioning context. Metrics cover future frames 1 through 16.
Predictions and references must have identical shapes and frame indices.

## Pixel metrics

RGB values are converted to float32 and divided by 255. MSE is computed over all
RGB pixels in each frame, then averaged over the 16 future frames.

**Video PSNR = -10 log10(max(video MSE, 1e-15)).**
Aggregate PSNR is the mean of per-video PSNR values.
SSIM uses a 7-pixel window, `channel_axis=2`, and `data_range=1`,
with frame averaging followed by video averaging.
Temporal MSE compares first-order frame differences.
The static baseline repeats frame 0. Optional ROIs use the same canvas coordinates.

## Motion metrics

Object queries: at most 6 in the wrist view and 2 in each external view.
Gripper queries: at most 3 per finger in the wrist view and 1 per finger in each
external view. There are at most 20 points in total.

CoTracker tracks each view separately and maps coordinates back to the full canvas.
Queries must be visible in the first reference frame. Later frames are scored
using reference visibility and camera bounds. Predicted invisibility or
out-of-frame positions do not remove errors. Nonfinite coordinates are rejected.

For each entity and frame, point error is the mean Euclidean distance over
reference-visible points, in pixels. ADE averages valid future frames.
FDE uses only frame 16 and is `null` when that frame has no visible reference.
Motion ADE compares displacements from frame 0.
The static baseline compares future reference positions against their initial positions.

Object and gripper metrics are aggregated separately, with valid-video,
point-frame, and terminal-frame counts.
