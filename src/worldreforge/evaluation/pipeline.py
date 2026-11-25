from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys

from worldreforge.data.manifest import load_manifest, summary as member_summary
from worldreforge.io import atomic_json, contained, output_path, read_json
from worldreforge.runtime import gpu_device


def run_pipeline(*, manifest, data_root, checkpoint, tokenizer, output, text_tokenizer=None,
                 batch_size=1, gpu=0, tracker_repository=None, tracker_checkpoint=None,
                 cpu_workers=8, pixels_only=False):
    manifest, data_root, output = Path(manifest).resolve(), Path(data_root).resolve(), output_path(output)
    rows = load_manifest(manifest, data_root, verify_files=True)
    device = gpu_device(gpu)
    if batch_size < 1 or cpu_workers < 1:
        raise ValueError("Batch size and CPU worker count must be positive")
    required = [Path(checkpoint) / "config.json", Path(tokenizer)]
    if text_tokenizer is not None:
        required.extend(Path(text_tokenizer) / name for name in ("vocab.json", "merges.txt"))
    if any(r.get("points") for r in rows) and not pixels_only:
        if tracker_repository is None or tracker_checkpoint is None:
            raise ValueError("Supply --tracker-repository and --tracker-checkpoint for motion evaluation")
        required.extend([Path(tracker_repository) / "cotracker/predictor.py", Path(tracker_checkpoint)])
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    stage = "inputs"
    try:
        frozen = output / "manifest.jsonl"
        frozen.write_bytes(manifest.read_bytes())
        if load_manifest(frozen, data_root) != rows:
            raise ValueError("Manifest changed during preparation")
        atomic_json(output / "PLAN.json", dict(started_at=datetime.now(timezone.utc).isoformat(),
            members=member_summary(rows), device=device, batch_size=batch_size,
            checkpoint=str(Path(checkpoint).resolve()), cpu_workers=cpu_workers, pixels_only=pixels_only))
        command = [sys.executable, "-m", "worldreforge", "infer", "--manifest", str(frozen),
            "--data-root", str(data_root), "--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer),
            "--output", str(output / "inference"), "--gpu", str(gpu), "--batch-size", str(batch_size)]
        if text_tokenizer is not None:
            command.extend(["--text-tokenizer", str(text_tokenizer)])
        stage = "inference"
        print("Running single-GPU inference...", flush=True)

        subprocess.run(command, check=True)
        receipt = read_json(output / "inference/COMPLETE.json")
        if (receipt.get("status") != "passed" or receipt.get("windows") != len(rows)
                or receipt.get("ids") != [r["id"] for r in rows]):
            raise ValueError("Inference coverage differs from requested members")
        stage = "evaluation"
        print("Inference complete. Running evaluation...", flush=True)
        report = evaluate(manifest=frozen, data_root=data_root,
            prediction_root=output / "inference/predictions", output=output / "evaluation",
            tracker_repository=tracker_repository, tracker_checkpoint=tracker_checkpoint,
            gpu=gpu, cpu_workers=cpu_workers, pixels_only=pixels_only)
        atomic_json(output / "COMPLETE.json", report["completion"])
        return report
    except BaseException as error:
        atomic_json(output / "FAILED.json", dict(status="failed", stage=stage,
                    error_type=type(error).__name__, error=str(error)))
        raise


def run_tracking(arguments, stage_root, expected_ids, logs):

    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    command = [sys.executable, "-m", "worldreforge", "track", *map(str, arguments)]
    with (logs / "tracking.log").open("x") as log:
        process = subprocess.Popen(command, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            atomic_json(logs / "tracking_process.json", dict(pid=process.pid))
            if process.wait() != 0:
                raise RuntimeError(f"Point tracking failed; see {logs / 'tracking.log'}")
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            raise
    receipt = read_json(stage_root / "COMPLETE.json")
    if (receipt.get("status") != "passed" or receipt.get("windows") != len(expected_ids)
            or receipt.get("ids") != expected_ids):
        raise ValueError("Tracking coverage differs from requested members")


def evaluate(*, manifest, data_root, prediction_root, output, tracker_repository=None,
             tracker_checkpoint=None, track_root=None, gpu=0, cpu_workers=8, pixels_only=False):
    manifest, data_root, output = Path(manifest).resolve(), Path(data_root).resolve(), output_path(output)
    prediction_root = Path(prediction_root).resolve()
    if cpu_workers < 1:
        raise ValueError("CPU worker count must be positive")
    rows = load_manifest(manifest, data_root)
    need_tracking = any(r.get("points") for r in rows) and not pixels_only and track_root is None
    if need_tracking and (tracker_repository is None or tracker_checkpoint is None):
        raise ValueError("Supply tracker repository/checkpoint, or reuse --track-root")
    if pixels_only and track_root is not None:
        raise ValueError("--pixels-only cannot be combined with --track-root")
    for row in rows:
        fields = ["gt"] + (["points", "reference"] if row.get("points") and not pixels_only else [])
        for key in fields:
            if not contained(data_root, row[key]["path"]).is_file():
                raise FileNotFoundError(f"Missing input {row['id']}/{key}")
    for path in (prediction_root, track_root):
        if path is not None and not Path(path).is_dir():
            raise FileNotFoundError(path)
    device = None
    if need_tracking:
        device = gpu_device(gpu)
        if not (Path(tracker_repository) / "cotracker/predictor.py").is_file() or not Path(tracker_checkpoint).is_file():
            raise FileNotFoundError("CoTracker repository or checkpoint is missing")
    output.mkdir(parents=True, exist_ok=False)
    stage = "inputs"
    try:
        frozen = output / "manifest.jsonl"
        frozen.write_bytes(manifest.read_bytes())
        if load_manifest(frozen, data_root) != rows:
            raise ValueError("Manifest changed during preparation")
        track_root = Path(track_root).resolve() if track_root is not None else output / "tracking"
        atomic_json(output / "PLAN.json", dict(started_at=datetime.now(timezone.utc).isoformat(),
            members=member_summary(rows), tracking_device=device,
            prediction_root=str(prediction_root), track_root=None if pixels_only else str(track_root),
            cpu_workers=cpu_workers, pixels_only=pixels_only,
            tracking="run" if need_tracking else "reuse_or_not_applicable"))
        if need_tracking:
            stage = "tracking"
            run_tracking(["--manifest", frozen, "--data-root", data_root,
                "--prediction-root", prediction_root, "--tracker-repository", tracker_repository,
                "--tracker-checkpoint", tracker_checkpoint, "--output", track_root, "--gpu", gpu],
                track_root, [r["id"] for r in rows if r.get("points")], output / "logs")
        stage = "scoring"
        from worldreforge.evaluation.runner import score
        report = score(rows, data_root, prediction_root, None if pixels_only else track_root,
                       output / "metrics", cpu_workers, pixels_only)
        stage = "result_validation"
        from worldreforge.evaluation.validation import validate_results
        validate_results(rows, output / "metrics", output / "checks/results.json", pixels_only=pixels_only)
        complete = dict(status="passed", execution_status="passed", scientific_status="not_inferred_by_integrity_checks",
            windows=len(rows), pixels_only=pixels_only,
            completed_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output / "COMPLETE.json", complete)
        return dict(completion=complete, overall=report["all"])
    except BaseException as error:
        atomic_json(output / "FAILED.json", dict(status="failed", execution_status="failed", stage=stage,
                    error_type=type(error).__name__, error=str(error)))
        raise
