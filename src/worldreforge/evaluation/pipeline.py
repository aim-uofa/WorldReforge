from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys

from worldreforge.data.manifest import PAYLOAD_FIELDS, parse_manifest, summary as member_summary
from worldreforge.io import atomic_json, output_path, read_json
from worldreforge.runtime import gpu_device


def check_tracker_files(repository, checkpoint):
    if repository is None or checkpoint is None:
        raise ValueError("Supply --tracker-repository and --tracker-checkpoint for motion evaluation")
    for path in (Path(repository) / "cotracker/predictor.py", Path(checkpoint)):
        if not path.is_file():
            raise FileNotFoundError(path)


def check_coverage(directory, expected_ids):
    receipt = read_json(directory / "COMPLETE.json")
    expected = {"status": "passed", "windows": len(expected_ids), "ids": expected_ids}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{directory.name} coverage differs from requested members")


def run_pipeline(*, manifest, data_root, checkpoint, tokenizer, output, text_tokenizer=None,
                 batch_size=1, gpu=0, tracker_repository=None, tracker_checkpoint=None,
                 cpu_workers=8, pixels_only=False):
    manifest, data_root, output = Path(manifest).resolve(), Path(data_root).resolve(), output_path(output)
    content = manifest.read_bytes()
    rows = parse_manifest(content, data_root, check_files=PAYLOAD_FIELDS)
    device = gpu_device(gpu)
    if batch_size < 1 or cpu_workers < 1:
        raise ValueError("Batch size and CPU worker count must be positive")
    required = [Path(checkpoint) / "config.json", Path(tokenizer)]
    if text_tokenizer is not None:
        required.extend(Path(text_tokenizer) / name for name in ("vocab.json", "merges.txt"))
    if any(r.get("points") for r in rows) and not pixels_only:
        check_tracker_files(tracker_repository, tracker_checkpoint)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    stage = "inputs"
    try:
        frozen = output / "manifest.jsonl"
        frozen.write_bytes(content)
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
        check_coverage(output / "inference", [row["id"] for row in rows])
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
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    check_coverage(stage_root, expected_ids)


def evaluate(*, manifest, data_root, prediction_root, output, tracker_repository=None,
             tracker_checkpoint=None, track_root=None, gpu=0, cpu_workers=8, pixels_only=False):
    manifest, data_root, output = Path(manifest).resolve(), Path(data_root).resolve(), output_path(output)
    prediction_root = Path(prediction_root).resolve()
    if cpu_workers < 1:
        raise ValueError("CPU worker count must be positive")
    content = manifest.read_bytes()
    fields = ("gt",) if pixels_only else ("gt", "points", "reference")
    rows = parse_manifest(content, data_root, check_files=fields)
    need_tracking = any(r.get("points") for r in rows) and not pixels_only and track_root is None
    if pixels_only and track_root is not None:
        raise ValueError("--pixels-only cannot be combined with --track-root")
    for path in (prediction_root, track_root):
        if path is not None and not Path(path).is_dir():
            raise FileNotFoundError(path)
    device = None
    if need_tracking:
        device = gpu_device(gpu)
        check_tracker_files(tracker_repository, tracker_checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    stage = "inputs"
    try:
        frozen = output / "manifest.jsonl"
        frozen.write_bytes(content)
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
