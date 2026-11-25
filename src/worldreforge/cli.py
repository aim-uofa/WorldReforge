from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="WorldReforge inference and evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Generate predictions, evaluate, and validate results")
    infer = commands.add_parser("infer", help="Generate videos on one GPU")
    for sub in (run, infer):
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--tokenizer", type=Path, required=True)
        sub.add_argument("--text-tokenizer", type=Path, help="Local Qwen text tokenizer directory")
        sub.add_argument("--batch-size", type=int, default=1)
        sub.add_argument("--gpu", type=int, default=0, help="One index within CUDA_VISIBLE_DEVICES")

    evaluate = commands.add_parser("eval", help="Evaluate existing predictions and validate results")
    evaluate.add_argument("--prediction-root", type=Path, required=True)
    evaluate.add_argument("--track-root", type=Path, help="Reuse existing prediction tracks")
    evaluate.add_argument("--gpu", type=int, default=0, help="GPU for point tracking")
    for sub in (run, evaluate):
        sub.add_argument("--tracker-repository", type=Path)
        sub.add_argument("--tracker-checkpoint", type=Path)
        sub.add_argument("--cpu-workers", type=int, default=8)
        sub.add_argument("--pixels-only", action="store_true")

    track = commands.add_parser("track", help="Track predicted videos on one GPU")
    track.add_argument("--prediction-root", type=Path, required=True)
    track.add_argument("--tracker-repository", type=Path, required=True)
    track.add_argument("--tracker-checkpoint", type=Path, required=True)
    track.add_argument("--gpu", type=int, default=0)

    score = commands.add_parser("score", help="Compute metrics from existing predictions and tracks")
    score.add_argument("--prediction-root", type=Path, required=True)
    score.add_argument("--track-root", type=Path)
    score.add_argument("--cpu-workers", type=int, default=8)
    score.add_argument("--pixels-only", action="store_true")
    for sub in (run, infer, evaluate, track, score):
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--data-root", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate-eval", help="Check input manifests and optional payloads")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--data-root", type=Path)
    validate.add_argument("--verify-files", action="store_true")
    acceptance = commands.add_parser("validate-results", help="Check metric coverage and aggregation")
    acceptance.add_argument("--manifest", type=Path, required=True)
    acceptance.add_argument("--metrics-root", type=Path, required=True)
    acceptance.add_argument("--output", type=Path, required=True)
    acceptance.add_argument("--pixels-only", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command in {"run", "eval"}:
        from worldreforge.evaluation.pipeline import evaluate, run_pipeline
        options = vars(args).copy()
        options.pop("command")
        report = (run_pipeline if args.command == "run" else evaluate)(**options)
    else:
        from worldreforge.data.manifest import load_manifest, summary
        rows = load_manifest(args.manifest, getattr(args, "data_root", None),
                             verify_files=getattr(args, "verify_files", False))
        if args.command == "validate-eval":
            report = summary(rows)
        elif args.command == "validate-results":
            from worldreforge.evaluation.validation import validate_results
            report = validate_results(rows, args.metrics_root, args.output, pixels_only=args.pixels_only)
        elif args.command == "infer":
            from worldreforge.inference.nano import infer
            report = infer(rows, args.data_root, args.checkpoint, args.tokenizer, args.output,
                           batch_size=args.batch_size, gpu=args.gpu, text_tokenizer=args.text_tokenizer)
        elif args.command == "track":
            from worldreforge.evaluation.track_runner import track
            report = track(rows, args.data_root, args.prediction_root, args.tracker_repository,
                           args.tracker_checkpoint, args.output, gpu=args.gpu)
        elif args.command == "score":
            from worldreforge.evaluation.runner import score
            report = score(rows, args.data_root, args.prediction_root, args.track_root,
                           args.output, args.cpu_workers, args.pixels_only)["all"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
