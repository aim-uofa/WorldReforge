from __future__ import annotations

import math
from pathlib import Path

from worldreforge.io import atomic_json, output_path, read_json
from worldreforge.evaluation.motion import METRICS
from worldreforge.evaluation.pixels import aggregate_videos


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite result value")


def equal_tree(actual, expected, location="summary"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"Keys differ: {location}")
        for key in expected:
            equal_tree(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                f"Numeric aggregate differs: {location}")
    else:
        require(actual == expected and type(actual) is type(expected), f"Value differs: {location}")


def check_motion(entity, measured):
    counts = ("query_points", "reference_pairs", "possible_pairs", "terminal_pairs", "valid_future_frames",
              "windows_with_ade", "windows_with_fde")
    for name in counts:
        require(type(entity[name]) is int and entity[name] >= 0, f"Invalid motion count: {name}")
    n, pairs, possible = (entity[k] for k in ("query_points", "reference_pairs", "possible_pairs"))
    terminal, frames = entity["terminal_pairs"], entity["valid_future_frames"]
    require(n <= 10 and possible == n * 16, "Point quota or possible-pair denominator changed")
    require(0 <= frames <= 16 and frames <= pairs <= frames * n and terminal <= n and terminal <= pairs,
            "Inconsistent visible point/frame counts")
    require(not terminal or frames > 0, "Terminal support without a valid future frame")
    require(entity["windows_with_ade"] == int(frames > 0) and
            entity["windows_with_fde"] == int(terminal > 0), "Motion valid-video counts differ")
    if not measured:
        require(n == pairs == possible == terminal == frames == 0, "Unexpected motion reference")
    for metric in METRICS:
        support = terminal if metric in {"fde_px", "static_fde_px"} else frames
        value = entity[metric]
        if support:
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"Invalid {metric}")
        else:
            require(value is None, f"Missing reference must stay null: {metric}")
    expected_fraction = pairs / possible if possible else None
    equal_tree(entity["reference_fraction"], expected_fraction, "reference_fraction")
    fraction = entity["prediction_visible_fraction"]
    require((pairs == 0 and fraction is None) or (pairs > 0 and type(fraction) in (int, float) and 0 <= fraction <= 1),
            "Invalid prediction visibility diagnostic")


def validate_results(rows, metrics_root, output, *, pixels_only=False):
    output = output_path(output)
    require(not output.exists(), "Validation output already exists; select a new output")
    metrics_root = Path(metrics_root)
    try:
        ids = [r["id"] for r in rows]
        require(len(ids) == len(set(ids)) and bool(ids), "Empty or duplicate expected member IDs")
        complete = read_json(metrics_root / "COMPLETE.json")
        require(complete["status"] == "passed" and complete["windows"] == len(ids)
                and complete["ids"] == ids and complete["pixels_only"] is pixels_only,
                "Scoring completion/member protocol differs")
        files = list((metrics_root / "cases").glob("*.json"))
        require({p.stem for p in files} == set(ids), "Missing or extra per-video metrics")
        results = []
        for row in rows:
            cid = row["id"]
            path = metrics_root / "cases" / f"{cid}.json"
            result = read_json(path)
            finite_tree(result)
            for key in ("id", "domain", "panel", "asset"):
                require(result.get(key) == row.get(key), f"Case metadata changed: {cid}/{key}")
            pixels = result["pixels"]
            require(pixels["frames_scored"] == 16, "Expected 16 future frames")
            for name in ("mse", "ssim", "psnr_db", "static_mse", "static_psnr_db", "temporal_difference_mse"):
                require(type(pixels[name]) in (int, float) and math.isfinite(pixels[name]), f"Missing/nonfinite {name}")
            require(-1 <= pixels["ssim"] <= 1, "SSIM outside [-1,1]")
            for name in ("mse", "roi_mse", "static_mse"):
                value = pixels[name]
                if value is not None:
                    require(0 <= value <= 1, f"Invalid normalized {name}")
                expected = -10 * math.log10(max(value, 1e-15)) if value is not None else None
                equal_tree(pixels[name.replace("mse", "psnr_db")], expected, name + "_to_PSNR")
            require(pixels["temporal_difference_mse"] >= 0, "Negative temporal error")
            measured = bool(row.get("points")) and not pixels_only
            status = "explicitly_skipped" if pixels_only else ("scored" if measured else "no_qualified_reference")
            require(result["motion_status"] == status and result["motion"]["unit"] == "pixel",
                    "Motion status or unit changed")
            for entity in ("object", "gripper"):
                check_motion(result["motion"][entity], measured)
            for key in METRICS:
                a, b = result["motion"]["object"][key], result["motion"]["gripper"][key]
                equal_tree(result["motion"]["balanced"][key], (a + b) / 2 if a is not None and b is not None else None,
                           "balanced." + key)
            results.append(result)
        groups = {"all": results}
        for key in ("domain", "panel"):
            for value in sorted({r[key] for r in results}):
                groups[f"{key}:{value}"] = [r for r in results if r[key] == value]
        expected = {k: aggregate_videos(v) for k, v in groups.items()}
        summary = read_json(metrics_root / "summary.json")
        finite_tree(summary)
        equal_tree(summary, expected)
        report = dict(status="passed", windows=len(results), pixels_only=pixels_only,
                      motion_reference_windows=sum(bool(r.get("points")) and not pixels_only for r in rows),
                      execution_status="passed", scientific_status="not_inferred_by_integrity_checks")
    except Exception as error:
        atomic_json(output, dict(status="failed", execution_status="failed", error_type=type(error).__name__, error=str(error)))
        raise
    atomic_json(output, report)
    return report
