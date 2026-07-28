"""
Run the core experiment matrix and collect a CSV summary.

각 실험은 train.py를 별도 프로세스로 실행합니다.
그래야 GPU 메모리와 TensorFlow 상태가 실험 사이에 덜 꼬이고, 실패한 조합도 로그로 분리됩니다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import config
from evaluate import write_summary_csv


DEFAULT_EXPERIMENTS = [
    {"channels": "all", "split_mode": "author", "normalize": "global"},
    {"channels": "current", "split_mode": "author", "normalize": "global"},
    {"channels": "all", "split_mode": "block", "normalize": "global"},
    {"channels": "all", "split_mode": "random", "normalize": "global"},
    {"channels": "all", "split_mode": "author", "normalize": "zscore"},
    {"channels": "current", "split_mode": "author", "normalize": "zscore"},
]


def run_one_experiment(args: argparse.Namespace, spec: dict, index: int) -> Path:
    """한 실험 조합을 실행하고 결과 JSON 경로를 반환합니다."""
    run_name = f"{args.run_prefix}_{index:02d}"
    run_id = f"{spec['channels']}_{spec['split_mode']}_{spec['normalize']}_{run_name}"
    metrics_path = config.RESULT_DIR / f"{run_id}_test_metrics.json"

    command = [
        sys.executable,
        "train.py",
        "--channels",
        spec["channels"],
        "--split-mode",
        spec["split_mode"],
        "--normalize",
        spec["normalize"],
        "--run-name",
        run_name,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--arc-weight-multiplier",
        str(args.arc_weight_multiplier),
        "--threshold",
        str(args.threshold),
        "--patience",
        str(args.patience),
    ]
    if args.h5_path:
        command.extend(["--h5-path", args.h5_path])

    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run arc fault CNN experiments.")
    parser.add_argument("--h5-path", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--arc-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--run-prefix", default="exp")
    parser.add_argument("--summary-csv", default=str(config.RESULT_DIR / "experiment_summary.csv"))
    args = parser.parse_args()

    rows = []
    for index, spec in enumerate(DEFAULT_EXPERIMENTS, start=1):
        metrics_path = run_one_experiment(args, spec, index)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(metrics)

    write_summary_csv(rows, Path(args.summary_csv))
    print(f"Saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
