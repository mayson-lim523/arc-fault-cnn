"""
Evaluation utilities for the arc fault CNN.

전체 정확도만 보면 80:20 불균형 데이터에서 착시가 생깁니다.
따라서 이 파일은 아크 recall과 정상 오검출률을 반드시 함께 계산합니다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import tensorflow as tf

import config
from data import make_tf_dataset, steps_for_split


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """클래스 0=정상, 1=아크 기준의 TN/FP/FN/TP를 계산합니다."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def metrics_from_counts(counts: Dict[str, int], params: int | None = None) -> Dict[str, float | int]:
    """혼동행렬 count에서 프로젝트 핵심 지표를 계산합니다."""
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    total = max(tn + fp + fn + tp, 1)

    accuracy = (tn + tp) / total
    arc_recall = tp / max(tp + fn, 1)
    false_positive_rate = fp / max(fp + tn, 1)
    arc_precision = tp / max(tp + fp, 1)
    arc_f1 = 2 * arc_precision * arc_recall / max(arc_precision + arc_recall, 1e-12)

    result: Dict[str, float | int] = {
        **counts,
        "accuracy": accuracy,
        "arc_recall": arc_recall,
        "false_positive_rate": false_positive_rate,
        "arc_precision": arc_precision,
        "arc_f1": arc_f1,
    }
    if params is not None:
        result["params"] = int(params)
    return result


def evaluate_model(
    model: tf.keras.Model,
    split: str = "test",
    channels: str = "all",
    normalize: str = "global",
    split_mode: str = "author",
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    threshold: float = 0.5,
    h5_path: str | None = None,
) -> Dict[str, float | int | str]:
    """
    모델을 평가하고 핵심 지표를 dict로 반환합니다.

    threshold:
        softmax의 아크 확률(prob[:, 1])이 threshold 이상이면 아크로 예측합니다.
        threshold를 낮추면 아크 recall은 올라갈 수 있지만 FP율도 올라갈 수 있습니다.
    """
    dataset = make_tf_dataset(
        split=split,
        channels=channels,
        normalize=normalize,
        split_mode=split_mode,
        batch_size=batch_size,
        shuffle=False,
        h5_path=h5_path,
    )
    steps = steps_for_split(split, batch_size, split_mode, h5_path)

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for step, (x_batch, y_batch) in enumerate(dataset):
        if step >= steps:
            break
        probabilities = model.predict(x_batch, verbose=0)
        arc_probability = probabilities[:, 1]
        predicted = (arc_probability >= threshold).astype(np.int64)
        all_true.append(y_batch.numpy().astype(np.int64))
        all_pred.append(predicted)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    counts = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(counts, params=model.count_params())
    metrics.update(
        {
            "split": split,
            "channels": channels,
            "normalize": normalize,
            "split_mode": split_mode,
            "threshold": threshold,
        }
    )
    return metrics


def save_metrics_json(metrics: Dict[str, float | int | str], output_path: Path) -> None:
    """평가 결과를 사람이 읽기 쉬운 JSON으로 저장합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_csv(rows: Iterable[Dict[str, float | int | str]], output_path: Path) -> None:
    """여러 실험 결과 dict를 하나의 CSV 표로 저장합니다."""
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to write")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_path",
        "channels",
        "split_mode",
        "normalize",
        "threshold",
        "accuracy",
        "arc_recall",
        "false_positive_rate",
        "arc_precision",
        "arc_f1",
        "params",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved arc fault CNN model.")
    parser.add_argument("--model-path", required=True, help="Path to a .keras model file")
    parser.add_argument("--h5-path", default=None, help="Optional override for arc_fault_dataset.h5")
    parser.add_argument("--split", default="test", choices=config.SPLIT_NAMES)
    parser.add_argument("--channels", default="all", choices=("all", "current"))
    parser.add_argument("--normalize", default="global", choices=("global", "zscore"))
    parser.add_argument("--split-mode", default="author", choices=("author", "random", "block"))
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    metrics = evaluate_model(
        model=model,
        split=args.split,
        channels=args.channels,
        normalize=args.normalize,
        split_mode=args.split_mode,
        batch_size=args.batch_size,
        threshold=args.threshold,
        h5_path=args.h5_path,
    )
    metrics["model_path"] = args.model_path

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output_json:
        save_metrics_json(metrics, Path(args.output_json))


if __name__ == "__main__":
    main()
