"""
Training script for the arc fault CNN.

Kaggle Notebook 예시:
    !python train.py --h5-path /kaggle/input/arc-fault-dataset/arc_fault_dataset.h5 \
        --channels all --split-mode author --normalize global --epochs 20 --run-name v1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

import config
from data import count_labels, make_tf_dataset, steps_for_split
from evaluate import evaluate_model, save_metrics_json
from model import build_arc_cnn, describe_model


def set_reproducible_seed(seed: int = config.SEED) -> None:
    """Python, NumPy, TensorFlow 난수를 고정해 실험 재현성을 높입니다."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_class_weight(split_mode: str, h5_path: str | None, arc_weight_multiplier: float) -> Dict[int, float]:
    """
    클래스 불균형을 보정하기 위한 class_weight를 계산합니다.

    데이터가 정상 약 80%, 아크 약 20%라서 보정하지 않으면 모델이 정상만 예측해도
    정확도 79.8% 근처가 나올 수 있습니다. 아크 놓침이 더 위험하므로 class 1 가중치를
    multiplier로 추가 조절할 수 있게 했습니다.
    """
    counts = count_labels(split="train", split_mode=split_mode, h5_path=h5_path)
    total = sum(counts.values())
    class_weight = {}
    for class_id in range(config.N_CLASSES):
        class_count = max(counts.get(class_id, 0), 1)
        class_weight[class_id] = total / (config.N_CLASSES * class_count)
    class_weight[1] *= arc_weight_multiplier
    return class_weight


def build_callbacks(model_path: Path, log_path: Path, patience: int) -> list:
    """최고 모델 저장, 조기 종료, 학습 로그 저장 콜백을 구성합니다."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(log_path)),
    ]


def train_model(args: argparse.Namespace) -> Path:
    """명령행 인자를 받아 모델 학습, 저장, 테스트 평가까지 수행합니다."""
    set_reproducible_seed(config.SEED)

    channel_count = 3 if args.channels == "all" else 1
    model = build_arc_cnn(
        input_shape=(config.WINDOW_LEN, channel_count),
        base_filters=args.base_filters,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    print(describe_model(model))

    run_id = f"{args.channels}_{args.split_mode}_{args.normalize}_{args.run_name}"
    model_path = config.MODEL_DIR / f"{run_id}.keras"
    log_path = config.RESULT_DIR / f"{run_id}_history.csv"
    metrics_path = config.RESULT_DIR / f"{run_id}_test_metrics.json"

    train_ds = make_tf_dataset(
        split="train",
        channels=args.channels,
        normalize=args.normalize,
        split_mode=args.split_mode,
        batch_size=args.batch_size,
        shuffle=True,
        h5_path=args.h5_path,
    )
    valid_ds = make_tf_dataset(
        split="valid",
        channels=args.channels,
        normalize=args.normalize,
        split_mode=args.split_mode,
        batch_size=args.batch_size,
        shuffle=False,
        h5_path=args.h5_path,
    )

    train_steps = steps_for_split("train", args.batch_size, args.split_mode, args.h5_path)
    valid_steps = steps_for_split("valid", args.batch_size, args.split_mode, args.h5_path)
    class_weight = make_class_weight(args.split_mode, args.h5_path, args.arc_weight_multiplier)

    print(f"class_weight = {json.dumps(class_weight, ensure_ascii=False)}")
    print(f"train_steps={train_steps}, valid_steps={valid_steps}, batch_size={args.batch_size}")

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        class_weight=class_weight,
        callbacks=build_callbacks(model_path, log_path, args.patience),
        verbose=1,
    )

    # ModelCheckpoint가 저장한 val_loss 기준 최고 모델을 다시 불러와 테스트 지표를 계산합니다.
    best_model = tf.keras.models.load_model(model_path)
    metrics = evaluate_model(
        model=best_model,
        split="test",
        channels=args.channels,
        normalize=args.normalize,
        split_mode=args.split_mode,
        batch_size=args.batch_size,
        threshold=args.threshold,
        h5_path=args.h5_path,
    )
    metrics["model_path"] = str(model_path)
    metrics["run_name"] = args.run_name
    save_metrics_json(metrics, metrics_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the arc fault 1D-CNN.")
    parser.add_argument("--h5-path", default=None, help="Optional override for arc_fault_dataset.h5")
    parser.add_argument("--channels", default="all", choices=("all", "current"))
    parser.add_argument("--normalize", default="global", choices=("global", "zscore"))
    parser.add_argument("--split-mode", default="author", choices=("author", "random", "block"))
    parser.add_argument("--run-name", default="v1", help="Short memo used in saved file names")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--arc-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
