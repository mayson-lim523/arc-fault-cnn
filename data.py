"""
Data loading and preprocessing for the arc fault CNN.

핵심 계약:
    load_split(split, channels="all", normalize="global", split_mode="author")

위 함수의 시그니처는 팀 공통 규약이므로 임의로 바꾸지 않습니다.
학습처럼 큰 데이터를 다룰 때는 같은 전처리 로직을 쓰는 make_tf_dataset()을 사용합니다.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterator, Tuple

import h5py
import numpy as np

import config


# 사용자가 지정할 수 있는 채널 별칭입니다.
# "current"는 고주파 채널을 뺀 대조군 실험에 쓰입니다.
CHANNEL_INDEX = {
    "all": np.array([0, 1, 2], dtype=np.int64),
    "current": np.array([0], dtype=np.int64),
}


def _resolve_h5_path(h5_path: str | None = None) -> str:
    """명령행 인자, 환경변수, config 기본값 순서로 h5 경로를 결정합니다."""
    return h5_path or os.environ.get("ARC_H5_PATH") or config.H5_PATH


def _validate_options(split: str, channels: str, normalize: str, split_mode: str) -> None:
    """잘못된 옵션을 초기에 잡아 팀원들이 원인을 빨리 찾게 합니다."""
    if split not in config.SPLIT_NAMES:
        raise ValueError(f"split must be one of {config.SPLIT_NAMES}, got {split!r}")
    if channels not in CHANNEL_INDEX:
        raise ValueError(f"channels must be one of {tuple(CHANNEL_INDEX)}, got {channels!r}")
    if normalize not in ("global", "zscore"):
        raise ValueError("normalize must be 'global' or 'zscore'")
    if split_mode not in ("author", "random", "block"):
        raise ValueError("split_mode must be 'author', 'random', or 'block'")


def _split_lengths(h5_file: h5py.File) -> Dict[str, int]:
    """h5 파일 안의 train/valid/test 샘플 수를 읽습니다."""
    return {name: int(h5_file[name].shape[0]) for name in config.SPLIT_NAMES}


def _split_slices(lengths: Dict[str, int]) -> Dict[str, slice]:
    """전체 샘플을 하나로 이어 붙였다고 가정했을 때 각 split의 위치를 계산합니다."""
    total = sum(lengths.values())
    train_end = lengths["train"]
    valid_end = train_end + lengths["valid"]
    return {
        "train": slice(0, train_end),
        "valid": slice(train_end, valid_end),
        "test": slice(valid_end, total),
    }


def _mode_indices(h5_file: h5py.File, split: str, split_mode: str) -> np.ndarray:
    """
    split_mode별로 읽어야 할 global index를 만듭니다.

    global index는 train, valid, test를 순서대로 이어 붙였다고 생각한 위치입니다.
    이렇게 해두면 author/random/block 분할 모두 같은 배치 리더를 재사용할 수 있습니다.
    """
    lengths = _split_lengths(h5_file)
    slices = _split_slices(lengths)
    total = sum(lengths.values())

    if split_mode == "author":
        # h5 제작자가 제공한 원래 분할을 그대로 사용합니다.
        return np.arange(slices[split].start, slices[split].stop, dtype=np.int64)

    if split_mode == "block":
        # 셔플하지 않고 앞쪽을 train, 중간을 valid, 뒤쪽을 test로 둡니다.
        # 인접 윈도우 누수 가능성을 확인하기 위한 보수적인 검증 모드입니다.
        return np.arange(total, dtype=np.int64)[slices[split]]

    # random은 전체 샘플을 섞은 뒤 author와 같은 비율로 다시 나눕니다.
    rng = np.random.default_rng(config.SEED)
    shuffled = rng.permutation(total).astype(np.int64)
    return shuffled[slices[split]]


def _global_to_dataset_positions(
    global_indices: np.ndarray, lengths: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """global index를 h5 dataset 이름(train/valid/test)과 내부 row index로 변환합니다."""
    starts = np.array(
        [0, lengths["train"], lengths["train"] + lengths["valid"]],
        dtype=np.int64,
    )
    stops = np.array(
        [
            lengths["train"],
            lengths["train"] + lengths["valid"],
            lengths["train"] + lengths["valid"] + lengths["test"],
        ],
        dtype=np.int64,
    )
    source_ids = np.searchsorted(stops, global_indices, side="right")
    local_indices = global_indices - starts[source_ids]
    return source_ids, local_indices


def _read_rows_by_global_indices(
    h5_file: h5py.File, global_indices: np.ndarray
) -> np.ndarray:
    """
    global index 배열에 해당하는 원본 row를 읽습니다.

    h5py의 fancy indexing은 각 dataset 내부 index가 오름차순이어야 안전합니다.
    그래서 dataset별로 묶고 정렬해서 읽은 뒤 하나의 batch로 합칩니다.
    학습에서는 batch 안의 순서가 중요하지 않으므로 순서를 복원하지 않습니다.
    """
    lengths = _split_lengths(h5_file)
    source_ids, local_indices = _global_to_dataset_positions(global_indices, lengths)
    rows = []
    for source_id, split_name in enumerate(config.SPLIT_NAMES):
        mask = source_ids == source_id
        if not np.any(mask):
            continue
        sorted_local = np.sort(local_indices[mask])
        rows.append(h5_file[split_name][sorted_local])
    if not rows:
        raise ValueError("empty batch requested")
    return np.concatenate(rows, axis=0)


def _iter_raw_batches(
    h5_path: str,
    split: str,
    split_mode: str,
    batch_size: int,
    shuffle: bool = False,
    sample_limit: int | None = None,
) -> Iterator[np.ndarray]:
    """h5 파일에서 원본 row batch를 순차적으로 읽어옵니다."""
    with h5py.File(h5_path, "r") as h5_file:
        indices = _mode_indices(h5_file, split, split_mode)
        if sample_limit is not None:
            indices = indices[:sample_limit]
        if shuffle:
            rng = np.random.default_rng(config.SEED)
            indices = rng.permutation(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield _read_rows_by_global_indices(h5_file, batch_indices)


def _rows_to_xy(
    rows: np.ndarray,
    channels: str,
    normalize: str,
    global_mean: np.ndarray | None,
    global_std: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    h5 원본 row를 Keras Conv1D 입력 형태로 변환합니다.

    원본: (N, 9601)
      - 앞 9600개: 3채널 x 3200포인트
      - 마지막 1개: 라벨

    변환 후 X: (N, 3200, C)
      - Conv1D는 channels-last 형식이므로 transpose가 반드시 필요합니다.
    """
    channel_idx = CHANNEL_INDEX[channels]
    y = rows[:, -1].astype(np.int64)

    # reshape 결과는 (N, 3, 3200)입니다. 여기서 채널을 고른 뒤 (N, 3200, C)로 바꿉니다.
    x = rows[:, :-1].reshape(-1, config.N_CHANNELS, config.WINDOW_LEN)
    x = x[:, channel_idx, :].transpose(0, 2, 1).astype(np.float32)

    if normalize == "global":
        if global_mean is None or global_std is None:
            raise ValueError("global normalization requires train mean/std")
        x = (x - global_mean.reshape(1, 1, -1)) / global_std.reshape(1, 1, -1)
    else:
        # zscore는 샘플마다, 채널마다 평균과 표준편차를 따로 계산합니다.
        # 특정 윈도우의 DC offset 영향을 줄이고 파형 모양에 더 집중시키는 방식입니다.
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        x = (x - mean) / np.maximum(std, 1e-6)

    return x.astype(np.float32), y


def _stat_cache_path(h5_path: str, channels: str, split_mode: str) -> Path:
    """정규화 통계를 재사용하기 위한 캐시 파일 경로를 만듭니다."""
    safe_name = Path(h5_path).stem.replace(" ", "_")
    return config.STAT_CACHE_DIR / f"{safe_name}_{channels}_{split_mode}_global_stats.npz"


def compute_channel_stats(
    channels: str = "all",
    split_mode: str = "author",
    h5_path: str | None = None,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    train split에서만 채널별 mean/std를 계산합니다.

    valid/test 통계를 섞으면 평가 데이터 정보가 학습 과정에 새는 누수가 됩니다.
    따라서 global 정규화는 어떤 split을 불러오더라도 항상 train 기준 통계를 씁니다.
    """
    _validate_options("train", channels, "global", split_mode)
    resolved_path = _resolve_h5_path(h5_path)
    cache_path = _stat_cache_path(resolved_path, channels, split_mode)

    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        return cached["mean"].astype(np.float32), cached["std"].astype(np.float32)

    channel_count = len(CHANNEL_INDEX[channels])
    total_sum = np.zeros(channel_count, dtype=np.float64)
    total_sq_sum = np.zeros(channel_count, dtype=np.float64)
    total_count = 0

    for rows in _iter_raw_batches(
        resolved_path,
        split="train",
        split_mode=split_mode,
        batch_size=config.STAT_CHUNK_SIZE,
        shuffle=False,
    ):
        # 정규화 통계는 정규화 전 원본 값으로 계산해야 합니다.
        # rows 전체를 float32 X 배열로 만들지 않고 바로 합계를 내서 메모리 사용을 줄입니다.
        raw = rows[:, :-1].reshape(-1, config.N_CHANNELS, config.WINDOW_LEN)
        raw = raw[:, CHANNEL_INDEX[channels], :].astype(np.float64)
        total_sum += raw.sum(axis=(0, 2))
        total_sq_sum += np.square(raw).sum(axis=(0, 2))
        total_count += raw.shape[0] * config.WINDOW_LEN

    mean = total_sum / total_count
    variance = np.maximum(total_sq_sum / total_count - np.square(mean), 1e-12)
    std = np.sqrt(variance)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    return mean.astype(np.float32), std.astype(np.float32)


def load_split(split, channels="all", normalize="global", split_mode="author"):
    """
    팀 공통 데이터 인터페이스입니다. 시그니처를 바꾸지 마세요.

    Args:
        split: "train", "valid", "test" 중 하나입니다.
        channels: "all"은 3채널 전체, "current"는 채널0 전류만 사용합니다.
        normalize: "global"은 train 통계로 채널별 표준화, "zscore"는 윈도우별 표준화입니다.
        split_mode: "author"는 h5 기본 분할, "random"/"block"은 누수 검증용 재분할입니다.

    Returns:
        X: np.ndarray, shape (N, 3200, C), dtype float32
        y: np.ndarray, shape (N,), dtype int64

    주의:
        train 전체를 numpy 배열로 반환하면 메모리를 많이 씁니다.
        실제 학습은 make_tf_dataset()을 권장하고, load_split()은 shape 확인/소규모 분석에 쓰세요.
    """
    _validate_options(split, channels, normalize, split_mode)
    h5_path = _resolve_h5_path()
    mean = std = None
    if normalize == "global":
        mean, std = compute_channel_stats(channels=channels, split_mode=split_mode, h5_path=h5_path)

    xs, ys = [], []
    for rows in _iter_raw_batches(
        h5_path,
        split=split,
        split_mode=split_mode,
        batch_size=config.STAT_CHUNK_SIZE,
        shuffle=False,
        sample_limit=config.LOAD_SPLIT_SAMPLE_LIMIT,
    ):
        x_batch, y_batch = _rows_to_xy(rows, channels, normalize, mean, std)
        xs.append(x_batch)
        ys.append(y_batch)

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def make_tf_dataset(
    split: str,
    channels: str = "all",
    normalize: str = "global",
    split_mode: str = "author",
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    h5_path: str | None = None,
    repeat: bool = False,
):
    """
    TensorFlow 학습용 Dataset을 만듭니다.

    load_split()과 동일한 전처리를 사용하지만, h5를 batch 단위로 읽어서 메모리를 아낍니다.
    repeat=True는 Keras fit에서 여러 epoch를 학습할 때 데이터가 고갈되지 않게 합니다.
    TensorFlow import 비용을 줄이기 위해 함수 내부에서 import합니다.
    """
    import tensorflow as tf

    _validate_options(split, channels, normalize, split_mode)
    resolved_path = _resolve_h5_path(h5_path)
    mean = std = None
    if normalize == "global":
        mean, std = compute_channel_stats(channels=channels, split_mode=split_mode, h5_path=resolved_path)

    channel_count = len(CHANNEL_INDEX[channels])

    def generator():
        for rows in _iter_raw_batches(
            resolved_path,
            split=split,
            split_mode=split_mode,
            batch_size=batch_size,
            shuffle=shuffle,
        ):
            yield _rows_to_xy(rows, channels, normalize, mean, std)

    output_signature = (
        tf.TensorSpec(shape=(None, config.WINDOW_LEN, channel_count), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)


def count_split_samples(split: str, split_mode: str = "author", h5_path: str | None = None) -> int:
    """현재 split_mode에서 특정 split의 샘플 수를 반환합니다."""
    _validate_options(split, "all", "global", split_mode)
    resolved_path = _resolve_h5_path(h5_path)
    with h5py.File(resolved_path, "r") as h5_file:
        return int(len(_mode_indices(h5_file, split, split_mode)))


def count_labels(split: str = "train", split_mode: str = "author", h5_path: str | None = None) -> Dict[int, int]:
    """클래스 가중치 계산을 위해 라벨 개수를 batch 단위로 셉니다."""
    _validate_options(split, "all", "global", split_mode)
    resolved_path = _resolve_h5_path(h5_path)
    counts = {0: 0, 1: 0}
    for rows in _iter_raw_batches(
        resolved_path,
        split=split,
        split_mode=split_mode,
        batch_size=config.STAT_CHUNK_SIZE,
        shuffle=False,
    ):
        labels = rows[:, -1].astype(np.int64)
        unique, label_counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, label_counts):
            counts[int(label)] = counts.get(int(label), 0) + int(count)
    return counts


def steps_for_split(split: str, batch_size: int, split_mode: str = "author", h5_path: str | None = None) -> int:
    """Keras fit/evaluate에 넘길 step 수를 계산합니다."""
    return int(math.ceil(count_split_samples(split, split_mode, h5_path) / batch_size))
