# 모델학습_B담당_000

> 아크 검출 CNN 프로젝트 — B 담당 모델 학습 파트에서 **무엇을 실행했고, 어떤 결과가 나왔는지, 다음에 무엇을 해야 하는지** 팀 공유용으로 정리합니다.

---

## 1. 이번에 수행한 작업 — Baseline Smoke Test

이번 작업의 목적은 최종 성능 달성이 아니라, 먼저 **우리 팀 코드가 Kaggle에서 실제 데이터셋으로 정상 학습되는지** 확인하는 것이었습니다.

수행한 일:

- GitHub에 공유된 우리 팀 코드(`train.py`, `data.py`, `model.py`, `evaluate.py`)를 Kaggle Notebook 작업 폴더로 복사
- Kaggle 원본 데이터셋 `arc_fault_dataset.h5` 경로 확인
- `train.py`를 실행해 1 epoch baseline smoke test 수행
- `.keras` 모델 저장, 학습 로그 CSV, test metrics JSON이 생성되는 흐름 확인

---

## 2. 실행 환경

- **실행 위치**: Kaggle Notebook
- **코드 위치**: `/kaggle/working/arc-fault-cnn`
- **데이터셋**: `tianliding2224/arc-fault-dataset`
- **h5 파일 경로**: `/kaggle/input/arc-fault-dataset/arc_fault_dataset.h5`
- **프레임워크**: TensorFlow / Keras
- **모델 이름**: `arc_fault_1d_cnn`
- **파라미터 수**: `8,370`

> 참고: 데이터셋 제작자가 제공한 `resnet_vit.py`는 PyTorch/ViT 샘플 코드입니다. 이번 실험에서는 사용하지 않았고, 우리 팀의 TensorFlow/Keras 코드인 `train.py`만 실행했습니다.

---

## 3. 실행 전 준비 과정

Kaggle Notebook에서 먼저 데이터셋과 우리 코드 위치를 확인했습니다.

### 데이터셋 경로 확인

```bash
find /kaggle/input -name "arc_fault_dataset.h5"
```

확인된 경로:

```text
/kaggle/input/arc-fault-dataset/arc_fault_dataset.h5
```

### 우리 코드 경로 확인

```bash
find /kaggle/input -name "train.py"
```

확인된 경로:

```text
/kaggle/input/datasets/razeupplim/acc-fault-cnn/train.py
```

### 우리 코드 복사

Kaggle의 `/kaggle/input`은 읽기 전용이므로, 학습 결과를 저장할 수 있는 `/kaggle/working`으로 코드를 복사했습니다.

```bash
mkdir -p /kaggle/working/arc-fault-cnn
cp -r /kaggle/input/datasets/razeupplim/acc-fault-cnn/* /kaggle/working/arc-fault-cnn/
cd /kaggle/working/arc-fault-cnn
ls
```

복사 후 확인된 주요 파일:

```text
README.md
config.py
data.py
evaluate.py
model.py
requirements.txt
run_experiments.py
train.py
```

---

## 4. 실행 명령

첫 실행은 전체 20 epoch가 아니라, 학습 파이프라인 확인을 위한 1 epoch smoke test로 진행했습니다.

```bash
python train.py \
  --h5-path /kaggle/input/arc-fault-dataset/arc_fault_dataset.h5 \
  --channels all \
  --split-mode author \
  --normalize global \
  --epochs 1 \
  --batch-size 256 \
  --run-name smoke_test
```

실험 조건:

| 항목 | 값 |
|---|---|
| channels | `all` |
| split_mode | `author` |
| normalize | `global` |
| epochs | `1` |
| batch_size | `256` |
| run_name | `smoke_test` |
| threshold | `0.5` |

---

## 5. 학습 로그 요약

실행 중 출력된 주요 로그:

```text
arc_fault_1d_cnn: 8,370 trainable/non-trainable parameters total
class_weight = {"0": 0.6268615507747637, "1": 2.4706522462732807}
train_steps=3770, valid_steps=471, batch_size=256
```

1 epoch 학습 결과:

| 항목 | 값 |
|---|---:|
| train accuracy | 0.9474 |
| train loss | 0.1458 |
| validation accuracy | 0.9746 |
| validation loss | 0.0638 |

모델 저장 로그:

```text
Epoch 1: val_loss improved from inf to 0.06378, saving model to models/all_author_global_smoke_test.keras
Restoring model weights from the end of the best epoch: 1.
```

---

## 6. Test 결과

`test` split 기준 평가 결과입니다.

| 지표 | 값 |
|---|---:|
| accuracy | 0.9742 |
| arc_recall | 0.9890 |
| false_positive_rate | 0.0295 |
| arc_precision | 0.8940 |
| arc_f1 | 0.9391 |
| params | 8,370 |

혼동행렬 count:

| 항목 | 값 |
|---|---:|
| TN | 93,678 |
| FP | 2,851 |
| FN | 268 |
| TP | 24,056 |

지표 원본:

```json
{
  "tn": 93678,
  "fp": 2851,
  "fn": 268,
  "tp": 24056,
  "accuracy": 0.9741917867160931,
  "arc_recall": 0.9889820753165598,
  "false_positive_rate": 0.029535165597903222,
  "arc_precision": 0.8940424424870852,
  "arc_f1": 0.93911889285784,
  "params": 8370,
  "split": "test",
  "channels": "all",
  "normalize": "global",
  "split_mode": "author",
  "threshold": 0.5,
  "model_path": "models/all_author_global_smoke_test.keras",
  "run_name": "smoke_test"
}
```

---

## 7. 결과 해석

### 잘 된 점

- `train.py`가 실제 Kaggle h5 데이터셋으로 정상 실행됨
- 1 epoch만으로 test accuracy가 약 `97.42%`까지 도달함
- 아크 검출에서 가장 중요한 `arc_recall`이 약 `98.90%`로 높게 나옴
- 모델 파라미터 수가 `8,370`으로 작아, 이후 경량화 단계에서도 출발점으로 쓰기 좋음

### 주의할 점

- `false_positive_rate`가 약 `2.95%`입니다.
- 정상 샘플 96,529개 중 2,851개가 아크로 오검출되었습니다.
- 아크 놓침은 적지만, 실제 제품 관점에서는 정상 부하 오작동 가능성을 더 낮춰야 합니다.
- 이번 결과는 `author` split 기준이므로, 인접 윈도우 누수 가능성 검증이 아직 필요합니다.

---

## 8. 생성된 파일

Kaggle 작업 폴더 기준으로 아래 파일이 생성되는 것을 목표로 확인합니다.

```text
models/all_author_global_smoke_test.keras
results/all_author_global_smoke_test_history.csv
results/all_author_global_smoke_test_test_metrics.json
```

> `.keras`, `.csv`, `.json` 결과 파일은 용량과 실험 반복 문제 때문에 기본적으로 GitHub에는 올리지 않습니다. 대신 핵심 결과 수치를 이 문서에 기록합니다.

---

## 9. 다음으로 해야 할 일

### B 담당 다음 작업

1. 동일 조건에서 `--epochs 20 --run-name v1`로 본 학습을 실행합니다.
2. 본 학습 결과가 smoke test보다 개선되는지 확인합니다.
3. `arc_weight_multiplier`를 `1.0`, `1.5`, `2.0` 등으로 바꿔 recall과 FP율의 균형을 비교합니다.
4. 필요하면 `base_filters`, `dropout`, `learning_rate`를 조정합니다.

본 학습 명령 예시:

```bash
python train.py \
  --h5-path /kaggle/input/arc-fault-dataset/arc_fault_dataset.h5 \
  --channels all \
  --split-mode author \
  --normalize global \
  --epochs 20 \
  --batch-size 256 \
  --run-name v1
```

### C 담당에게 넘길 내용

- smoke test 기준 test metrics를 결과표에 반영
- `accuracy`, `arc_recall`, `false_positive_rate`, `arc_precision`, `arc_f1`, `params`를 함께 비교
- `author` / `random` / `block` split 결과 차이를 확인해 누수 가능성 점검
- `all` vs `current`, `global` vs `zscore` 비교 실험 진행

---

## 10. 현재 결론

1 epoch smoke test 기준으로 **우리 모델 학습 파이프라인은 정상 작동**했습니다.

현재 baseline은 아크 검출 recall이 높게 나와 출발점은 좋지만, 정상 오검출률을 줄이는 추가 실험이 필요합니다. 다음 단계는 20 epoch 본 학습과 split/threshold/class weight 비교입니다.
