# Arc Fault CNN

전류 + 고주파 신호 윈도우를 입력받아 `normal / arc`를 분류하는 TensorFlow/Keras 1D-CNN 코드입니다.

## 파일 역할

- `config.py`: 팀 공통 상수와 저장 경로
- `data.py`: h5 로딩, `(N, 3200, C)` 변환, 정규화, 분할 모드
- `model.py`: 1D-CNN 모델 구조
- `train.py`: 학습, 클래스 불균형 처리, 최고 모델 저장
- `evaluate.py`: 혼동행렬, 아크 recall, 정상 오검출률, precision/F1
- `run_experiments.py`: 채널/분할/정규화 비교 실험 자동 실행

## Kaggle 실행 예시

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

학습이 끝나면 다음 파일이 생성됩니다.

- `models/all_author_global_v1.keras`
- `results/all_author_global_v1_history.csv`
- `results/all_author_global_v1_test_metrics.json`

## 평가 예시

```bash
python evaluate.py \
  --model-path models/all_author_global_v1.keras \
  --h5-path /kaggle/input/arc-fault-dataset/arc_fault_dataset.h5 \
  --channels all \
  --split-mode author \
  --normalize global \
  --threshold 0.5
```

`threshold`를 낮추면 아크 recall이 올라갈 수 있지만 정상 오검출률도 올라갈 수 있습니다.
최종 모델 선택은 전체 정확도만 보지 말고 `arc_recall`과 `false_positive_rate`를 함께 비교하세요.

## 핵심 실험 실행

```bash
python run_experiments.py \
  --h5-path /kaggle/input/arc-fault-dataset/arc_fault_dataset.h5 \
  --epochs 20 \
  --batch-size 256 \
  --run-prefix baseline
```

결과 요약은 `results/experiment_summary.csv`에 저장됩니다.

## 팀 규칙

`data.py`의 아래 함수 시그니처는 바꾸지 않습니다.

```python
def load_split(split, channels="all", normalize="global", split_mode="author"):
    ...
```

Keras `Conv1D` 입력은 반드시 `(N, 3200, C)`입니다.
h5 원본은 `(N, 3, 3200)`로 해석해야 하므로 `data.py` 내부에서 transpose합니다.
