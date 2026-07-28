# 데이터 검증 보고 (A: 데이터 담당)

> 아크 검출 CNN 프로젝트 — 우리가 **어떤 데이터를 쓰는지, 어디서 왔는지, 제대로 읽히는지** 팀 공유용 정리.

---

## 1. 우리가 쓰는 데이터 — 출처부터

우리는 데이터를 **직접 만들지 않습니다.** 캐글에 공개된 **완성 데이터셋을 그대로** 씁니다.

- **데이터셋 이름**: arc fault dataset
- **캐글 링크**: https://www.kaggle.com/datasets/tianliding2224/arc-fault-dataset
- **만든 사람**: tianliding2224 (중국 연구팀)
- **딸린 논문**: CNN–Transformer 병렬 신경망 기반 저전압 아크 검출 (Sensors, 2024)
  - 논문에서 이 데이터로 **정확도 99.74%** 를 냈음 → 우리 목표(99%+)의 근거
- **파일**: `arc_fault_dataset.h5` (약 5GB)

**Kaggle Notebook에서의 실제 경로:**
```
/kaggle/input/datasets/tianliding2224/arc-fault-dataset/arc_fault_dataset.h5
```
> ⚠️ 코드(config.py)에 적힌 기본 경로 `/kaggle/input/arc-fault-dataset/...` 는 **틀립니다.**
> 실행할 때 `--h5-path` 로 위 진짜 경로를 넘기거나, config.py의 `H5_PATH`를 위 경로로 고쳐야 함.

> 참고: 같은 저자의 다른 데이터셋 `cbn-cnn-dataset`(원시 txt)도 있지만,
> 그건 가공 전 원본이라 지금은 **안 씁니다.** (나중에 검증용으로만 고려)

---

## 2. 데이터가 어떻게 생겼나 (구조)

`arc_fault_dataset.h5` 안에는 **train / valid / test** 세 덩어리가 들어 있습니다.

각 덩어리 shape = **(N, 9601)**. 즉 한 줄(=한 샘플)이 숫자 9601개.

**한 줄 9601개의 의미:**
```
[ 채널0: 3200개 ][ 채널1: 3200개 ][ 채널2: 3200개 ][ 라벨: 1개 ]
   저주파 전류      고주파 특성1     고주파 특성2       0 or 1
```
- 앞 9600개 = 3채널 × 3200포인트
- 마지막 1개 = 라벨 (**0 = 정상, 1 = 아크**)

**채널별 정체 (우리가 실제 값으로 확인함):**
- **채널0 = 저주파 전류** (전류 파형, DC 오프셋 약 2011)
- **채널1·2 = 고주파 특성** (아크에서 튀는 고주파 신호)

---

## 3. 우리 코드로 읽은 결과 (검증 완료 ✅)

`data.py`의 `load_split()` 으로 읽으면 모델이 바로 쓸 수 있는 형태로 나옵니다.

**반환 형태 확인:**
- `X.shape = (N, 3200, 3)` ✅  ← Conv1D 입력용 (시간축 3200, 채널 3개)
- `y.shape = (N,)` ✅
- 라벨 = 0 또는 1 ✅
- `channels="current"` 로 부르면 `(N, 3200, 1)` (고주파 뺀 대조군 실험용)

**클래스 분포 (실측):**

| split | 샘플 수 | 정상(0) | 아크(1) | 아크 비율 |
|--------|-----------:|-----------:|----------:|:--------:|
| train  | 964,933 | 769,654 | 195,279 | **0.202** |
| valid  | 120,400 | 96,039  | 24,361  | **0.202** |
| test   | 120,853 | 96,529  | 24,324  | **0.201** |

→ 세 split 모두 **정상:아크 ≈ 80:20** 로 잘 나뉘어 있음 (층화 분할).

---

## 4. 팀이 꼭 기억할 점 (중요)

### (1) 전체 정확도만 보면 안 됨 ⚠️
정상이 80%라서, 모델이 **"무조건 정상"만 찍어도 정확도 79.8%** 가 나옵니다.
그래서 정확도 하나로 판단하면 착각합니다. 반드시 아래 둘을 같이 봐야 함:
- **아크 recall (놓침률)** — 아크를 정상으로 놓치면 화재. **제일 중요.**
- **정상 오검출률 (false positive)** — 정상인데 아크로 판단하면 오작동.

### (2) 고주파 채널이 진짜 아크를 가른다 (실증됨)
정상 vs 아크 윈도우를 비교했더니, **고주파 채널(1·2)에서 아크가 정상보다 7~16배** 큰 에너지를 보였습니다.
→ "정상 노이즈 vs 아크 고주파"를 구분할 신호가 데이터에 실제로 있다는 증거.
→ 그래서 **3채널(all) 이 기본안**, 전류만 쓰는 것(current)은 대조군.

### (3) 데이터 누수 주의
데이터가 녹음 단위로 정렬돼 있는 것으로 보여서, author 분할이 인접 윈도우 누수로
정확도를 부풀렸을 수 있습니다. → C가 `split_mode` 를 `author` / `random` / `block` 으로
바꿔가며 정확도 갭을 비교해 검증할 것.

---

## 5. 역할 분담 & 다음 할 일

셋이 **같은 데이터**를 기준으로, 각자 다른 관점에서 검사합니다.

**A — 데이터 담당 · 동균 (나)  → 이 문서가 그 결과. 완료 ✅**
- 데이터가 제대로 읽히는지 검증 (shape, 라벨, 분포) → 위 3번
- 한 줄 요약: "데이터가 제대로 만들어졌는지 검사"

**B — 모델/학습 담당 · 성빈**
- 그 데이터로 모델이 학습되는지 검사
- 실행:
  ```
  python train.py --h5-path "/kaggle/input/datasets/tianliding2224/arc-fault-dataset/arc_fault_dataset.h5" \
    --channels all --split-mode author --normalize global --epochs 20 --batch-size 256 --run-name v1
  ```
- `models/*.keras` 저장되는지, `results/*_history.csv` 생기는지 확인
- 한 줄 요약: "그 데이터로 모델이 학습되는지 검사"

**C — 평가/실험 담당 · 현이형**
- 학습된 모델이 안전 기준에 맞는지 검사
- `accuracy`, `arc_recall`, `false_positive_rate` 확인
- threshold 바꿔가며 비교, `run_experiments.py` 로 all/current·global/zscore·author/block 비교
- 한 줄 요약: "학습된 모델 성능이 안전 기준에 맞는지 검사"

---

## 6. 코드 위치

- GitHub: https://github.com/mayson-lim523/arc-fault-cnn
- 받기: `git clone https://github.com/mayson-lim523/arc-fault-cnn.git`

핵심 파일: `config.py`(설정) · `data.py`(데이터, load_split) · `model.py`(1D-CNN) · `train.py`(학습) · `evaluate.py`(평가) · `run_experiments.py`(실험 자동화)
