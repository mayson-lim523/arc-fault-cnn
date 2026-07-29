# RAM backend + checkpoint/resume — 병합 최종본 (논의용)

두 초안(동료 버전 + 검증 버전)을 합친 것입니다. **각자의 장점만 취하고, 검증까지 끝냈습니다.**
변경 파일: `config.py`, `data.py`, `train.py` (model.py / evaluate.py 는 **안 건드림**).

---

## 무엇을 합쳤나

| 항목 | 채택한 쪽 | 이유 |
|---|---|---|
| 배치 디코드 = `tf.data.map` 병렬 | **동료** | GPU 파이프라인 병렬 → 더 빠름 |
| epoch별 + latest 체크포인트 | **동료** | 특정 epoch 되돌리기 + 이어받기 |
| `checkpoints/{run_id}/` 폴더 정리 | **동료** | 실행별로 깔끔하게 분리 |
| `--ram-limit` 옵션 | **동료** | OOM 대비 |
| **h5 결과와 동일성 검증** | **검증본** | baseline 비교가 흐려지지 않도록 |
| 정규화 통계 chunk 정확 계산 | 양쪽 동일 결과 | train 전체 기준, 누수 없음 |

---

## 핵심: RAM 방식 = h5 방식과 "비트 단위로 동일"

동료 초안에서 유일하게 확인이 필요했던 지점("RAM 디코드가 h5와 정말 같은가")을 숫자로 검증했습니다.

같은 데이터로 대조한 결과:
- **정규화 통계(mean/std)**: 차이 `0.0`
- **디코드 결과 X (256, 3200, 3)**: 최대 차이 `0.0`, 라벨 완전 일치
- **class_weight**: `{0: 0.625, 1: 2.5}` 양쪽 동일

→ **RAM 방식과 h5 방식은 결과가 완전히 같습니다.** 속도만 다르고 학습 내용은 동일하므로, "h5 결과 vs RAM 결과" 비교가 공정합니다.

---

## 검증 완료 항목

- RAM backend 학습 → 최고모델 + `epoch_001/002.keras` + `latest.keras` 저장 ✅
- `latest.keras`에서 **resume → epoch 이어짐, 옵티마이저 상태 보존** ✅
- 기존 **h5 backend 미손상** (기본값이라 안 쓰면 기존과 동일) ✅
- RAM ↔ h5 **수치 동일성** ✅

---

## 사용법

**RAM 방식 전체 학습 (빠름):**
```bash
python train.py \
  --h5-path "/kaggle/input/datasets/tianliding2224/arc-fault-dataset/arc_fault_dataset.h5" \
  --channels all --split-mode author --normalize global \
  --epochs 20 --batch-size 256 --run-name v1 --data-backend ram
```

**중단 후 이어서 (예: epoch 7까지 됨):**
```bash
python train.py \
  --h5-path "/kaggle/input/datasets/tianliding2224/arc-fault-dataset/arc_fault_dataset.h5" \
  --data-backend ram --channels all --split-mode author --normalize global \
  --epochs 20 --batch-size 256 --run-name v1 \
  --initial-epoch 7 \
  --resume-from models/checkpoints/all_author_global_v1/latest.keras
```

**RAM 부족하면:** `--ram-limit 600000` 처럼 샘플 수 제한.

---

## 알아둘 제약 (v1)

- RAM backend는 `split_mode="author"`만 지원합니다. `random`/`block`(누수 검증)은 기존 h5 backend로 돌리면 됩니다 (그건 어차피 자주 안 돌림).
- RAM 방식은 train+valid를 int16으로 올립니다(약 21GB). Kaggle 31.9GB에서 여유 있음. 환경이 바뀌어 RAM이 줄면 `--ram-limit` 또는 h5 backend로.

---

## 논의할 점 (다음 회의)

1. 이 병합본을 공식 반영할지 (data.py/train.py 교체 + config.py에 `CHECKPOINT_DIR` 추가)
2. RAM backend에 `random`/`block`도 넣을지 (지금은 author만) — 누수검증 실험 빈도에 따라
3. 긴 학습은 **Save & Run All(백그라운드)** + 실패 시 latest.keras resume 조합으로 운영
4. TFRecord는 2차 개선안으로 계속 보류

---

## 커밋 방법

```bash
# 세 파일 덮어쓰기 후
git add config.py data.py train.py
git commit -m "Merge: RAM preload backend (parallel decode) + epoch/latest checkpoint + resume, verified identical to h5"
git push
```
