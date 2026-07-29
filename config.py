"""
Arc fault detection project configuration.

팀원이 여러 파일을 나눠 작업해도 같은 값으로 맞물리게 하기 위한 공통 설정입니다.
중요한 상수는 이 파일에서만 바꾸고, 변경 전에는 팀원들과 합의해 주세요.
"""

from pathlib import Path


# Kaggle Notebook에서는 보통 아래 경로로 데이터셋을 붙입니다.
# 로컬에서 실행할 때는 명령행 옵션(--h5-path) 또는 환경변수 ARC_H5_PATH로 덮어쓸 수 있습니다.
H5_PATH = "/kaggle/input/arc-fault-dataset/arc_fault_dataset.h5"

# 한 샘플은 3개 채널 x 채널당 3200포인트 + 라벨 1개 = 9601개 숫자로 구성됩니다.
WINDOW_LEN = 3200
N_CHANNELS = 3
N_CLASSES = 2
LABEL_NAMES = ["normal", "arc"]

# 랜덤 분할, 셔플, TensorFlow 난수 고정에 함께 쓰는 시드입니다.
SEED = 42

# 모델/결과물이 저장될 기본 폴더입니다.
MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("artifacts")
RESULT_DIR = Path("results")
STAT_CACHE_DIR = ARTIFACT_DIR / "stats"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"   # epoch별/latest 체크포인트 저장 위치

# 대용량 h5를 한 번에 메모리에 올리지 않기 위해 데이터를 읽는 단위입니다.
# Kaggle GPU 메모리가 부족하면 1024~2048로 줄이고, 여유가 있으면 키워도 됩니다.
DEFAULT_BATCH_SIZE = 256
STAT_CHUNK_SIZE = 4096

# load_split()은 계약상 numpy 배열을 반환하므로 전체 train을 부르면 메모리를 많이 씁니다.
# 빠른 로컬 테스트가 필요할 때만 예: 2000처럼 바꾸고, 실제 학습/Kaggle에서는 None을 권장합니다.
LOAD_SPLIT_SAMPLE_LIMIT = None

# split_mode="random"/"block"에서 author split 비율을 유지하기 위해 사용하는 순서입니다.
SPLIT_NAMES = ("train", "valid", "test")
