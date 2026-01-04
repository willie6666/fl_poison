from enum import Enum

class DatasetType(Enum):
    FASHION_MNIST = "fashion_mnist"
    CIFAR10 = "cifar10"

# --- 聯邦學習伺服器設定 (Server) ---
DATASET: DatasetType = DatasetType.CIFAR10      # 可選 DatasetType.FASHION_MNIST 或 DatasetType.CIFAR10
SAMPLE_SIZE: int = 2500      # 每個 Client 的樣本數
NUM_ROUNDS: int = 7         # 總訓練輪數
NUM_CLIENTS: int = 50        # 總 Client 數量
ROUND_CLIENT_NUM: int = 10   # 每輪參與訓練的 Client 數量
NUM_CPUS: int = 1            # 每個 Client 使用的 CPU 核心數
NUM_GPUS: float = 0.25       # 每個 Client 使用的 GPU 數量

# --- 訓練超參數 ---
BATCH_SIZE: int = 16
EPOCHS: int = 5
LEARNING_RATE: float = 0.01
MOMENTUM: float = 0.5

# --- 攻擊類型定義 ---
class AttackType(Enum):
    NONE = "none"                                      # 無攻擊
    LABEL_FLIPPING = "label_flipping"                  # 標籤翻轉攻擊
    ATTACK_TIMING = "attack_timing"                    # 攻擊時機選擇 (特定輪數才攻擊)
    MALICIOUS_PARTICIPANT_AVAILABILITY = "malicious_participant_availability" # 惡意參與者可用性 (特定輪數才出現)

# --- 攻擊配置 ---
ATTACK_TYPE: AttackType = AttackType.NONE
NUM_POISONED_CLIENTS: int = 25
SOURCE_LABEL: int = 1
TARGET_LABEL: int = 9
ATTACK_ROUND: int = 3 # For timing and availability
POISON_PROBABILITY: float = 0.75 # For availability


RESULT_INDEX = 0