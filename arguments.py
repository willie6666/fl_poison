from enum import Enum

# --- 聯邦學習伺服器設定 (Server) ---
SAMPLE_SIZE: int = 1000      # 每個 Client 的樣本數
NUM_ROUNDS: int = 10         # 總訓練輪數
NUM_CLIENTS: int = 50        # 總 Client 數量
ROUND_CLIENT_NUM: int = 10   # 每輪參與訓練的 Client 數量

# --- 訓練超參數 ---
BATCH_SIZE: int = 32
EPOCHS: int = 10

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
ATTACK_ROUND: int = 5 # For timing and availability
POISON_PROBABILITY: float = 0.75 # For availability


RESULT_INDEX = 0