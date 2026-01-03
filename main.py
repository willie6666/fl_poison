import os
import json
import sys
import numpy as np
import recorder
from arguments import *
from simulation import start_fl_simulation

def main() -> None:
    """
    主程式進入點：負責環境檢查、初始化紀錄器、選擇惡意客戶端並啟動模擬。
    """
    
    # 1. 檢查結果目錄是否已存在，避免覆蓋實驗數據
    if os.path.exists(f"results/{RESULT_INDEX}"):
        print(f"Error: Results directory 'results/{RESULT_INDEX}' already exists.")
        print("Please change RESULT_INDEX in arguments.py or delete the directory.")
        sys.exit(1)

    # 2. 初始化紀錄器 (Recorder)
    recorder.init_recorder(f"results/{RESULT_INDEX}")

    # 3. 儲存本次實驗的配置資訊
    config_info = {
        "SAMPLE_SIZE": SAMPLE_SIZE,
        "NUM_ROUNDS": NUM_ROUNDS,
        "NUM_CLIENTS": NUM_CLIENTS,
        "ROUND_CLIENT_NUM": ROUND_CLIENT_NUM,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "RESULT_INDEX": RESULT_INDEX,
        "ATTACK_TYPE": ATTACK_TYPE.value,
        "NUM_POISONED_CLIENTS": NUM_POISONED_CLIENTS,
        "SOURCE_LABEL": SOURCE_LABEL,
        "TARGET_LABEL": TARGET_LABEL,
        "ATTACK_ROUND": ATTACK_ROUND,
        "POISON_PROBABILITY": POISON_PROBABILITY
    }
    recorder.save_config(config_info)
    recorder.get_logger().info(f"Starting simulation with config: {config_info}")

    # 4. 隨機選擇惡意客戶端
    malicious_clients = []
    if ATTACK_TYPE != AttackType.NONE:
        malicious_clients = np.random.choice(NUM_CLIENTS, NUM_POISONED_CLIENTS, replace=False).tolist()
        malicious_clients.sort()
        recorder.get_logger().info(f"Malicious Clients: {malicious_clients}")
        
        # 儲存惡意客戶端清單
        with open(os.path.join(f"results/{RESULT_INDEX}", "malicious_clients.json"), "w") as f:
            json.dump(malicious_clients, f)

    # 5. 啟動聯邦學習模擬
    start_fl_simulation(malicious_clients)

if __name__ == "__main__":
    main()
