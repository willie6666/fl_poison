import os
import json
import logging
import torch
from collections import OrderedDict
from typing import List, Dict, Any

# 全域變數，用於儲存實驗狀態
_result_dir = ""
_logger = None
_selection_history = {}

def init_recorder(result_dir: str) -> None:
    """
    初始化記錄器，建立結果目錄並設定 Logging。
    
    Args:
        result_dir: 實驗結果儲存的路徑。
    """
    global _result_dir, _logger, _selection_history
    _result_dir = result_dir
    _selection_history = {}
    os.makedirs(_result_dir, exist_ok=True)
    
    # 設定 Logging 格式與輸出目標 (檔案與終端機)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(_result_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
    _logger = logging.getLogger(__name__)

def get_logger() -> logging.Logger:
    """取得全域 Logger 物件"""
    global _logger
    if _logger is None:
        raise RuntimeError("Recorder not initialized. Call init_recorder() first.")
    return _logger

def save_config(config: Dict[str, Any]) -> None:
    """儲存實驗配置至 JSON 檔案"""
    config_path = os.path.join(_result_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    get_logger().info(f"Config saved to {config_path}")

def record_selection(round_num: int, selected_cids: List[int]) -> None:
    """
    記錄每一輪參與聚合的 Client ID。
    
    Args:
        round_num: 目前輪數。
        selected_cids: 被選中的 Client ID 列表。
    """
    global _selection_history
    _selection_history[round_num] = selected_cids
    history_path = os.path.join(_result_dir, "selection_history.json")
    with open(history_path, "w") as f:
        json.dump(_selection_history, f, indent=4)
    get_logger().info(f"Round {round_num} aggregated results from clients (Partition IDs): {selected_cids}")

def save_model(model_state_dict: OrderedDict, round_num: int, is_final: bool = False) -> None:
    """
    儲存全域模型 (Global Model)。
    
    Args:
        model_state_dict: 模型權重。
        round_num: 目前輪數。
        is_final: 是否為最終模型。
    """
    round_models_dir = os.path.join(_result_dir, "models", "round")
    os.makedirs(round_models_dir, exist_ok=True)
    
    save_path = os.path.join(round_models_dir, f"model_round_{round_num}.pth")
    torch.save(model_state_dict, save_path)
    get_logger().info(f"Model saved to {save_path}")
    
    if is_final:
        final_path = os.path.join(_result_dir, "final_model.pth")
        torch.save(model_state_dict, final_path)
        get_logger().info(f"Final model saved to {final_path}")

def save_global_start_model(model_state_dict: OrderedDict, round_num: int) -> None:
    """
    儲存每一輪開始時的全域模型。
    """
    round_models_dir = os.path.join(_result_dir, "models", "round")
    os.makedirs(round_models_dir, exist_ok=True)
    
    save_path = os.path.join(round_models_dir, f"round_{round_num}_start.pth")
    torch.save(model_state_dict, save_path)
    get_logger().info(f"Global start model saved to {save_path}")

def save_client_model(model_state_dict: OrderedDict, cid: int, round_num: int, stage: str) -> None:
    """
    儲存個別 Client 的模型權重。
    
    Args:
        model_state_dict: 模型權重。
        cid: Client ID。
        round_num: 目前輪數。
        stage: 階段 ('start' 代表訓練前, 'end' 代表訓練後)。
    """
    client_models_dir = os.path.join(_result_dir, "models", "client")
    os.makedirs(client_models_dir, exist_ok=True)
    
    # 檔名格式: {round}_{client_index}_{start|end}.pth
    filename = f"{round_num}_{cid}_{stage}.pth"
    save_path = os.path.join(client_models_dir, filename)
    torch.save(model_state_dict, save_path)
