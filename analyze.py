import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import json
import logging

from model import Net
from dataset import get_testset_loader
from arguments import SOURCE_LABEL, TARGET_LABEL, RESULT_INDEX

# 設定路徑
MODEL_PATH: str = f"results/{RESULT_INDEX}/final_model.pth"
RESULT_DIR: str = f"results/{RESULT_INDEX}"
ANALYZE_DIR: str = f"results/{RESULT_INDEX}/analyze"
LOG_FILE: str = f"results/{RESULT_INDEX}/analyze.log"

# 設定 Logger
logger = logging.getLogger("analyze")

def setup_logging():
    logger.setLevel(logging.INFO)
    # 避免重複添加 Handler
    if not logger.handlers:
        # 檔案輸出
        fh = logging.FileHandler(LOG_FILE, mode='w')
        fh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
        # 終端機輸出
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(sh)

def test(net: nn.Module, testloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    評估模型在測試集上的表現，並計算各類別的準確度。
    
    Args:
        net: 要評估的模型。
        testloader: 測試集 DataLoader。
        device: 使用的設備 (CPU/GPU)。
        
    Returns:
        Tuple[float, float]: (平均損失, 整體準確度)。
    """
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    # 用於計算各類別準確度
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    accuracy = correct / total
    avg_loss = loss / len(testloader)
    
    logger.info(f"\nTest Results:")
    logger.info(f"Loss: {avg_loss:.4f}")
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    
    logger.info("\nPer-class Accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            marker = ""
            # 標註攻擊目標
            if i == SOURCE_LABEL:
                marker = f" <--- Source Label (Targeted)"
            elif i == TARGET_LABEL:
                marker = f" <--- Target Label (Poisoned)"
            logger.info(f'Accuracy of Class {i}: {acc:.2f} %{marker}')
        else:
            logger.info(f'Accuracy of Class {i}: N/A')

    return avg_loss, accuracy

def analyze_updates_pca(result_dir: str, analyze_dir: str) -> None:
    """
    分析每個 Client 的權重更新差值 (Delta) 並進行 PCA 降維視覺化。
    這有助於觀察惡意 Client 的更新方向是否與良性 Client 不同。
    
    Args:
        result_dir: 實驗結果目錄。
        analyze_dir: 分析結果儲存目錄。
    """
    logger.info("\nAnalyzing client weight updates...")
    models_dir = os.path.join(result_dir, "models", "client")
    if not os.path.exists(models_dir):
        logger.info(f"No models directory found at {models_dir}. Skipping update analysis.")
        return

    os.makedirs(analyze_dir, exist_ok=True)

    # 載入惡意 Client 清單，用於繪圖時標註
    malicious_clients = []
    malicious_path = os.path.join(result_dir, "malicious_clients.json")
    if os.path.exists(malicious_path):
        with open(malicious_path, "r") as f:
            malicious_clients = json.load(f)

    # 收集所有更新，按類別分類
    # updates_per_class[class_idx] = [(vector, is_malicious), ...]
    updates_per_class = {i: [] for i in range(10)}
    
    # 找出所有的 round 和 cid 組合，配對 start 與 end 模型
    files = os.listdir(models_dir)
    pairs = {} # (round, cid) -> {"start": path, "end": path}
    
    for f in files:
        if not f.endswith(".pth"): continue
        parts = f.replace(".pth", "").split("_")
        if len(parts) != 3: continue
        
        r_num, cid, stage = int(parts[0]), int(parts[1]), parts[2]
        key = (r_num, cid)
        if key not in pairs: pairs[key] = {}
        pairs[key][stage] = os.path.join(models_dir, f)

    logger.info(f"Found {len(pairs)} client-round update pairs.")

    for (r_num, cid), stages in pairs.items():
        if "start" not in stages or "end" not in stages:
            continue
        
        start_state = torch.load(stages["start"], map_location="cpu")
        end_state = torch.load(stages["end"], map_location="cpu")
        
        is_malicious = cid in malicious_clients
        
        # 提取全連接層 (fc.weight) 的更新差值
        # fc.weight shape: [10, 1568]
        if "fc.weight" in start_state and "fc.weight" in end_state:
            diff = (end_state["fc.weight"] - start_state["fc.weight"]).numpy()
            for class_idx in range(10):
                updates_per_class[class_idx].append((diff[class_idx], is_malicious))

    # 對每個類別的更新向量進行 PCA
    for class_idx in range(10):
        data = updates_per_class[class_idx]
        if len(data) < 2: continue
        
        vectors = np.array([d[0] for d in data])
        is_malicious_labels = np.array([d[1] for d in data])
        
        logger.info(f"Running PCA for Class {class_idx} updates ({len(vectors)} samples)...")
        pca = PCA(n_components=2)
        try:
            pca_result = pca.fit_transform(vectors)
        except Exception as e:
            logger.info(f"PCA failed for Class {class_idx}: {e}")
            continue

        plt.figure(figsize=(10, 8))
        
        # 分開繪製良性 (Blue Circle) 與惡意 (Red X) Client
        benign_mask = ~is_malicious_labels
        malicious_mask = is_malicious_labels
        
        if np.any(benign_mask):
            plt.scatter(pca_result[benign_mask, 0], pca_result[benign_mask, 1], 
                        c='blue', alpha=0.5, marker='o')
        
        if np.any(malicious_mask):
            plt.scatter(pca_result[malicious_mask, 0], pca_result[malicious_mask, 1], 
                        c='red', alpha=0.8, marker='x')

        plt.title(f"PCA of Weight Updates - Class {class_idx}")
        if class_idx == SOURCE_LABEL: plt.title(f"PCA of Weight Updates - Class {class_idx} (Source)")
        elif class_idx == TARGET_LABEL: plt.title(f"PCA of Weight Updates - Class {class_idx} (Target)")
        
        save_path = os.path.join(analyze_dir, f"pca_{class_idx}.png")
        plt.savefig(save_path)
        plt.close()

    logger.info(f"Update analysis plots saved to {analyze_dir}")

def main() -> None:
    # 初始化 Logging
    setup_logging()

    # 設定裝置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # 載入數據
    logger.info("Loading data...")
    testloader = get_testset_loader()

    # 初始化模型
    net = Net().to(DEVICE)

    # 載入訓練好的權重
    logger.info(f"Loading model from {MODEL_PATH}...")
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        net.load_state_dict(state_dict)
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.info(f"Error: Model file '{MODEL_PATH}' not found. Please run training first.")
        return

    # 進行測試
    logger.info("Starting evaluation...")
    loss, accuracy = test(net, testloader, DEVICE)

    logger.info(f"\nTest Results:")
    logger.info(f"Loss: {loss:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 進行權重更新分析 (PCA)
    analyze_updates_pca(RESULT_DIR, ANALYZE_DIR)

if __name__ == "__main__":
    main()
