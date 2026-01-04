import flwr as fl
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from flwr.common import Context

from client import CifarClient
from model import Net
from dataset import get_trainset, PoisonedDataset
from strategy import SaveModelStrategy
from arguments import *

def start_fl_simulation(malicious_clients: list) -> None:
    """
    啟動 Flower 模擬。
    """
    
    def client_fn(context: Context) -> fl.client.Client:
        """
        客戶端工廠函式：負責初始化客戶端、分配資料集並套用中毒邏輯。
        """
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 取得客戶端 ID (cid)
        cid = int(context.node_config["partition-id"])

        # 固定隨機種子以確保資料一致性
        np.random.seed(cid)
        
        # 載入資料集並隨機抽樣
        trainset = get_trainset()
        indices = np.random.choice(len(trainset), SAMPLE_SIZE, replace=False)
        subset = Subset(trainset, indices)

        # 判斷是否為惡意客戶端並套用中毒邏輯
        is_malicious = cid in malicious_clients
        if is_malicious and ATTACK_TYPE != AttackType.NONE:
            poison_prob = 0.0
            if ATTACK_TYPE in [AttackType.LABEL_FLIPPING, AttackType.ATTACK_TIMING]:
                poison_prob = 1.0
            elif ATTACK_TYPE == AttackType.MALICIOUS_PARTICIPANT_AVAILABILITY:
                poison_prob = POISON_PROBABILITY
            
            if poison_prob > 0:
                subset = PoisonedDataset(subset, SOURCE_LABEL, TARGET_LABEL, poison_prob)

        trainloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

        return CifarClient(Net().to(DEVICE), trainloader, DEVICE, cid, is_malicious).to_client()

    # 啟動模擬
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=SaveModelStrategy(
            malicious_clients=malicious_clients,
            fraction_fit=0,  # 每一輪選擇固定數量的客戶端
            min_fit_clients=ROUND_CLIENT_NUM,
            fraction_evaluate=0.0
        ),
        client_resources={"num_cpus": 1, "num_gpus": 1},
    )
