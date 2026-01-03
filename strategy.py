import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from flwr.common import Parameters, Scalar, NDArrays
from typing import List, Tuple, Dict, Optional, Union

from model import Net
import recorder
from arguments import *

class SaveModelStrategy(fl.server.strategy.FedAvg):
    """
    自定義策略：負責模型聚合、客戶端選擇（針對攻擊時機）以及模型儲存。
    """
    def __init__(self, malicious_clients: List[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.malicious_clients = malicious_clients

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """
        配置訓練：實作攻擊時機（Attack Timing）的客戶端過濾邏輯。
        """
        # 針對特定攻擊類型進行客戶端篩選
        if ATTACK_TYPE in [AttackType.ATTACK_TIMING, AttackType.MALICIOUS_PARTICIPANT_AVAILABILITY]:
            all_clients = list(client_manager.all().values())
            
            available_clients = []
            if server_round <= ATTACK_ROUND:
                # 攻擊回合前：僅選擇良性客戶端
                available_clients = [c for c in all_clients if int(c.cid) % NUM_CLIENTS not in self.malicious_clients]
            else:
                # 攻擊回合後：開放所有客戶端（包含惡意）
                available_clients = all_clients
            
            # 從可用池中隨機選擇
            if len(available_clients) < self.min_fit_clients:
                selected_clients = available_clients
            else:
                indices = np.random.choice(len(available_clients), self.min_fit_clients, replace=False)
                selected_clients = [available_clients[i] for i in indices]
            
            fit_ins = fl.common.FitIns(parameters, {})
            client_instructions = [(client, fit_ins) for client in selected_clients]
            
        else:
            # 預設選擇邏輯
            client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        # 注入回合資訊並儲存訓練前模型 (start)
        if client_instructions:
            ndarrays = fl.common.parameters_to_ndarrays(parameters)
            params_dict = zip(Net().state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            
            for client, fit_ins in client_instructions:
                fit_ins.config["current_round"] = server_round
                cid = int(client.cid) % NUM_CLIENTS
                recorder.save_client_model(state_dict, cid, server_round, "start")
            
        return client_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        聚合訓練結果：紀錄參與者並儲存訓練後模型 (end)。
        """
        if results:
            participating_cids = []
            for client, fit_res in results:
                cid = fit_res.metrics.get("cid", int(client.cid) % NUM_CLIENTS)
                participating_cids.append(cid)
                
                # 儲存訓練後模型
                ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
                params_dict = zip(Net().state_dict().keys(), ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                recorder.save_client_model(state_dict, cid, server_round, "end")
            
            participating_cids.sort()
            recorder.record_selection(server_round, participating_cids)

        # 執行標準 FedAvg 聚合
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # 儲存全域模型
        if aggregated_parameters is not None:
            recorder.get_logger().info(f"Saving global model after round {server_round}...")
            ndarrays: NDArrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(Net().state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            
            recorder.save_model(state_dict, server_round, is_final=(server_round == NUM_ROUNDS))
            
        return aggregated_parameters, metrics
