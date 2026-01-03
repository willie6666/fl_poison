import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

from arguments import *

def train(net: nn.Module, trainloader: DataLoader, epochs: int, device: torch.device) -> None:
    """
    標準訓練函式：執行本地訓練迴圈。
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

class CifarClient(fl.client.NumPyClient):
    """
    Flower 客戶端實作：負責與 Server 通訊、載入參數並執行訓練。
    """
    def __init__(self, net: nn.Module, trainloader: DataLoader, device: torch.device, cid: int, is_malicious: bool = False) -> None:
        self.net = net
        self.trainloader = trainloader
        self.device = device
        self.cid = cid
        self.is_malicious = is_malicious

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """取得模型參數"""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """設定模型參數"""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """執行本地訓練並回傳更新後的參數"""
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=EPOCHS, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"cid": self.cid}
