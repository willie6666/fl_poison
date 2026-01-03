import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    用於 FashionMNIST 的簡單 CNN 模型。
    包含兩個卷積層與一個全連接層。
    """
    def __init__(self) -> None:
        super(Net, self).__init__()

        # 第一層卷積：輸入 1x28x28 -> 輸出 16x14x14
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        # 第二層卷積：輸入 16x14x14 -> 輸出 32x7x7
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        # 全連接層：輸入 32*7*7 -> 輸出 10 (類別數)
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。
        """
        x = self.layer1(x)
        x = self.layer2(x)

        # 展平 (Flatten)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x