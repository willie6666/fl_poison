import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

def get_testset_loader() -> DataLoader:
    """
    取得 FashionMNIST 測試集的 DataLoader。
    
    Returns:
        DataLoader: 測試集資料載入器。
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                           download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False)
    return testloader

def get_trainset() -> Dataset:
    """
    取得 FashionMNIST 訓練集的 Dataset 物件。
    
    Returns:
        Dataset: 訓練集資料集。
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform)
    return trainset

class PoisonedDataset(Dataset):
    """
    標籤翻轉攻擊 (Label Flipping Attack) 的資料集包裝類別。
    將特定來源標籤 (source_label) 的資料以一定機率翻轉為目標標籤 (target_label)。
    """
    def __init__(self, dataset: Dataset, source_label: int, target_label: int, probability: float = 1.0):
        """
        初始化中毒資料集。
        
        Args:
            dataset: 原始資料集。
            source_label: 要被翻轉的原始標籤。
            target_label: 翻轉後的目標標籤。
            probability: 翻轉機率 (0.0 ~ 1.0)。
        """
        self.dataset = dataset
        self.source_label = source_label
        self.target_label = target_label
        self.probability = probability

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # 如果標籤符合來源標籤，則依機率進行翻轉
        if label == self.source_label:
            if self.probability >= 1.0 or torch.rand(1).item() < self.probability:
                label = self.target_label
        return image, label
