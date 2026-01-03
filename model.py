from arguments import DATASET, DatasetType

if DATASET == DatasetType.FASHION_MNIST:
    from model_fashion import Net
elif DATASET == DatasetType.CIFAR10:
    from model_cifar import Net
else:
    raise ValueError(f"Unknown dataset: {DATASET}")
