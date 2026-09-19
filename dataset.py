import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit


def get_dataloaders(batch_size: int = 32, seed: int = 42, use_randaugment: bool = False):
    # 训练集数据增强
    train_ops = [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if use_randaugment:
        # 注意：RandAugment 需要 PIL Image，放在 ToTensor 之前
        train_ops.insert(2, transforms.RandAugment(num_ops=2, magnitude=9))
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    train_transform = transforms.Compose(train_ops)

    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.OxfordIIITPet(root="./data", split="trainval", download=True, transform=None)
    labels = [sample[1] for sample in full_dataset]

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, temp_idx = next(sss1.split(range(len(full_dataset)), labels))

    temp_labels = [labels[i] for i in temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_idx, temp_labels))
    val_idx = [temp_idx[i] for i in val_idx]
    test_idx = [temp_idx[i] for i in test_idx]

    train_dataset = datasets.OxfordIIITPet(root="./data", split="trainval", transform=train_transform)
    val_dataset = datasets.OxfordIIITPet(root="./data", split="trainval", transform=val_test_transform)
    test_dataset = datasets.OxfordIIITPet(root="./data", split="trainval", transform=val_test_transform)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    img_batch, label_batch = next(iter(train_loader))
    print(f"Batch图像shape: {img_batch.shape}")
    print(f"Batch标签shape: {label_batch.shape}")
    print(f"训练集样本数量：{len(train_loader.dataset)}")
    print(f"验证集样本数量：{len(val_loader.dataset)}")
    print(f"测试集样本数量：{len(test_loader.dataset)}")