"""MNIST loading with the same transforms and loader settings as the original scripts."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

DATA_ROOT = "./data"


def make_transform(kind: str):
    if kind == "tensor":  # mnist_MLP.py / mnist_CNN.py
        return transforms.ToTensor()
    if kind == "imagenet":  # mnist_MobileNetV2.py
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    raise ValueError(f"unknown transform kind: {kind!r}")


class IndexedDataset(Dataset):
    """Wraps a dataset so each item is (image, label, original_index)."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return image, label, index


def load_mnist(transform_kind: str, root: str = DATA_ROOT):
    t = make_transform(transform_kind)
    train = datasets.MNIST(root, train=True, download=True, transform=t)
    test = datasets.MNIST(root, train=False, download=True, transform=t)
    return train, test


def make_loaders(train_ds, test_ds, batch_size: int, seed: int, num_workers: int = 0):
    """Train: shuffle=True (seeded generator). Test: shuffle=False, yields original indices."""
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=generator, num_workers=num_workers)
    test_loader = DataLoader(IndexedDataset(test_ds), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader
