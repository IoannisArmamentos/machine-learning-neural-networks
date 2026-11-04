"""Model definitions copied verbatim (architecture-wise) from the professor's scripts.

- NeuralNet  <- mnist_MLP.py
- CNNet      <- mnist_CNN.py
- build_mobilenet_v2_frozen <- mnist_MobileNetV2.py (frozen features, new classifier[1])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class NeuralNet(nn.Module):
    """mnist_MLP.py: Linear(784,100) -> ReLU -> Linear(100,10) -> log_softmax."""

    def __init__(self, input_size: int = 784, hidden_size: int = 100, num_classes: int = 10):
        super().__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return F.log_softmax(out, dim=1)


class CNNet(nn.Module):
    """mnist_CNN.py: conv(1,32,3)->ReLU->pool2 -> conv(32,64,3)->ReLU->pool2 -> Linear(1600,10)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class EnhancedMLP(nn.Module):
    """Question 4: 784 -> 500 -> ReLU -> Dropout(0.2) -> 200 -> ReLU -> Dropout(0.2) -> 10 -> log_softmax.
    Output and loss follow the original MLP (log_softmax + NLLLoss)."""

    def __init__(self, input_size: int = 784, hidden1: int = 500, hidden2: int = 200,
                 num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.l3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        out = self.drop1(self.relu1(self.l1(x)))
        out = self.drop2(self.relu2(self.l2(out)))
        out = self.l3(out)
        return F.log_softmax(out, dim=1)


def build_mobilenet_v2_frozen() -> nn.Module:
    """mnist_MobileNetV2.py: ImageNet weights, features frozen, classifier[1] -> 10 classes."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    return model


def _flatten(images: torch.Tensor) -> torch.Tensor:
    return images.reshape(-1, 28 * 28)


def _identity(images: torch.Tensor) -> torch.Tensor:
    return images


@dataclass(frozen=True)
class ModelSpec:
    """Everything that differs between the three original scripts."""

    name: str
    build: Callable[[], nn.Module]
    make_loss: Callable[[], nn.Module]
    epochs: int
    transform: str          # key understood by a2.data.make_transform
    prepare_input: Callable[[torch.Tensor], torch.Tensor]
    batch_size: int = 100
    learning_rate: float = 0.001


# Hyperparameters exactly as in the original scripts / assignment text.
MODEL_SPECS: dict[str, ModelSpec] = {
    "mlp": ModelSpec("mlp", NeuralNet, nn.NLLLoss, epochs=5,
                     transform="tensor", prepare_input=_flatten),
    "cnn": ModelSpec("cnn", CNNet, nn.NLLLoss, epochs=5,
                     transform="tensor", prepare_input=_identity),
    "mobilenetv2": ModelSpec("mobilenetv2", build_mobilenet_v2_frozen, nn.CrossEntropyLoss, epochs=3,
                             transform="imagenet", prepare_input=_identity),
}

# Question 4 model. Kept out of MODEL_SPECS so the Question 1-3 scripts ("--model all") are unchanged.
# Training configuration identical to the original MLP spec.
ENHANCED_MLP_SPEC = ModelSpec("mlp_enhanced", EnhancedMLP, nn.NLLLoss, epochs=5,
                              transform="tensor", prepare_input=_flatten)
