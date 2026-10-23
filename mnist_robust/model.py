"""Madry MNIST ağının ölçeklenebilir iki-convolution sürümü."""

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Logits üretir; girişler normalize edilmemiş [0, 1] MNIST pikselleridir.

    width=32, hidden=1024 yazarların MNIST ağındaki katman boyutlarıdır.
    Daha küçük varsayılan ağ hızlı ilk deneyler için seçilmiştir.
    """

    def __init__(self, width: int = 16, hidden: int = 128):
        super().__init__()
        for name, value in (("width", width), ("hidden", hidden)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} pozitif bir tamsayı olmalı.")
        self.width = width
        self.hidden = hidden
        self.features = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 2 * width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
