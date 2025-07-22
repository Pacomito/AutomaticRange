import torch
import torch.nn as nn

class AutomaticRangeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  # 2 input channels
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Output: [min_pred, max_pred]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        out_sorted, _ = torch.sort(x, dim=1)
        return out_sorted