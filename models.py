import torch
import torch.nn as nn
import torchvision.models as models

class AutomaticRangeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load ResNet18 (no pretrained weights here)
        resnet = models.resnet18(weights=None)

        # Change the first conv layer to accept 2 channels
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Use all layers except the final classification head
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Exclude final fc layer

        # Custom regression head for predicting (min, max)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output: [min_pred, max_pred]
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.head(out)

        # Sigmoid to constrain between 0 and 1
        out = torch.sigmoid(out)

        # Sort predictions to ensure min <= max
        out_sorted, _ = torch.sort(out, dim=1)

        return out_sorted