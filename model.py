import torch
import torch.nn as nn
import torchvision.models as models


class OrthosageLandmarkRegressor(nn.Module):
    """ResNet50 backbone regressing (x, y) coordinates for a fixed set of landmarks."""

    def __init__(self, num_landmarks: int = 5):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_landmarks * 2),  # flat [x1, y1, x2, y2, ...]
        )

    def forward(self, x):
        return self.backbone(x)
