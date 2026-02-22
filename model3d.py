import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class VideoResNet2D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Pretrained 2D ResNet
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove final FC layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Final classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: (B, C, T, H, W)

        B, C, T, H, W = x.shape

        # reshape to process frames independently
        x = x.permute(0, 2, 1, 3, 4)        # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)       # (B*T, C, H, W)

        features = self.feature_extractor(x)  # (B*T, 512, 1, 1)
        features = features.flatten(1)        # (B*T, 512)

        # reshape back
        features = features.view(B, T, 512)

        # temporal average pooling
        features = features.mean(dim=1)       # (B, 512)

        out = self.classifier(features)

        return out
