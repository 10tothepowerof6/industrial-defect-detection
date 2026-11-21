"""U-Net with an ImageNet-pretrained ResNet-18 encoder and its BCE + Dice loss."""

import segmentation_models_pytorch as smp
import torch
from torch import nn


def build_unet(encoder: str = "resnet18", encoder_weights: str | None = "imagenet") -> nn.Module:
    """One output channel of logits. Use encoder_weights=None when loading a checkpoint."""
    return smp.Unet(
        encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1
    )


class BCEDiceLoss(nn.Module):
    """BCEWithLogitsLoss + soft Dice on the batch. Takes logits; never apply sigmoid first."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = self.bce(logits, target)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * target).sum()
        dice = (2 * intersection + self.smooth) / (
            probabilities.sum() + target.sum() + self.smooth
        )
        return self.bce_weight * bce + self.dice_weight * (1 - dice)
