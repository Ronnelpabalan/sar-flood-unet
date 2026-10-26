"""Loss for binary water segmentation that ignores unlabelled pixels.

Targets use the dataset convention: 1 = water, 0 = not water, -1 = ignore.
"""

import torch
import torch.nn.functional as F


def bce_dice_loss(logits, target, ignore_index=-1, smooth=1.0):
    """Binary cross-entropy + Dice loss over valid pixels only.

    logits: (B, 1, H, W) or (B, H, W) raw model outputs (before sigmoid)
    target: (B, H, W) integer labels

    BCE scores every pixel on its own. Dice scores the overlap between the
    predicted and true water areas for the whole batch, which keeps the loss
    informative when water is only a small share of the pixels.
    """
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0.0               # no labelled pixels: zero loss, keeps the graph
    x = logits[valid]
    t = target[valid].float()
    bce = F.binary_cross_entropy_with_logits(x, t)
    p = torch.sigmoid(x)
    dice = 1 - (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)
    return bce + dice
