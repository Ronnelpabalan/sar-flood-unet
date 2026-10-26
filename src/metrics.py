"""Pixel-wise metrics for binary water segmentation.

Label convention (Sen1Floods11): 1 = water, 0 = not water, -1 = no data.
Metrics are computed for the water class only, over valid pixels.
"""

import numpy as np


def valid_mask(label, s1=None):
    """Return a boolean mask of pixels that can be scored.

    A pixel is valid if it has a label (not -1) and, when `s1` is given,
    no SAR band is NaN at that pixel.

    label: (H, W) array of -1 / 0 / 1
    s1:    optional (bands, H, W) SAR array
    """
    valid = label != -1
    if s1 is not None:
        valid &= ~np.isnan(s1).any(axis=0)
    return valid


def confusion_counts(pred, label, valid=None):
    """Count true/false positives/negatives for water over valid pixels.

    pred:  (H, W) bool or 0/1 array, True/1 = predicted water
    label: (H, W) array of -1 / 0 / 1
    valid: optional (H, W) bool mask; defaults to label != -1
    """
    if valid is None:
        valid = label != -1
    p = pred[valid].astype(bool)
    t = label[valid] == 1
    return {
        "tp": int(np.sum(p & t)),
        "fp": int(np.sum(p & ~t)),
        "fn": int(np.sum(~p & t)),
        "tn": int(np.sum(~p & ~t)),
    }


def _safe_div(num, den):
    return num / den if den > 0 else float("nan")


def metrics_from_counts(c):
    """Turn confusion counts into IoU, F1, precision, recall and accuracy."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    return {
        "iou": _safe_div(tp, tp + fp + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
    }


class MetricAccumulator:
    """Pool confusion counts over many chips, then compute metrics once.

    Pooling (summing counts over all pixels of a split) avoids undefined
    per-chip IoU on chips that contain no water.
    """

    def __init__(self):
        self.counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    def add(self, pred, label, valid=None):
        for k, v in confusion_counts(pred, label, valid).items():
            self.counts[k] += v

    def compute(self):
        return metrics_from_counts(self.counts)
