import numpy as np
import torch
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils.base import Activation


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [
            channel
            for channel in range(xs[0].shape[1])
            if channel not in ignore_channels
        ]
        xs = [
            torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device))
            for x in xs
        ]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, frequency=None, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    bs, c, _, _ = pr.size()
    if frequency is None:
        frequency = [1] * c
    weight = np.stack([np.full((256, 256), f) for f in frequency])
    weight = torch.from_numpy(weight).repeat(bs, 1, 1, 1).to("cuda")
    intersection = torch.sum(gt * pr * weight)
    union = torch.sum(gt * weight) + torch.sum(pr * weight) - intersection + eps
    return (intersection + eps) / union


class FWIoU(base.Metric):
    __name__ = "fwiou_score"

    def __init__(
        self,
        eps=1e-7,
        threshold=0.5,
        frequency=None,
        activation=None,
        ignore_channels=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.frequency = frequency

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return iou(
            y_pr,
            y_gt,
            frequency=self.frequency,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
