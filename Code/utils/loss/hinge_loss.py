import torch
from torch import nn


__all__ = ['HingeLoss']


class HingeLoss(nn.Module):
    """
    SVM Loss Implementation
    """

    def __init__(self, margin=1.0):
        super(HingeLoss, self).__init__()
        self.margin = margin # SVM margin parameter

    def __call__(self, outputs, labels):
        n = labels.size(0)  # Number of samples
        correct_scores = outputs[range(n), labels].unsqueeze(1)
        margins = outputs - correct_scores + self.margin
        # Subtract margin from mean calculation
        loss = torch.max(margins, torch.tensor(0.0)).mean() - self.margin
        return loss
