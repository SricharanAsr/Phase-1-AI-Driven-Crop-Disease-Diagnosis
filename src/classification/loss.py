import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance during training.
    
    Focal Loss reduces the loss contribution from easy examples and enables 
    the model to focus more on hard, misclassified examples.
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """Initializes FocalLoss.

        Args:
            alpha (float): Balancing factor for classes.
            gamma (float): Focusing parameter for hard examples.
            reduction (str): 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass for Focal Loss.

        Args:
            inputs (torch.Tensor): Logits from the model of shape (N, C).
            targets (torch.Tensor): Ground truth indices of shape (N).

        Returns:
            torch.Tensor: Calculated focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
