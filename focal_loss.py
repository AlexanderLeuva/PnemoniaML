import torch
import torch.nn as nn
import torch.nn.functional as F

# For binary classification specifically:
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Binary focal loss
        
        Args:
            inputs (torch.Tensor): Model predictions (N, 1) or (N,)
            targets (torch.Tensor): Ground truth labels (N,)
        """
        # Ensure proper shape
        if inputs.shape != targets.shape:
            inputs = inputs.view(-1)
            
        # Get probabilities using sigmoid
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Calculate focal loss
        focal_loss = (1-pt)**self.gamma * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss