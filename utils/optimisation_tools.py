import torch
import torch.nn as nn

# Example: Custom MASE loss function
class MASELoss(nn.Module):
    def __init__(self, seasonality=1):
        super(MASELoss, self).__init__()
        self.seasonality = seasonality

    def forward(self, y_pred, y_true):
        # n = y_true.size(0)
        mae = torch.mean(torch.abs(y_pred - y_true))
        naive_error = torch.mean(torch.abs(y_true[self.seasonality:] - y_true[:-self.seasonality]))
        mase = mae / naive_error
        return mase
