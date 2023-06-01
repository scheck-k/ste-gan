from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from einops import repeat, rearrange
import torch.nn.functional as F


class AverageFilter(nn.Module):
    
    def __init__(self, in_channels: int, window_size: int = 9, pad_signal: bool = True):
        """
        Initializes an average filter for a multi-channel signal x.
        :param window_size: The window size used for averaging. Must be an uneven number. Defaults to 9.
        """
        super().__init__()
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=window_size, stride=1)
        self.padding = window_size // 2
        self.pad_signal = pad_signal
        
    def forward(self, x: Tensor) -> Tensor:
        if self.pad_signal:
            x_padded = F.pad(x, (self.padding, self.padding), mode="reflect",)
        else:
            x_padded = x
        out = self.avg_pool(x_padded)
        return out
