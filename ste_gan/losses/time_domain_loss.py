from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from einops import repeat, rearrange
import torch.nn.functional as F

from ste_gan.layers.average_filter import AverageFilter



class TimeDomainFeatureLoss(torch.nn.Module):
    """Implements a Time-Domain (TD) feature loss.
    It calculates the TD features of the real and fake EMG signals and subsequently
    calculates the L1 distance.
    """

    def __init__(
        self, 
        num_channels,
        win_size_samples: int = 21, # Approx. 27ms at 800Hz
        win_shift_samples: int = 8,  # Approx. 10ms at 800Hz
        apply_padding_windowing: bool = True,
        average_filter_window_size: int = 9,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.average_filter = AverageFilter(num_channels, average_filter_window_size)
        self.win_size_samples = win_size_samples
        self.win_shift_samples = win_shift_samples
        self.apply_padding_windowing = apply_padding_windowing
        self.avg_filter_window_size = average_filter_window_size

    def window_signal(self, _x: Tensor) -> Tensor:
        if self.apply_padding_windowing:
            pad_len = self.win_size_samples // 2
            _x = F.pad(_x, (0,0, pad_len, pad_len), mode="reflect")
            # Use the "unfold" operation to
        windowed_x_padded = _x.unfold(1, self.win_size_samples, self.win_shift_samples)
        return windowed_x_padded

    def frame_means(self, x: Tensor) -> Tensor:
        x_windowed = self.window_signal(x)
        return torch.mean(x_windowed, -1)

    def frame_power(self, x: Tensor) -> Tensor:
        x_windowed = self.window_signal(x)
        return torch.sum(x_windowed ** 2, -1)
    
    def double_average(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b t c -> b c t')
        x_double_avg = self.average_filter(self.average_filter(x))
        x_double_avg = rearrange(x_double_avg, 'b c t -> b t c')
        return x_double_avg

    def calculate_time_domain_features(self, raw_x: Tensor) -> Tensor:
        lowpass_x = self.double_average(raw_x) 
        high_freq_x = raw_x - lowpass_x
        rectified_high_freq_x = torch.abs(high_freq_x)

        td_features = torch.stack([
            self.frame_means(lowpass_x),
            self.frame_power(lowpass_x),
            self.frame_power(rectified_high_freq_x),
            self.frame_means(rectified_high_freq_x),
        ], dim=-1)
        return td_features

    def time_domain_loss(self, x_real: Tensor, x_generated: Tensor):
        td_feats_real = self.calculate_time_domain_features(x_real)
        td_feats_gen = self.calculate_time_domain_features(x_generated)
        return F.l1_loss(input=td_feats_gen, target=td_feats_real.detach())


class MultiTimeDomainFeatureLoss(torch.nn.Module):
    """Implements the Multi-Time-Domain loss.
    It calculates  the sum of L1 losses between Time-Domain (TD) features of generated and real EMG.
    We use different window sizes and shifts in our implementation.
    """

    def __init__(
        self, 
        num_channels: int, 
    ):
        super().__init__()
        self.time_domain_losses = nn.ModuleList([
            # 25ms win size / 10ms win shift
            TimeDomainFeatureLoss(num_channels, win_size_samples=20, win_shift_samples=8),
            # 64ms win size / 16ms win shift
            TimeDomainFeatureLoss(num_channels, win_size_samples=51, win_shift_samples=13),
            # 100ms win size / 25ms win shift
            TimeDomainFeatureLoss(num_channels, win_size_samples=80, win_shift_samples=16),
        ])

    def time_domain_loss(self, x_real: Tensor, x_generated: Tensor) -> Tuple[Tensor, List[Tensor]]:
        loss = 0.0
        td_losses_vals = []
        for td_loss_obj in self.time_domain_losses:
            td_loss_val = td_loss_obj.time_domain_loss(x_real, x_generated)
            td_losses_vals.append(td_loss_val)
            loss += td_loss_val
        return loss, td_losses_vals
    
    def forward(self, x_real: Tensor, x_generated: Tensor) -> Tensor:
        loss, vals = self.time_domain_loss(x_real, x_generated)
        return loss

    