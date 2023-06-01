from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from typing import *
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import torch

import ste_gan
from ste_gan.layers.average_filter import AverageFilter

def get_envelope(emg_signal: np.ndarray, num_points_avg_filter: int = 40, pad_signal: bool = True) -> np.ndarray:
    num_ch = emg_signal.shape[1]
    avg_filter = AverageFilter(num_ch, window_size=num_points_avg_filter, pad_signal=pad_signal)
    abs_emg_batch = torch.abs(torch.from_numpy(emg_signal).unsqueeze(0))
    abs_emg_batch = rearrange(abs_emg_batch, 'b t c -> b c t')
    abs_emg_batch_filt = avg_filter(abs_emg_batch)
    abs_emg_batch_filt = rearrange(abs_emg_batch_filt, 'b c t -> b t c')
    return abs_emg_batch_filt.squeeze().numpy()

def plot_emg_signal_with_envelope(
    emg_signal: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "EMG Signal",
    ylim: Tuple[float, float] = (-1.0, 1.0),
    channels: int = [0,1,2,3,4],
    emg_sig_alpha: float = 0.3,
):
    if not ax:
        _, ax = plt.subplots()
    num_ch = emg_signal.shape[1]
    emg_abs_env = get_envelope(emg_signal)
    num_samples = min(len(emg_abs_env), len(emg_signal))
    x_ticks = np.arange(num_samples)
    
    cmap = plt.get_cmap('tab10')
    for ch_idx in (channels):
        color = cmap(ch_idx)
        ax.plot(x_ticks, emg_signal[:num_samples, ch_idx], label=f"Ch. {ch_idx+1}", alpha=emg_sig_alpha,
                color=color, )
        ax.plot(x_ticks, emg_abs_env[:num_samples, ch_idx,], label=f"Abs Env. Ch. {ch_idx+1}", color=color)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"Sample")
    ax.set_ylabel(f"Amplitude")
    return ax
    
def plot_real_vs_fake_emg_signal_with_envelope(
    real_emg_signal: np.ndarray,
    fake_emg_signal: np.ndarray,
    file_id: str,
    show: False,
    save_as: Optional[Path] = None,
    tb_summary_writer: Optional[SummaryWriter] = None,
    tb_tag_prefix: str = "emg_real_vs_fake_envelope",
    global_step: int = 0
):
    fig, (ax1, ax2, ) = plt.subplots(2)
    fig.suptitle(f'Real. vs. fake EMG Signal ({file_id})')
    plot_emg_signal_with_envelope(real_emg_signal, ax1, title=f"Real EMG signal")
    plot_emg_signal_with_envelope(fake_emg_signal, ax2, title=f"Fake EMG signal")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    if show:
        plt.show()
        
    if tb_summary_writer:
        tag = f"{tb_tag_prefix}_{file_id}"
        tb_summary_writer.add_figure(tag, fig, global_step)
    
    return fig