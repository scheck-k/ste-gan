"""Implements the Discriminators.
We have kept the original discrimiantors of the CarGAN repository (https://github.com/descriptinc/cargan) and included smaller versions
which have a similar receptive field for the smaller EMG sample rate.

"""
import logging

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils import spectral_norm, weight_norm

import ste_gan
from ste_gan.layers.conv import NormedConv1d, NormedConv2d


class DiscriminatorP(nn.Module):
    
    def __init__(self,num_emg_channels: int, period, norm="weight_norm", name="DiscriminatorP"):
        super().__init__()
        self.name = name
        self.num_emg_channels = num_emg_channels
        self.layers = nn.ModuleList([
            NormedConv2d(num_emg_channels, 32, (5, 1), (3, 1), padding=(2, 0), norm=norm),
            NormedConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0), norm=norm),
            NormedConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0), norm=norm),
            NormedConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0), norm=norm),
            NormedConv2d(1024, 1024, (5, 1), 1, padding=(2, 0), norm=norm)])
        self.output = NormedConv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0))
        self.period = period

    def forward(self, x):
        # 1d to 2d
        x = F.pad(x, (0, self.period - x.shape[-1] % self.period), "reflect")
        x = x.view(x.shape[0], x.shape[1], x.shape[2] // self.period, self.period)
        fmaps = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            fmaps.append(x)
        fmaps.append(self.output(x))
        return fmaps



class DiscriminatorSmallerS(torch.nn.Module):

    def __init__(self, num_emg_channels, norm="weight_norm", name="DiscriminatorS"):
        super().__init__()
        self.name = name
        self.num_emg_channels = num_emg_channels

        self.layers = nn.ModuleList([
            NormedConv1d(num_emg_channels, 128, 15, 1, padding=7, norm=norm),
            NormedConv1d(128, 256, 37, 2, groups=4, padding=18, norm=norm),
            NormedConv1d(256, 512, 37, 2, groups=16, padding=18, norm=norm),
            NormedConv1d(512, 1024, 5, 1, padding=2, norm=norm)])
        self.output = NormedConv1d(1024, 1, 3, 1, padding=1)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            fmaps.append(x)
        fmaps.append(self.output(x))
        return fmaps


class DiscriminatorSmallerP(torch.nn.Module):
    
    def __init__(self,num_emg_channels: int, period, norm="weight_norm", name="DiscriminatorP"):
        super().__init__()
        self.name = name
        self.num_emg_channels = num_emg_channels
        self.layers = nn.ModuleList([
            NormedConv2d(num_emg_channels, 32, (3, 1), (1, 1), padding=(2, 0), norm=norm),
            NormedConv2d(32, 256, (3, 1), (3, 1), padding=(2, 0), norm=norm),
            NormedConv2d(256, 512, (3, 1), (3, 1), padding=(2, 0), norm=norm),
        ])
        self.output = NormedConv2d(512, 1, kernel_size=(3, 1), padding=(1, 0))
        self.period = period

    def forward(self, x):
        # 1d to 2d
        x = F.pad(x, (0, self.period - x.shape[-1] % self.period), "reflect")
        x = x.view(x.shape[0], x.shape[1], x.shape[2] // self.period, self.period)
        fmaps = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            fmaps.append(x)
        fmaps.append(self.output(x))
        return fmaps


class DiscriminatorS(torch.nn.Module):

    def __init__(self, num_emg_channels, norm="weight_norm", name="DiscriminatorS"):
        super().__init__()
        self.name = name
        self.num_emg_channels = num_emg_channels

        self.layers = nn.ModuleList([
            NormedConv1d(num_emg_channels, 128, 15, 1, padding=7, norm=norm),
            NormedConv1d(128, 128, 41, 2, groups=4, padding=20, norm=norm),
            NormedConv1d(128, 256, 41, 2, groups=16, padding=20, norm=norm),
            NormedConv1d(256, 512, 41, 4, groups=16, padding=20, norm=norm),
            NormedConv1d(512, 1024, 41, 4, groups=16, padding=20, norm=norm),
            NormedConv1d(1024, 1024, 41, 1, groups=16, padding=20, norm=norm),
            NormedConv1d(1024, 1024, 5, 1, padding=2, norm=norm)])
        self.output = NormedConv1d(1024, 1, 3, 1, padding=1)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            fmaps.append(x)
        fmaps.append(self.output(x))
        return fmaps


class DiscriminatorSmall(nn.Module):
    
    def __init__(self, num_emg_channels:int, num_multi_pool=5, num_multi_scale=3):
        super().__init__()
        self.num_emg_channels = num_emg_channels
        
        prime_ratios = [2, 3, 5, 7, 11]
        self.multi_pooled_disc = nn.ModuleList([
            DiscriminatorSmallerP(num_emg_channels, prime_ratios[i], name=f"DiscriminatorP-{prime_ratios[i]}")
            for i in range(num_multi_pool)])

        self.multi_scale_disc = nn.ModuleList([
            DiscriminatorSmallerS(
                num_emg_channels=num_emg_channels,
                norm="spectral_norm" if i == 0 else "weight_norm",
                name=f"DiscriminatorS-{i}",
            )
            for i in range(num_multi_scale)])
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
        
        self.discriminator_names = [d.name for d in self.multi_pooled_disc] + [d.name for d in self.multi_scale_disc]
 
    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        results = []
        
        for disc in self.multi_pooled_disc:
            results.append(disc(x))

        for disc in self.multi_scale_disc:
            results.append(disc(x))
            x = self.downsample(x)
            
        return results


class Discriminator(nn.Module):

    def __init__(self, num_emg_channels:int, num_multi_pool=5, num_multi_scale=3):
        super().__init__()
        self.num_emg_channels = num_emg_channels
        
        prime_ratios = [2, 3, 5, 7, 11]
        self.multi_pooled_disc = nn.ModuleList([
            DiscriminatorP(num_emg_channels, prime_ratios[i], name=f"DiscriminatorP-{prime_ratios[i]}")
            for i in range(num_multi_pool)])

        self.multi_scale_disc = nn.ModuleList([
            DiscriminatorS(
                num_emg_channels=num_emg_channels,
                norm="spectral_norm" if i == 0 else "weight_norm",
                name=f"DiscriminatorS-{i}",
            )
            for i in range(num_multi_scale)])
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
        
        self.discriminator_names = [d.name for d in self.multi_pooled_disc] + [d.name for d in self.multi_scale_disc]
        
    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        results = []
        
        for disc in self.multi_pooled_disc:
            results.append(disc(x))

        for disc in self.multi_scale_disc:
            results.append(disc(x))
            x = self.downsample(x)
            
        return results

    
def init_emg_discriminators(cfg: omegaconf.DictConfig) -> Discriminator :
    num_emg_channels = cfg.data.num_emg_channels
    discriminator_small = cfg.model.discriminator_small
    
    if discriminator_small:
        logging.info(f"Initializing small discriminators with {num_emg_channels} channels")
        return DiscriminatorSmall(num_emg_channels)
        
    logging.info(f"Initializing FULL discriminators with {num_emg_channels} channels")
    return Discriminator(num_emg_channels)


if __name__ == "__main__":
    BATCH_SIZE = 16
    EMG_SAMPLE_RATE_HZ = ste_gan.EMG_SAMPLE_RATE
    NUM_EMG_CHANNELS = ste_gan.NUM_EMG_CHANNELS
    EMG_CHUNK_SIZE = ste_gan.CHUNK_SIZE
    EMG_AR_SIZE = ste_gan.AR_INPUT_SIZE
    SPEECH_UNIT_DIM = ste_gan.SPEECH_UNITS_FEAT_SIZE
    SPEECH_UNIT_FRAMES = ste_gan.CHUNK_SIZE // ste_gan.HOPSIZE
    
    emg_ar_input = torch.randn(BATCH_SIZE, ste_gan.AR_INPUT_SIZE)
    random_emg_signal = torch.randn(BATCH_SIZE, EMG_CHUNK_SIZE, NUM_EMG_CHANNELS)
    random_speech_units = torch.randn(BATCH_SIZE, SPEECH_UNIT_FRAMES, SPEECH_UNIT_DIM)
    session_indices = torch.zeros(BATCH_SIZE, dtype=torch.int32)
    speaking_mode_indices = torch.zeros(BATCH_SIZE, dtype=torch.int32)
    
    discriminator = Discriminator()
    dis_result = discriminator(random_emg_signal)
    
    for dis_idx, dis_fmap in enumerate(dis_result):
        print(f"Dis {dis_idx+1}: {discriminator.discriminator_names[dis_idx]} - Feat map size: {len(dis_fmap)}")
        for feat_idx, dis_feats in enumerate(dis_fmap):
            print(f"\t {feat_idx} {dis_feats.shape}")

        
        
        
    