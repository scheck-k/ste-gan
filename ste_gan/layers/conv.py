"""Contains various convolution layers used by models.
"""

from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils import spectral_norm, weight_norm

import ste_gan
from ste_gan.constants import DataType


# WNConv1D and GBLocks from: https://github.com/descriptinc/cargan/blob/master/cargan/model/gantts/generator.py
def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)



class GBlock(nn.Module):

    def __init__(self, input_dim, output_dim, upsample=1, kernel_size=3):
        super().__init__()

        ############################################################
        # Create first residual block consisting of conv1 and res1 #
        ############################################################
        
        self.conv1 = [nn.ReLU()]
        if upsample > 1:
            self.conv1 += [nn.Upsample(scale_factor=upsample)]
        self.conv1 += [
            WNConv1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                padding=get_padding(kernel_size)),
            nn.ReLU(),
            WNConv1d(
                output_dim,
                output_dim,
                kernel_size=kernel_size,
                dilation=3,
                padding=get_padding(kernel_size, 3))]

        self.res1 = [nn.Upsample(scale_factor=upsample)] if upsample > 1 else []
        self.res1 += [WNConv1d(input_dim, output_dim, kernel_size=1)]

        ####################################################
        # Create second residual block consisting of conv2 #
        ####################################################
        self.conv2 = [
            nn.ReLU(),
            WNConv1d(
                output_dim,
                output_dim,
                kernel_size=kernel_size,
                dilation=9,
                padding=get_padding(kernel_size, 9)),
            nn.ReLU(),
            WNConv1d(
                output_dim,
                output_dim,
                kernel_size=kernel_size,
                dilation=27,
                padding=get_padding(kernel_size, 27))]

        # Convert list of layers into nn.Sequential
        self.conv1 = nn.Sequential(*self.conv1)
        self.res1 = nn.Sequential(*self.res1)
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        x = self.conv1(x) + self.res1(x)
        return x + self.conv2(x)


# Discriminator Conv Layers from: https://github.com/descriptinc/cargan/blob/master/cargan/model/hifigan/discriminator.py

def NormedConv1d(*args, **kwargs):
    norm = kwargs.pop("norm", "weight_norm")
    if norm == "weight_norm":
        return weight_norm(nn.Conv1d(*args, **kwargs))
    elif norm == "spectral_norm":
        return spectral_norm(nn.Conv1d(*args, **kwargs))

def NormedConv2d(*args, **kwargs):
    norm = kwargs.pop("norm", "weight_norm")
    if norm == "weight_norm":
        return weight_norm(nn.Conv2d(*args, **kwargs))
    elif norm == "spectral_norm":
        return spectral_norm(nn.Conv2d(*args, **kwargs))
    


## Residual Blocks of the EMG encoder
class ResBlock(nn.Module):
    """https://github.com/dgaddy/silent_speech/blob/main/architecture.py"""
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)