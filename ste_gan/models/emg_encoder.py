
"""The EMG encoder is based on the transduction model of this repo:

https://github.com/dgaddy/silent_speech/blob/main/architecture.py

"""
import logging
from random import random, randrange

import json
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import ste_gan
from ste_gan.layers.conv import ResBlock

from ste_gan.layers.transformer import TransformerEncoderLayer
from pathlib import Path
from typing import *




class EMGEncoder(nn.Module):
    """Base class for EMG encoders."""
    
    def __init__(self) -> None:
        super().__init__()
    
    def forward(x: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError(f"Must be implemented by subclasses.")


class EMGEncoderTransformer(EMGEncoder):
    """The Conv-Transformer EMG encoder: https://github.com/dgaddy/silent_speech/blob/main/architecture.py"""
    
    def __init__(
            self, 
            num_ins, 
            num_outs, 
            num_aux_outs, 
            model_size: int = 768,
            num_extra_res_blocks: int = 3,
            dropout:  float = 0.2,
            num_transformer_layers: int = 6,
        ):
        super().__init__()

        res_blocks = [
            ResBlock(num_ins, model_size, 2),
        ]
        for _ in range(num_extra_res_blocks):
            res_blocks.append(ResBlock(model_size, model_size, 2))

        # If we use the speech unit loss we must half the sample rate once more
        self.conv_blocks = nn.Sequential(*res_blocks)
        self.w_raw_in = nn.Linear(model_size, model_size)

        logging.info("Initializing standard Transformer model")
        encoder_layer = TransformerEncoderLayer(d_model=model_size, nhead=8,
                                                    relative_positional=True,
                                                    relative_positional_distance=100, dim_feedforward=3072,
                                                    dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        self.w_out = nn.Linear(model_size, num_outs)
        self.w_aux = nn.Linear(model_size, num_aux_outs)

    def forward(self, x_raw: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            r = randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:] # shift left r
                x_raw[:,-r:,:] = 0
                
        # Conv blocks
        x_raw = x_raw.transpose(1,2) # put channel before time for conv
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1,2)
        x_raw = self.w_raw_in(x_raw)

        x = x_raw
        x = x.transpose(0,1) # put time first
        x = self.transformer(x)
        x = x.transpose(0,1)
        # Linear out
        return self.w_out(x), self.w_aux(x)


def init_emg_encoder(
    cfg: omegaconf.DictConfig,
    device: torch.device = None,
) -> EMGEncoder:
    emg_encoder_config = cfg.emg_encoder
    num_ins: int = cfg.data.num_emg_channels
    
    # We always have speech units as target for the EMG encoder
    # If we train with MFCCs as inputs for the STE_GAN we still optimize the speech unit loss
    num_outs: int = ste_gan.SPEECH_UNITS_FEAT_SIZE
    num_aux_outs: int = len(ste_gan.PHONEME_INVENTORY)
    emg_encoder_type = emg_encoder_config["type"]
    emg_encoder_params = emg_encoder_config["params"]
    
    logging.info(f"Initializing EMG encoder with type {emg_encoder_type}")
    emg_encoder_args = dict(num_ins=num_ins, num_outs=num_outs, num_aux_outs=num_aux_outs)
    if emg_encoder_type == "EMGEncoderTransformer":
        emg_encoder = EMGEncoderTransformer(**emg_encoder_args, **emg_encoder_params)
    else:
        raise ValueError(f"Unknown EMG encoder type: {emg_encoder_type}")
    
    if device:
        emg_encoder = emg_encoder.to(device)
    
    return emg_encoder

def load_emg_encoder(
    cfg: omegaconf.DictConfig,
    device: torch.device,
    emg_encoder_checkpoint_path: Path,
) -> EMGEncoder:
    emg_encoder = init_emg_encoder(cfg, device)
    logging.info(f"Loading checkpoint path: {emg_encoder_checkpoint_path}")
    state_dict = torch.load(emg_encoder_checkpoint_path, map_location={"device": device})
    emg_encoder.load_state_dict(state_dict)
    emg_encoder.eval()
    return emg_encoder
