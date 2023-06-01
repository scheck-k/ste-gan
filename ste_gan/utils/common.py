import logging
from dataclasses import dataclass
from pathlib import Path
from typing import *

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ste_gan.models.generator import EMGGenerator


def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    prefix = '_orig_mod.' 
    for sd_key in state_dict.keys():
        if prefix in sd_key:
            new_state_dict[sd_key.replace(prefix, "")] = state_dict[sd_key]
        else:
            new_state_dict[sd_key] = state_dict[sd_key]
    return new_state_dict

def load_latest_checkpoint(
    checkpoint: Path,
    device: torch.device,
    generator: nn.Module,
    discriminator: nn.Module,
    optim_g: Optimizer,
    optim_d: Optimizer
) -> Tuple[nn.Module, nn.Module, Optimizer, Optimizer, int, int]:
    epochs = []
    for f in checkpoint.glob('checkpoint-*.pt'):
        try:
            epoch_int = int(f.stem.split('-')[1])
            epochs.append(epoch_int)
        except ValueError:
            pass
    epochs.sort()
    latest = f'{epochs[-1]:08d}'
    
    logging.info(f"LOADING GENERATOR CHECKPOINT: {checkpoint / f'netG-{latest}.pt'}")
    generator.load_state_dict(
        fix_state_dict(torch.load(checkpoint / f'netG-{latest}.pt', map_location=device)))
    logging.info(f"LOADING DISCRIMINATOR CHECKPOINT: {checkpoint / f'netD-{latest}.pt'}")
    discriminator.load_state_dict(
        fix_state_dict(torch.load(checkpoint / f'netD-{latest}.pt', map_location=device)))
    logging.info(f"LOADING OPTIMIZATION CHECKPOINT: {checkpoint / f'checkpoint-{latest}.pt'}")
    ckpt = fix_state_dict(torch.load(
        checkpoint / f'checkpoint-{latest}.pt',
        map_location=device))
    optim_g.load_state_dict(ckpt['optG'])
    optim_d.load_state_dict(ckpt['optD'])
    start_epoch, steps = ckpt['epoch'], ckpt['steps']
    logging.info(f"START EPOCH {start_epoch}")
    logging.info(f"STEPS {steps}")
    
    return (
        generator, discriminator,
        optim_g, optim_d,
        start_epoch, steps
    )


def initialize_emg_generator(
    generator: nn.Module,
    gen_ckpt_file_path: Path,
    device: torch.device,
) -> EMGGenerator:
    generator.load_state_dict(
        torch.load(gen_ckpt_file_path, map_location=device)
    )
    generator.eval()
    return generator
