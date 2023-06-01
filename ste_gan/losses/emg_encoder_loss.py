"""Implements the losses involving the EMG encoder. They are: 
- The phoneme classification loss using EMG encoder predictions given fake EMG signals
- The speech unit loss, which takes the distance between GT and predicted speech units given fake EMG signals.
"""

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from ste_gan.models.emg_encoder import EMGEncoder
from typing import *

from ste_gan.constants import SILENCE_PHONEME_INDEX
from dataclasses import dataclass
from einops import rearrange


@dataclass
class EMGEncoderLossOutput:
    speech_unit_pred: Tensor
    phoneme_pred: Tensor
    speech_unit_loss: Tensor
    phoneme_loss: Tensor
    phoneme_targets: Tensor

    @property
    def num_phones(self) -> int:
        return len(torch.flatten(self.phoneme_targets))
    
    @property
    def num_silence_phones(self) -> int:
        silence_mask = torch.flatten(self.phoneme_targets) == SILENCE_PHONEME_INDEX 
        num_silence = silence_mask.sum().item()
        return num_silence
    
    @property
    def num_correct_phones(self) -> int:
        pred_argmax = self.phoneme_pred.argmax(-1)
        pred_argmax_flat = torch.flatten(pred_argmax)
        target_flat = torch.flatten(self.phoneme_targets)
        num_correct = (pred_argmax_flat == target_flat).sum().item()
        return int(num_correct)
    
    @property
    def num_correct_phones_no_silence(self) -> int:
        pred_argmax = self.phoneme_pred.argmax(-1)
        pred_argmax_flat = torch.flatten(pred_argmax)
        target_flat = torch.flatten(self.phoneme_targets)
        correct_mask = (pred_argmax_flat == target_flat)
        is_not_silent_mask = (target_flat != SILENCE_PHONEME_INDEX)
        num_correct = (correct_mask & is_not_silent_mask).sum().item()
        return num_correct
    

class EMGEncoderLoss(nn.Module):
    
    def __init__(self, emg_encoder: EMGEncoder) -> None:
        super().__init__()
        self.emg_encoder = emg_encoder
        self.emg_encoder.eval()
        
    def speech_unit_loss(self, speech_unit_target: Tensor, speech_unit_pred: Tensor) -> Tensor:
        target_flat = rearrange(speech_unit_target, 'b t d -> (b t) d')
        pred_flat = rearrange(speech_unit_pred, 'b t d -> (b t) d')
        pairwise_dists = F.pairwise_distance(target_flat, pred_flat)
        return torch.mean(pairwise_dists)

    def forward(
        self, 
        emg_signal: Tensor, 
        target_speech_units: Tensor,
        target_phoneme_sequence: Tensor
    ) -> EMGEncoderLossOutput:
        # Forward Pass with the EMG encoder
        speech_unit_pred, phoneme_pred = self.emg_encoder(emg_signal)
        # Pairwise euclidian loss
        speech_unit_loss_value =  self.speech_unit_loss(target_speech_units, speech_unit_pred)
        # Phoneme loss is the cross entropy
        phoneme_loss = F.cross_entropy(
            rearrange(phoneme_pred, 'b t p -> b p t'),
            target_phoneme_sequence
        )
        return EMGEncoderLossOutput(speech_unit_pred, phoneme_pred, speech_unit_loss_value, phoneme_loss, target_phoneme_sequence)
    
