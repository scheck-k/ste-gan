"""
Audio pre-processing utilities.
Some functions are from: 
- https://github.com/dgaddy/silent_speech/blob/main/read_emg.py
- https://github.com/dgaddy/silent_speech/blob/main/data_utils.py
"""
import string
from pathlib import Path
from typing import *

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from textgrids import TextGrid
from torch import Tensor
from torchaudio.transforms import MFCC

from ste_gan.constants import PHONEME_INVENTORY


def normalize_volume(audio):
    rms = librosa.feature.rms(audio)
    max_rms = rms.max() + 0.01
    target_rms = 0.2
    audio = audio * (target_rms/max_rms)
    max_val = np.abs(audio).max()
    if max_val > 1.0: # this shouldn't happen too often with the target_rms of 0.2
        audio = audio / max_val
    return audio


def load_audio(audio_file_path: Path, start=None, end=None, max_frames=None,
               sampling_rate: int = 16_000, target_rms: float = 0.2, normalize: bool = True):
    audio, r = sf.read(audio_file_path)
    if r != sampling_rate:
        audio = librosa.resample(audio, r, sampling_rate)

    if len(audio.shape) > 1:
        audio = audio[:, 0]  # select first channel of stero audio
    if start is not None or end is not None:
        audio = audio[start:end]

    if normalize:
        audio = normalize_volume(audio)
    return audio


def read_text(file_path: Path) -> str:
    return file_path.read_text().lower()


def align_speech_units_and_mfccs(speech_units: np.ndarray, mfccs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(mfccs) % 2 == 1:
        mfccs = mfccs[:-1]  # Delete last element so that we have acoustic feats divisable by 2
    speech_units = speech_units[:(len(mfccs) // 2)]
    mfccs = mfccs[:2 * len(speech_units)]  # Speech units are half the number of samples as acoustic feats -->
    return speech_units, mfccs


def read_phonemes(textgrid_fname, max_len=None, sampling_rate: int = 16_000,
                  coeff=50.0):# coeff=50.0 -> Speech units shape

    tg = TextGrid(textgrid_fname)
    phone_ids = np.zeros(int(tg['phones'][-1].xmax*coeff)+1, dtype=np.int64)
    phone_ids[:] = -1
    phone_ids[-1] = PHONEME_INVENTORY.index('sil') # make sure list is long enough to cover full length of original sequence
    for interval in tg['phones']:
        phone = interval.text.lower()
        if phone in ['', 'sp', 'spn']:
            phone = 'sil'
        if phone[-1] in string.digits:
            phone = phone[:-1]
        ph_id = PHONEME_INVENTORY.index(phone)

        min_index = int(interval.xmin*coeff)
        max_index = int(interval.xmax*coeff)
        phone_ids[min_index:max_index] = ph_id
    assert (phone_ids >= 0).all(), 'missing aligned phones'

    if max_len is not None:
        phone_ids = phone_ids[:max_len]
        assert phone_ids.shape[0] == max_len
    return phone_ids



def cut_audio_to_soft_speech_match_unit_frame_rate(audio: np.ndarray, sample_rate: int = 16_000,
                                                   speech_unit_frequency: int = 50):
    """
    Cuts a mono audio signal to the length which would be reconstructed by soft speech units.
    The audio is cut on the right side such that it matches the expected length and no leftover audio frames.
    Args:
        audio: The mono audio as numpy array of shape (audio_samples,)
        sample_rate:  The sample rate of the audio.
        speech_unit_frequency: The frequency of speech units. Should be 50Hz for soft speech units.

    Returns: The cut audio.

    """
    downsample_rate = sample_rate // speech_unit_frequency
    num_units = len(audio) // downsample_rate
    num_audio_samples_after_cut = num_units * downsample_rate
    cut_mono_audio = audio[:num_audio_samples_after_cut]
    return cut_mono_audio



class MFCCsCalculator(nn.Module):
    
    def __init__(
        self, 
        n_mfcc = 25, 
        win_length: int = 512, 
        hop_length: int = 160,
        sample_rate: int = 16_000
    ):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.mel_kwargs = dict(
            n_fft=win_length, # 32ms
            win_length=win_length, #32ms
            hop_length=hop_length, # 
            center=False,
            onesided=True,
            n_mels=80, # from 128 to 80
        )
        self.mfcc_module = MFCC(sample_rate=sample_rate,  n_mfcc=n_mfcc, melkwargs=self.mel_kwargs)

    def forward(self, wav):
        padding = (self.win_length - self.hop_length) // 2
        wav = F.pad(wav, (padding, padding), "reflect")
        mfccs = self.mfcc_module(wav)
        return mfccs
        
    def from_audio_path(self, audio_path: Path) -> Tensor:
        audio = load_audio(audio_path)
        audio = cut_audio_to_soft_speech_match_unit_frame_rate(audio)
        audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(torch.float32)
        mfccs = self(audio)
        mfccs = mfccs.squeeze().T
        return mfccs
 


