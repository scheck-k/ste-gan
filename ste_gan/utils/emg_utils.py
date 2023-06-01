"""
EMG pre-processing utilities.
Some functions are from: 
- https://github.com/dgaddy/silent_speech/blob/main/read_emg.py
- https://github.com/dgaddy/silent_speech/blob/main/data_utils.py
"""
import string
from pathlib import Path
from typing import *

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from scipy.signal import butter, filtfilt, hilbert, iirnotch, resample
from textgrids import TextGrid
from torch import Tensor


def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)


def average_by_points(signal: np.ndarray, points: int):
    f = np.ones(points)/ float(points)
    v = np.convolve(signal, f, mode='same')
    return v

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def bandpass_signal(signal, fs):
    b, a = scipy.signal.butter(2, (2, 400), 'bandpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def lowpass_after_bandpass(signal, fs):
    b, a = scipy.signal.butter(2, 10, 'lowpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)


def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1, 8):
        signal = notch(signal, freq * harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal)) / old_freq
    sample_times = np.arange(0, times[-1], 1 / new_freq)
    result = np.interp(sample_times, times, signal)
    return result


def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:, i], *args, **kwargs))
    return np.stack(results, 1)


def cut_emg_to_hubert_units(
    emg: torch.Tensor,
    hubert_units: torch.Tensor,
    emg_sr: int,
    hubert_sr: int = 50
) -> torch.Tensor:
    num_emg_samples_expected = len(hubert_units) * (emg_sr // hubert_sr)
    assert num_emg_samples_expected <= len(emg)
    return emg[:num_emg_samples_expected]


def double_average(x):
    assert len(x.shape) == 1
    f = np.ones(9)/9.0
    v = np.convolve(x, f, mode='same')
    w = np.convolve(v, f, mode='same')
    return w


def butter_lowpass(cutoff, fs, order=4):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

    
def calculate_hilbert_envelope(
    single_raw_emg_signal: np.ndarray,
):
    # Convert to numpy
    # Hilbert Transform
    analytic_signal = hilbert(single_raw_emg_signal)
    # Get the envelope
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


def calculate_hilbert_transform_feats(
    single_raw_emg_signal: np.ndarray,
    input_emg_sample_rate: int = 800,
    target_feat_sample_rate: int = 100,
    lowpass_filter_hz: int = 20,
    max_num_frames: int = -1
) -> np.ndarray:
    """
    Implements the Hilbert Transformation features as described in Sharma et al.: 
    'A comparative study of different EMG features for acoustics-to-EMG mapping'"""
    assert len(single_raw_emg_signal.shape) == 1, "Not supporting batch operations for Hilbert Features"
    amplitude_envelope = calculate_hilbert_envelope(single_raw_emg_signal, )
    
    # Lowpass filter at 20Hz
    b, a = butter_lowpass(lowpass_filter_hz, input_emg_sample_rate)
    amplitude_envelope_lowpass = filtfilt(b, a,  amplitude_envelope)
    
    # Downsample from 800Hz -> 100Hz
    downsampling_factor = input_emg_sample_rate / target_feat_sample_rate
    num_expected_samples = int(len(amplitude_envelope_lowpass) / downsampling_factor)
    amplitude_envelope_lowpass_downsampled = resample(amplitude_envelope_lowpass, num_expected_samples)
    
    if max_num_frames >= 0:
        amplitude_envelope_lowpass_downsampled = amplitude_envelope_lowpass_downsampled[:max_num_frames]
    
    return amplitude_envelope_lowpass_downsampled
    

def pre_process_emg_signal(
    raw_emg: np.ndarray,
    raw_emg_before: np.ndarray,
    raw_emg_after: np.ndarray,
    emg_raw_target_sample_rate: int,
    apply_mu_law: bool = False,
    emg_source_sample_rate: int = 1000,
    remove_channels: Iterable[int] = set()
):
    """Pre-processes an EMG Signal

    Args:
        raw_emg (np.ndarray): _description_
        emg_before (np.ndarray): _description_
        emg_after (np.ndarray): _description_
        emg_raw_target_sample_rate (int): _description_
        emg_feat_target_sample_rate (int): _description_
        emg_source_sample_rate (int, optional): _description_. Defaults to 1000.
    """
    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    
    # Filter EMG data as in the original code
    x = apply_to_all(notch_harmonics, x, 60, emg_source_sample_rate)
    x = apply_to_all(remove_drift, x, emg_source_sample_rate)
    
    x = x[raw_emg_before.shape[0]:x.shape[0] - raw_emg_after.shape[0], :]

    # Subsample
    emg_orig = apply_to_all(subsample, x, emg_raw_target_sample_rate, emg_source_sample_rate)

    return emg_orig



def get_emg_features(
    emg_data_input: np.ndarray,
    frame_length_samples: int = 26, ## Approx. 32ms at 800Hz
    hop_length_samples: int = 8,  # 10 ms at 800Hz 
    debug=False,
    add_stft: bool = False,
    add_hilbert: bool = True,
    emg_sr: int = 800,
    pad: bool = False,
    subtract_mean: bool = True
):
    if pad:
        padding = (frame_length_samples - hop_length_samples) // 2
        emg_data = np.pad(emg_data_input, ((padding, padding), (0, 0)), 'reflect')
    else:
        emg_data = emg_data_input
        
    if subtract_mean:
        xs = emg_data - emg_data.mean(axis=0, keepdims=True)
    else:
        xs = emg_data

    frame_features = []
    for i in range(emg_data.shape[1]):
        x = xs[:,i]
        w = double_average(x)
        p = x - w
        r = np.abs(p)

        w_h = librosa.util.frame(x=w, frame_length=frame_length_samples, hop_length=hop_length_samples).mean(axis=0)
        p_w = librosa.feature.rms(y=w, frame_length=frame_length_samples, hop_length=hop_length_samples, center=False)
        p_w = np.squeeze(p_w, 0)
        p_r = librosa.feature.rms(y=r, frame_length=frame_length_samples, hop_length=hop_length_samples, center=False)
        p_r = np.squeeze(p_r, 0)
        z_p = librosa.feature.zero_crossing_rate(p, frame_length=frame_length_samples, hop_length=hop_length_samples, center=False)
        z_p = np.squeeze(z_p, 0)
        r_h = librosa.util.frame(x=r, frame_length=frame_length_samples, hop_length=hop_length_samples).mean(axis=0)

        if add_stft:
            s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=frame_length_samples, hop_length=hop_length_samples, center=False))
            
        if add_hilbert:
            hil_env = calculate_hilbert_transform_feats(emg_data_input[:, i], max_num_frames=w_h.shape[0], input_emg_sample_rate=emg_sr)

        feat_list = [w_h, p_w, p_r, z_p, r_h]
        if add_hilbert:
            feat_list.append(hil_env)
        frame_features.append(np.stack(feat_list, axis=1))
        if add_stft:
            frame_features.append(s.T)

    # Stack to num_frames x num_channels x num_feats
    frame_features =  np.stack(frame_features, axis=1) #np.concatenae(frame_features, axis=1)

    return frame_features.astype(np.float32)

