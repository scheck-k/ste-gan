# Modified code from: https://github.com/dgaddy/silent_speech/blob/main/data_collection/clean_audio.py
# Uses MetricGAN Plus

# Ignore FutureWarnings from the interaction of librosa & old noisereduce versions
import warnings

import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import numpy as np

import noisereduce as nr
import soundfile as sf
import librosa
import torch

from speechbrain.pretrained import SpectralMaskEnhancement

CLEAN_METRICGAN = True


def normalize_volume(audio):
    rms = librosa.feature.rms(y=audio)
    max_rms = rms.max() + 0.01
    target_rms = 0.2
    audio = audio * (target_rms / max_rms)
    max_val = np.abs(audio).max()
    if max_val > 1.0:  # this shouldn't happen too often with the target_rms of 0.2
        audio = audio / max_val
    return audio

def clean_directory(directory):
    silence, rate = sf.read(os.path.join(directory, '0_audio.flac'))

    audio_file_names = []
    # load audio files in numerical order
    while True:
        i = len(audio_file_names)
        fname = os.path.join(directory, f'{i}_audio.flac')
        if os.path.exists(fname):
            audio_file_names.append(fname)
        else:
            break

    all_audio_file_names = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('_audio.flac')]
    assert len(audio_file_names) == len(all_audio_file_names), 'error discovering audio files'

    all_rmses = []
    for fname in audio_file_names:
        data, rate = sf.read(fname)
        rms = librosa.feature.rms(y=data)[0]
        all_rmses.append(rms)

    silent_cutoff = 0.02
    smoothing_width = 20
    target_rms = 0.2
    clip_to = 0.99

    max_rmses = [np.max(r) for r in all_rmses]
    smoothed_maxes = []
    is_silent = False
    for i in range(len(max_rmses)):
        vs = [max_rmses[j] for j in range(max(0,i-smoothing_width),min(i+1+smoothing_width,len(max_rmses))) if max_rmses[j] > silent_cutoff]
        if len(vs) == 0:
            is_silent = True
            break
        smoothed_maxes.append(np.mean(vs))

    if is_silent:
        print('long run of quiet audio, skipping volume normalization')

    if CLEAN_METRICGAN:
        model_id = "speechbrain/metricgan-plus-voicebank"
        print(f"Loading SpectralMaskEnhancement model {model_id}")
        enhance_model = SpectralMaskEnhancement.from_hparams(
            source=model_id,
            savedir="pretrained_models/metricgan-plus-voicebank",
        )
    else:
        enhance_model = None

    # We use 16kHz audio
    sample_rate = 16_000
    for i, fname in enumerate(tqdm.tqdm(audio_file_names, total=len(audio_file_names))):
        data, rate = sf.read(fname)

        clean = nr.reduce_noise(audio_clip=data, noise_clip=silence)

        old_clean_len = len(clean)
        if CLEAN_METRICGAN:
            assert rate == 16_000
            clean = torch.from_numpy(clean).float().unsqueeze(0)
            clean = enhance_model.enhance_batch(clean, lengths=torch.tensor([1.]))
            clean = clean.detach().cpu().squeeze().numpy()
            assert len(clean) == old_clean_len

        if rate != sample_rate:
            clean = librosa.resample(clean, rate, sample_rate)

        if not is_silent:
            clean = normalize_volume(clean)
        clean_full_name = fname[:-5] + '_clean.flac'
        sf.write(clean_full_name, clean, sample_rate)


if __name__ == "__main__":
    assert len(sys.argv) > 1, 'requires at least 1 argument: the directories to process'
    for i in range(1, len(sys.argv)):
        print('cleaning', sys.argv[i])
        clean_directory(sys.argv[i])
