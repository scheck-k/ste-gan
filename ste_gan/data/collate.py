"""Collate function for STE-GAN
    Adapted from CarGAN repo: https://github.com/descriptinc/cargan
"""
import torch

import ste_gan
from ste_gan.constants import DataType

###############################################################################
# Collate function
###############################################################################
VALID_PARTITIONS = {"train", "valid", "test"}


def check_partition(partition: str) -> None:
       # Raise on bad partition
    if partition not in VALID_PARTITIONS:
        raise ValueError(
            f'Partition must be one of ["train", "valid", "test"]')


def ste_gan_collate(
    batch, partition="train", 
    emg_train_length: int = ste_gan.TRAIN_EMG_LENGTH,
    chunksize: int = ste_gan.CHUNK_SIZE,
    hopsize: int = ste_gan.HOPSIZE
):
    check_partition(partition)
    emg_signals = []
    speech_units = []
    phonemes = []
    session_indices = []
    speaking_mode_indices = []
    mfccs = []
    for sample in batch:
        emg_signals.append(sample[DataType.REAL_EMG])
        speech_units.append(sample[DataType.SPEECH_UNITS])
        phonemes.append(sample[DataType.PHONEMES])
        session_indices.append(sample[DataType.SESSION_INDEX])
        speaking_mode_indices.append(sample[DataType.SPEAKING_MODE_INDEX])
        mfccs.append(sample[DataType.MFCCS])

    
    train_feature_length = emg_train_length // hopsize
    # Collate the features
    speech_units, phonemes, session_indices, speaking_mode_indices, start_indices, mfccs \
        = prepare_ste_gan_features(speech_units, phonemes, session_indices, speaking_mode_indices, partition, mfccs,
                                   train_feature_length=train_feature_length)
                                
    
    # Collate the EMG signal
    emg_signals = prepare_emg(emg_signals, partition, start_indices, emg_train_length, hopsize)
    
    return {
        DataType.REAL_EMG: emg_signals,
        DataType.SPEECH_UNITS: speech_units,
        DataType.PHONEMES: phonemes,
        DataType.SESSION_INDEX: session_indices,
        DataType.SPEAKING_MODE_INDEX: speaking_mode_indices,
        DataType.MFCCS: mfccs,
    }

###############################################################################
# Collate utilities
###############################################################################

def prepare_emg(emg_list, partition, start_indices, train_emg_length, hopsize):
    """Pad a batch of EMG signals
    :param emg_list: The list of EMG tensors
    
    """
    check_partition(partition)
    # Prepare EMG for training or validation
        # Prepare audio for training or validation
    if partition in ['train', 'valid']:

        # Convert indices from frames to samples
        start_indices *= hopsize

        # Crop EMG 
        return torch.stack([
            x[start_idx:start_idx + train_emg_length]
            for x, start_idx in zip(emg_list, start_indices)])
    
    # Prepare audio for testing
    else: # partition == 'test':
        assert len(emg_list) == 1
        return emg_list[0].unsqueeze(0)
    

def prepare_ste_gan_features(speech_units_list, phonemes_list, session_id_list, speaking_mode_id_list, partition,
                             mfccs_list,
                             train_feature_length=ste_gan.TRAIN_FEATURE_LENGTH):
    """
    Pad a batch of input features consisting of the soft speech units inputs and session IDS.
    """
    check_partition(partition)
  
    length = train_feature_length
    session_ids = torch.stack(session_id_list, axis=0)
    speaking_modes = torch.stack(speaking_mode_id_list, axis=0)
    
    # Prepare features for training
    if partition == 'train':
        speech_units = []
        phonemes = []
        start_idxs = []
        mfccs = []
        
        # Itterate through the per-utt samples
        for speech_units_utt, phonemes_utt, mfccs_utt in zip(speech_units_list, phonemes_list,
                                                             mfccs_list):
        # Get a random start index
            start_idx = torch.randint(
                0, # Start of sampling 
                1 + max(0, len(speech_units_utt) - length), # max start index for sampling
                (1,)).item()
            
            # Save slice and start point
            mfccs_len = 2 * length
            mfcc_start_idx = 2 * start_idx
            
            speech_units.append(speech_units_utt[start_idx:start_idx + length])
            phonemes.append(phonemes_utt[start_idx:start_idx + length])
            mfccs.append(mfccs_utt[mfcc_start_idx:mfcc_start_idx + mfccs_len])
            start_idxs.append(start_idx)

        # Convert to arrays
        speech_units = torch.stack(speech_units, axis=0)
        phonemes = torch.stack(phonemes, axis=0)
        start_idxs = torch.tensor(start_idxs, dtype=torch.int)
        mfccs = torch.stack(mfccs, axis=0)

    # Prepare features for validation (start_idxs is all zeros)
    elif partition == 'valid':
        speech_units = torch.stack([
            speech_unit_utt[:length] for speech_unit_utt in speech_units_list
        ], axis=0)
        phonemes = torch.stack([
            phonemes_utt[:length] for phonemes_utt in phonemes_list
        ], axis=0)
        mfccs = torch.stack([
            mfccs[:2*length] for mfccs in mfccs_list
        ], axis=0)
        start_idxs = torch.zeros(len(speech_units), dtype=torch.int)
    
    # Prepare features for testing
    else: # partition == 'test':
        assert len(speech_units_list) == 1
        assert len(phonemes_list) == 1
        assert len(mfccs_list) == 1
        speech_units = speech_units_list[0].unsqueeze(0)
        phonemes = phonemes_list[0].unsqueeze(0)
        mfccs = mfccs_list[0].unsqueeze(0)
        start_idxs = torch.zeros(1, dtype=torch.int)
    
    return speech_units, phonemes, session_ids, speaking_modes, start_idxs, mfccs
