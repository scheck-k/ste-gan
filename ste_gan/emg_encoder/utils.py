
"""Utility methods for training the EMG Encoder.
    
    Adapted from the silent_speech repo: https://github.com/dgaddy/silent_speech

"""
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

import torch
from pathlib import Path
from typing import *
import time
import time
import random

from ste_gan.emg_encoder.constants import EMG_SIGNAL_TO_SPEECH_UNITS, SEQ_LEN
from ste_gan.data.emg_dataset import EMGDataset
import ste_gan
from ste_gan import DataType


@jit
def time_warp(costs):
    dtw = np.zeros_like(costs)
    dtw[0,1:] = np.inf
    dtw[1:,0] = np.inf
    eps = 1e-4
    for i in range(1,costs.shape[0]):
        for j in range(1,costs.shape[1]):
            dtw[i,j] = costs[i,j] + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1,j-1])
    return dtw


def align_from_distances(distance_matrix, debug=False):
    # for each position in spectrum 1, returns best match position in spectrum2
    # using monotonic alignment
    dtw = time_warp(distance_matrix)

    i = distance_matrix.shape[0]-1
    j = distance_matrix.shape[1]-1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min([(i-1,j),(i,j-1),(i-1,j-1)], key=lambda x: dtw[x[0],x[1]])

    if debug:
        visual = np.zeros_like(dtw)
        visual[range(len(results)),results] = 1
        plt.matshow(visual)
        plt.show()

    return results


def create_output_dir_name(
    emg_data_set_roots: List[Path],
    emg_enc_name: str = "EMGTransformer-Soft_Speech-Units",
    include_timestap: bool = False,
    debug: bool = False,
    seq_len: int = SEQ_LEN,
) -> str: 
    data_set_names = "_".join(data_dir.name for data_dir in emg_data_set_roots)
    timestr = time.strftime("%Y%m%d-%H%M%S") + "_" if include_timestap else ""
    debug_str = "DEBUG_" if debug else ""
    
    return f"{debug_str}{timestr}{emg_enc_name}__seq_len__{seq_len}__data_{data_set_names}"


def is_data_dict_silent(data_dict: Dict):
    return data_dict[ste_gan.DataType.SPEAKING_MODE_ID] == ste_gan.SpeakingMode.SILENT


def collate_raw(batch):
    emg = [ex[DataType.REAL_EMG] for ex in batch]
    target_speech_units = [ex[DataType.SPEECH_UNITS] for ex in batch]
    target_phonemes = [ex[DataType.PHONEMES] for ex in batch]
    silent = [is_data_dict_silent(ex) for ex in batch]
    lengths = [len(ex[DataType.REAL_EMG]) for ex in batch]
    speech_unit_lengths = [length // EMG_SIGNAL_TO_SPEECH_UNITS for length in lengths ]

    result = {
        DataType.REAL_EMG: emg,
        "lengths": lengths,
        DataType.SPEECH_UNITS: target_speech_units,
        'speech_unit_lengths': speech_unit_lengths,
        DataType.PHONEMES: target_phonemes,
        'silent': silent,
    }
    return result

def combine_fixed_length(tensor_list, length):
    total_length = sum(t.size(0) for t in tensor_list)
    if total_length % length != 0:
        pad_length = length - (total_length % length)
        tensor_list = list(tensor_list) # copy
        added_tensor = torch.zeros(pad_length, *tensor_list[0].size()[1:],
                                   dtype=tensor_list[0].dtype, device=tensor_list[0].device)
        tensor_list.append(added_tensor)
        total_length += pad_length
    tensor = torch.cat(tensor_list, 0)
    n = total_length // length
    return tensor.view(n, length, *tensor.size()[1:])

def decollate_tensor(tensor, lengths):
    b, s, d = tensor.size()
    tensor = tensor.view(b*s, d)
    results = []
    idx = 0
    for length in lengths:
        assert idx + length <= b * s
        results.append(tensor[idx:idx+length])
        idx += length
    return results


def init_voiced_datasets_emg_encoder_training(
    emg_dataset_root: Path,
):
    trainset = EMGDataset(
        emg_dataset_root, partition="train",
        only_include_voiced=True,
        only_include_silent=False,
        return_mfccs=False,
        return_emg_feats=False,
        filter_by_length=False,
    )
    def init_eval_set(partition: str) -> EMGDataset:
        return EMGDataset(
            emg_dataset_root, 
            partition,
            filter_by_length=False,
            return_mfccs=False,
            return_emg_feats=False,
            only_include_silent=False, 
            only_include_voiced=True, # Here we only test on voiced data
            session_id_to_idx=trainset.session_id_to_idx,
            speaking_mode_id_to_idx=trainset.speaking_mode_id_to_idx,
        )
    
    devset = init_eval_set("valid")
    testset = init_eval_set("test")
    EMGDataset.check_no_data_overlap([trainset, devset, testset])
    
    return trainset, devset, testset


def init_datasets_for_emg_encoder_train(
    emg_dataset_root: Path,
    emg_synth_root_dir: Optional[Path] = None,
    emg_synth_dir_name: str = "",
    max_num_synth_train_utts: int = -1
):
    trainset = EMGDataset(
        emg_dataset_root, partition="train",
        only_include_voiced=False,
        only_include_silent=False,
        synth_root_dir=emg_synth_root_dir, 
        filter_by_length=False,
        synth_emg_dir_name=emg_synth_dir_name,
        max_num_synth_utts=max_num_synth_train_utts
    )
    def init_eval_set(partition: str) -> EMGDataset:
            return EMGDataset(
            emg_dataset_root, 
            partition,
            synth_root_dir=None,
            filter_by_length=False,
            only_include_silent=True, # Important -- Only test on silent EMG as in Gaddy data.
            only_include_voiced=False,
            session_id_to_idx=trainset.session_id_to_idx,
            speaking_mode_id_to_idx=trainset.speaking_mode_id_to_idx,
        )
    
    devset = init_eval_set("valid")
    testset = init_eval_set("test")
    EMGDataset.check_no_data_overlap([trainset, devset, testset])

    return trainset, devset, testset

class SizeAwareSampler(torch.utils.data.Sampler):
    def __init__(self, emg_dataset: EMGDataset, max_len):
        self.dataset = emg_dataset
        self.max_len = max_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batch = []
        batch_length = 0
        for idx in indices:
            length = self.dataset.emg_lengths[idx]
            if length + batch_length > self.max_len:
                yield batch
                batch = []
                batch_length = 0
            batch.append(idx)
            batch_length += length
        # dropping last incomplete batch

 