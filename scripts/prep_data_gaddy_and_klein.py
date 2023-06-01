"""
This script pre-processes the data set of David Gaddy and Dan Klein: "Digital Voicing of Silent Speech"
It is a modified script of the authors' code: https://github.com/dgaddy/silent_speech/blob/main/read_emg.py
    
It carries out the following pre-processing steps:
- Reads EMG data, filters it and downsamples it to 800 Hz.
- Extract Soft Speech Units from Audio via the Soft HuBERT model.
- Extracts MFCCs 
- Phoneme targets are loaded from the forced alignment directory.
"""
import argparse
import json
import os
import random
import re
from functools import lru_cache, partial
from pathlib import Path
from typing import *

import numpy as np
import soundfile as sf
import torch.cuda
import torch.nn.functional as F
import tqdm

from ste_gan.constants import PHONEME_INVENTORY
from ste_gan.utils.audio_utils import (
    MFCCsCalculator, cut_audio_to_soft_speech_match_unit_frame_rate,
    read_phonemes)
from ste_gan.utils.emg_utils import get_emg_features, pre_process_emg_signal


def get_utterance_file_id_from_sample_dict(sample_dict: Dict) -> str:
    emg_path = Path(sample_dict["emg_path"])
    utt_idx = emg_path.stem.split("_", maxsplit=1)[0]
    session_id = emg_path.parent.name
    data_split = emg_path.parents[1].name
    session_id = f"{data_split}_{session_id}"
    
    silent = sample_dict["silent"]
    silent_str = "silent" if silent else "normal"

    return f"{session_id}__{utt_idx}__{silent_str}"


class EMGDirectory(object):
    def __init__(self, session_index, directory, silent, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.silent = silent
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index

    def __repr__(self):
        return self.directory


def get_textgrid_path(sess: str, index: int, text_align_directory: str, ) -> str:
    text_align_path = f'{text_align_directory}/{sess}/{sess}_{index}_audio.TextGrid'
    return text_align_path


def load_raw_emg_and_before_after_signals(base_dir, index) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_emg = np.load(os.path.join(base_dir, f'{index}_emg.npy'))
    before = os.path.join(base_dir, f'{index - 1}_emg.npy')
    after = os.path.join(base_dir, f'{index + 1}_emg.npy')
    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0, raw_emg.shape[1]])
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0, raw_emg.shape[1]])
    
    return raw_emg, raw_emg_before, raw_emg_after


def load_utterance(directory_info, index, limit_length=False, debug=False, text_align_directory=None, hubert=None, device=None):
    assert hubert is not None
    assert device is not None
    
    base_dir = directory_info.directory
    index = int(index)
    silent = directory_info.silent
    
    raw_emg, raw_emg_before, raw_emg_after = load_raw_emg_and_before_after_signals(base_dir, index)
    
    emg_orig = pre_process_emg_signal(
        raw_emg, raw_emg_before, raw_emg_after,
        emg_raw_target_sample_rate=800,
    )
    emg_features = get_emg_features(emg_orig, frame_length_samples=26, hop_length_samples=8, pad=True)
    
    audio_path = Path(base_dir) / f'{index}_audio_clean.flac'
    if not audio_path.exists():
        raise ValueError(f"Audio path does not exist (run clean_audio.py): {audio_path}")
    audio, sr = sf.read(audio_path)
    assert sr == 16_000, "Audio must be sampled to 16kHz"
    
    audio = cut_audio_to_soft_speech_match_unit_frame_rate(audio)
    mfccs_calc = MFCCsCalculator()
    audio_for_mfccs = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(torch.float32)
    mfccs = mfccs_calc(audio_for_mfccs)
    mfccs = mfccs.squeeze().T.numpy()
    
    if not silent:
        if emg_features.shape[0] > mfccs.shape[0]:
            emg_features = emg_features[:mfccs.shape[0], :]
        elif mfccs.shape[0] > emg_features.shape[0]:
            mfccs = mfccs[:emg_features.shape[0], :]

    # Calculate soft speech units  speech units from audio
    audio_t = np.expand_dims(np.expand_dims(audio, 0), 0)
    audio_t = torch.from_numpy(audio_t).float().to(device)
    speech_units = hubert.units(audio_t).squeeze().detach().cpu().numpy()

    if len(mfccs) % 2 == 1:
        mfccs = mfccs[:-1]  # Delete last element so that we have acoustic feats divisable by 2
    speech_units = speech_units[:(len(mfccs) // 2)]
    mfccs = mfccs[:2 * len(speech_units)]  # Speech units are half the number of samples as acoustic feats -->

    if not silent:
        # Limit Frames MFFcs for one-off errors after MFCCs were altered
        emg_features = emg_features[:len(mfccs)]
    else:
        speech_units = None

    # If we only have silent EMG input, we should not align EMG and audio data, as the audio data length does
    # not correspond to the silently spoken utterance.
    if not silent:
        assert emg_features.shape[0] == mfccs.shape[0]
        if speech_units is not None:
            assert emg_features.shape[0] == 2 * speech_units.shape[0], 'Error in speech units length'

    # Do not delete the first window of EMG features since now we padded the EMG signal.
    emg_orig = emg_orig[: 8 * emg_features.shape[0]]
    assert emg_orig.shape[0] == emg_features.shape[0] * 8, \
        f"Shape mismatch. EMG orig {emg_orig.shape} -- {emg_features.shape}"

    with open(os.path.join(base_dir, f'{index}_info.json')) as f:
        info = json.load(f)

    sess = os.path.basename(base_dir)
    tg_fname = get_textgrid_path(sess, index, text_align_directory)

    if os.path.exists(tg_fname):
        phonemes = read_phonemes(tg_fname, speech_units.shape[0])
    else:
        if speech_units is not None:
            phonemes = np.zeros(speech_units.shape[0], dtype=np.int64) + PHONEME_INVENTORY.index('sil')
        else:
            phonemes = np.zeros(mfccs.shape[0] // 2, dtype=np.int64) + PHONEME_INVENTORY.index('sil')

    return mfccs, emg_features, info['text'], (info['book'], info['sentence_index']), phonemes, emg_orig.astype(np.float32), speech_units, audio, audio_path


def only_include_alphanumeric_chars(text: str):
    return re.sub(r'\W+', '', text.strip())


def get_reference_id_from_silent_sample(sample: Dict) -> Tuple[str, str, str]:
    audio_file = Path(sample["audio_file"])
    utt_id = audio_file.stem.split("_")[0]
    session_id = audio_file.parent.name
    data_split = audio_file.parents[1].name
    ref_identifier = data_split, session_id, utt_id
    return ref_identifier


class EMGDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        base_dir=None, 
        limit_length=False,
        dev=False,
        test=False,
        no_testset=False,
        text_align_directory=None,
        testset_file=None,
        silent_data_directories=None,
        voiced_data_directories=None,
        hubert=None,
        device=None
    ):
        for var in text_align_directory, testset_file, silent_data_directories, voiced_data_directories, hubert, device:
            assert var is not None

        assert text_align_directory is not None
        assert testset_file is not None
        self.silent_data_directories = silent_data_directories
        self.voiced_data_directories = voiced_data_directories
        self.mfcc_calc = MFCCsCalculator()
        self.hubert = hubert
        self.device = device
        
        self.text_align_directory = text_align_directory
        self.testset_file = testset_file

        if no_testset:
            devset = []
            testset = []
        else:
            with open(testset_file) as f:
                testset_json = json.load(f)
                devset = testset_json['dev']
                testset = testset_json['test']

        directories = []
        if base_dir is not None:
            directories.append(EMGDirectory(0, base_dir, False))
        else:
            for sd in self.silent_data_directories:
                for session_dir in sorted(os.listdir(sd)):
                    if "DS_Store" in session_dir:
                        continue
                    directories.append(EMGDirectory(len(directories), os.path.join(sd, session_dir), True))

            has_silent = len(self.silent_data_directories) > 0
            for vd in self.voiced_data_directories:
                for session_dir in sorted(os.listdir(vd)):
                    if "DS_Store" in session_dir:
                        continue
                    directories.append(EMGDirectory(len(directories), os.path.join(vd, session_dir), False,
                                                    exclude_from_testset=has_silent))

        self.example_indices = []
        self.voiced_data_locations = {}  # map from book/sentence_index to directory_info/index
        for directory_info in directories:
            for fname in os.listdir(directory_info.directory):
                if fname == ".DS_Store":
                    continue
                m = re.match(r'(\d+)_info.json', fname)
                if m is not None:
                    idx_str = m.group(1)
                    with open(os.path.join(directory_info.directory, fname)) as f:
                        info = json.load(f)
                        text = info["text"]

                        # Only add the utterance to the exmaple indices if its text has alphanumeric characters
                        # For instance, ignore utterance_ids which are only points etc.
                        text_alphanumeric = only_include_alphanumeric_chars(text)

                        if text_alphanumeric and info['sentence_index'] >= 0:  # boundary clips of silence are marked -1
                            
                            location_in_testset = [info['book'], info['sentence_index']] in testset
                            location_in_devset = [info['book'], info['sentence_index']] in devset

                            #if directory_info.exclude_from_testset:
                            #    print(f"{fname} dev?{location_in_devset} test?{location_in_testset}")
                            if (test and location_in_testset and not directory_info.exclude_from_testset) \
                                    or (dev and location_in_devset and not directory_info.exclude_from_testset) \
                                    or (not test and not dev and not location_in_testset and not location_in_devset):

                                    self.example_indices.append((directory_info, int(idx_str)))

                            if not directory_info.silent:
                                location = (info['book'], info['sentence_index'])
                                self.voiced_data_locations[location] = (directory_info, int(idx_str))

        self.example_indices.sort()
        random.seed(0)
        random.shuffle(self.example_indices)

        self.limit_length = limit_length
        self.num_sessions = len(directories)

    def __len__(self):
        return len(self.example_indices)

    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]
        mfccs, emg_features, text, book_location, phonemes, raw_emg, speech_units, audio, audio_path \
        = load_utterance(directory_info, idx, self.limit_length, text_align_directory=self.text_align_directory,
                         hubert=self.hubert, device=self.device)
        
        # Last processing of the EMG signal
        # Legacy downscaling of EMG 
        raw_emg = raw_emg / 100.0 
        # Legacy upsacling via tanh
        raw_emg = np.tanh(raw_emg)

        session_ids = np.full(emg_features.shape[0], directory_info.session_index, dtype=np.int64)

        audio_file = str(audio_path)

        speech_units = torch.from_numpy(speech_units) if speech_units is not None else None
        result = {
            'audio': torch.from_numpy(audio),
            'mfccs': torch.from_numpy(mfccs),
            'speech_units': speech_units,
            'emg_features': torch.from_numpy(emg_features),
            'text': text,
            'file_label': idx, 
            'session_ids': torch.from_numpy(session_ids),
            'book_location': book_location, 
            'silent': directory_info.silent,
            'raw_emg': torch.from_numpy(raw_emg),
        }
        
        # Load audio of audible articulation of the same content
        if directory_info.silent:
            voiced_directory, voiced_idx = self.voiced_data_locations[book_location]
            voiced_mfccs, voiced_emg_features, _, _, phonemes, _, voiced_speech_units, voiced_audio, voiced_audio_path \
                = load_utterance(voiced_directory, voiced_idx, False, text_align_directory=self.text_align_directory,
                                 hubert=self.hubert, device=self.device)

            result['parallel_speech_units'] = torch.from_numpy(voiced_speech_units)
            result['parallel_voiced_emg_features'] = torch.from_numpy(voiced_emg_features)
            result["parallel_mfccs"] = torch.from_numpy(voiced_mfccs)
            result["parallel_audio"] = torch.from_numpy(voiced_audio)
            audio_file = str(voiced_audio_path)

        result['phonemes'] = torch.from_numpy(
            phonemes).pin_memory()  # either from this example if vocalized or aligned example if silent
        result['emg_path'] = f'{directory_info.directory}/{idx}_emg.npy'
        result['audio_file'] = audio_file
        
        return result


def get_silent_samples_to_reference_session_ids_and_utterance_ids(data_set: EMGDataset) -> Set[Tuple[str, str, str]]:
    reference_audio_sess_id_and_utt_id = set()
    for sample in tqdm.tqdm(data_set, total=len(data_set)):
        if not sample['silent']:
            continue
        ref_identifier = get_reference_id_from_silent_sample(sample)
        reference_audio_sess_id_and_utt_id.add(ref_identifier)
    return reference_audio_sess_id_and_utt_id


def save_samples_of_data_set(
    emg_data_set: EMGDataset,
    root_path: Path,
    emg_sample_rate: int,
    soft_speech_units_sample_rate: int,
    dev_set_reference_identifiers: Set[Tuple[str, str, str,]],
    test_set_reference_identifiers: Set[Tuple[str, str, str]], 
    dry_run: bool = False,
    ignore_silent: bool = False,
):
    num_samples = len(emg_data_set)
    assert (emg_sample_rate % soft_speech_units_sample_rate) == 0
    unit_to_emg_ratio = emg_sample_rate // soft_speech_units_sample_rate

    for sample in tqdm.tqdm(emg_data_set, total=num_samples):
        silent = sample["silent"]
        utt_file_id = get_utterance_file_id_from_sample_dict(sample)
        if ignore_silent and silent:
            print(f"Warning: Ignoring silent sample: {utt_file_id}")
            continue

        ref_identifier = get_reference_id_from_silent_sample(sample)
        # Move audible train samples which were the references of dev or test files to "normal" dev / test splits
        if ref_identifier in dev_set_reference_identifiers:
            used_split_name = "valid"
            print(f"Moving sample with ID {ref_identifier} to data set valid")
        elif ref_identifier in test_set_reference_identifiers:
            used_split_name = "test"
            print(f"Moving sample with ID {ref_identifier} to data set test")
        else:
            used_split_name = "train" 
        
        emg_target_subset_dir  = root_path / used_split_name
        emg_target_subset_dir.mkdir(exist_ok=True, parents=True)
        
        phonemes = sample["phonemes"]
        
        if not silent:
            units = sample["speech_units"]
            mfccs = sample["mfccs"]
            audio = sample["audio"]
        else:
            units = sample["parallel_speech_units"]
            mfccs = sample["parallel_mfccs"]
            audio = sample["parallel_audio"]
        emg = sample["raw_emg"]
        emg_features = sample["emg_features"]
        if len(mfccs) % 2 == 1:
            mfccs = mfccs[:-1]  # Delete last element so that we have acoustic feats divisable by 2
        units = units [:(len(mfccs) // 2)]
        mfccs = mfccs[:2 * len(units)]  # Speech units are half the number of samples as acoustic feats -->
        
        if not silent:
            assert np.abs(len(mfccs) - len(emg_features)) <= 2, \
                f"More than two frames diff between MFCCS {mfccs.shape} and EMG feats {emg_features.shape}"
        
            if emg_features.shape[0] > mfccs.shape[0]:
                emg_features = emg_features[:mfccs.shape[0], :]
            elif mfccs.shape[0] > emg_features.shape[0]:
                mfccs = mfccs[:emg_features.shape[0], :]
                units = units[:len(mfccs)//2]
                emg = emg[len(units) * unit_to_emg_ratio]
            
            if not len(units) * unit_to_emg_ratio == len(emg):
                print(f"WARNING: Lengths do not match...")
                print(units.shape)
                print(unit_to_emg_ratio)
                print(emg.shape)
                assert(False)
            assert len(units) * unit_to_emg_ratio == len(emg)
            assert len(emg_features) ==  2 * len(units)
            
        assert len(units) == len(phonemes)

        for sub_dir_name, data in zip(
            ["emg", "phonemes", "units", "emg_feats", "mfccs"],
            [emg, phonemes, units, emg_features, mfccs]
        ):
            sub_dir = emg_target_subset_dir / sub_dir_name
            file_path = sub_dir / f"{utt_file_id}.pt"
            print(f"Saving data of shape {data.shape} to: {file_path}")
            print(f"{sub_dir_name} -- {data.shape} -> {file_path.absolute()}")
            if not dry_run:
                sub_dir.mkdir(exist_ok=True, parents=True)
                torch.save(data, file_path)

        # Save transcriptions
        transcriptions = sample["text"]
        sub_dir = emg_target_subset_dir / "transcriptions"
        file_path = sub_dir / f"{utt_file_id}.txt"
        print(f"{transcriptions} -> {file_path.absolute()}")
        if not dry_run:
            sub_dir.mkdir(exist_ok=True, parents=True)
            with open(file_path, '+w') as fp:
                fp.write(transcriptions)
                
        # Save audio
        audio_save_path = emg_target_subset_dir / "audio" / f"{utt_file_id}.wav"
        print(f"audio -- {audio.shape} -> {audio_save_path}")
        if not dry_run:
            audio_save_path.parent.mkdir(exist_ok=True, parents=True)
            sf.write(audio_save_path, audio.numpy(), samplerate=16_000)
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data_dir", type=Path,
                        default=Path("raw_data/emg_data/"))
    parser.add_argument("--text_alignment_dir", type=Path,
                        default=Path("raw_data/text_alignments/"))
    parser.add_argument("--testset_file", type=Path,
                        default=Path("raw_data/testset_largedev.json"))
    parser.add_argument("--target_dir", type=Path,
                        default=Path("data/gaddy_complete"))
    parser.add_argument("--emg_sr", type=int, default=800)
    parser.add_argument("--unit_sr", type=int, default=50)
    parser.add_argument("--dry_run", type=bool, default=False)
    
    args = parser.parse_args()
    target_dir = Path(args.target_dir)
    src_dir = Path(args.source_data_dir)
    # Load data sets
    # Setup train / dev / test sets
    silent_data_dirs = [src_dir / "silent_parallel_data"]
    voiced_data_dirs = [src_dir / "voiced_parallel_data", src_dir / "nonparallel_data"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## Init Soft HuBERT
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", ).to(device)

    init_emg_data = partial(
        EMGDataset, 
        text_align_directory=args.text_alignment_dir,
        testset_file=args.testset_file,
        silent_data_directories=silent_data_dirs,
        voiced_data_directories=voiced_data_dirs,
        hubert=hubert,
        device=device,
    ) 
    print(f"Setting up EMG dev data set")
    dev_set = init_emg_data(dev=True, test=False)
    print(f"Setting up EMG test data set")
    test_set = init_emg_data(dev=False, test=True,)

    print(f"Dev. Data set size: {len(dev_set)}")
    print(f"Test Data set size: {len(test_set)}")

    print(f"Setting up Dev-set and Test-Set reference IDs for the vocalized references")
    dev_set_ref_identifiers = get_silent_samples_to_reference_session_ids_and_utterance_ids(dev_set)
    test_set_ref_identifiers = get_silent_samples_to_reference_session_ids_and_utterance_ids(test_set)
    print(f"Setting up complete EMG data set (train+dev+test)")
    data_set_all = init_emg_data(dev=False, test=False, no_testset=True)
    print(f"Data set size: {len(data_set_all)}")
    
    print(f"Length of dev ref ID set size: {dev_set_ref_identifiers}")
    print(f"Length of test ref ID set size: {test_set_ref_identifiers}")
    
    print(f"Dev Set Ref Ids:")
    for ref_idx, ref_id in enumerate(dev_set_ref_identifiers):
        print(f"{ref_idx+1}: {ref_id}")
    print(f"Test set Ref Ids:")
    for ref_idx, ref_id in enumerate(test_set_ref_identifiers):
        print(f"{ref_idx+1}: {ref_id}")
    
    save_samples_of_data_set(
        data_set_all, target_dir,
        emg_sample_rate=args.emg_sr,
        soft_speech_units_sample_rate=args.unit_sr,
        dry_run=args.dry_run,
        dev_set_reference_identifiers=dev_set_ref_identifiers,
        test_set_reference_identifiers=test_set_ref_identifiers,
    )


    

if __name__ == "__main__":
    import sys
    main()
