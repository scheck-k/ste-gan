import json
import logging
import sys
import torch
from functools import lru_cache
from pathlib import Path
import ste_gan
from typing import *
import pandas as pd



class EMGDataset(torch.utils.data.Dataset):
    """Implements the data set for loading EMG signals along with features obtained from parallel audio. """

    def __init__(
        self,
        root_dir: Path,
        partition: str = "train",
        session_id_to_idx: Optional[Dict] = None,
        speaking_mode_id_to_idx: Optional[Dict] = None,
        only_include_voiced: bool = True,
        only_include_silent: bool = False,
        filter_by_length: bool = True,
        strict: bool = False,
        return_mfccs: bool = True,
        return_emg_feats: bool = True,
        train_emg_length: int = ste_gan.TRAIN_EMG_LENGTH,
    ):
        """Initializes the data set.

        Args:
            root_dir (Path): The root directory in which data is saved.
            partition (str, optional): The data split / partition. Corresponds to a subdir. Defaults to "train".
            session_id_to_idx (Optional[Dict], optional): A mapping of session Ids to session indices.
            If not supplied, it will be computed using the pre-loaded data. Defaults to None.
            speaking_mode_id_to_idx (Optional[Dict], optional): A mapping of speaking mode IDs to indices.
            If not supplied, it wil lbe computed using the pre-loaded data of the split. Defaults to None.
            strict (bool): If True, the length of EMG signals, speech units, and phonemes will be checked upon loading.
        """
        self.partition = partition
        self.root_dir = root_dir
        self.split_dir = root_dir / partition
        self.strict = strict
        self.return_mfccs = return_mfccs
        self.return_emg_feats = return_emg_feats
        
        is_test = self.partition == "test"
        
        assert not(only_include_silent and only_include_voiced), "Either only_include_silent or only_include_voiced can be True."
        
        # Load sorted paths to all EMG files
        self.emg_dir = (self.split_dir / ste_gan.DataDir.EMG)
        
        emg_paths = list(sorted(self.emg_dir.glob("*.pt")))
        

        self.utt_ids = []
        
        def filter_emg_paths(in_emg_paths: List[Path]) -> Tuple[List[Path], List[int], int]:
            out_paths = []
            out_lens = []
            num_filtered = 0
            for emg_path in in_emg_paths:
                emg_len = len(torch.load(emg_path))
                if filter_by_length and emg_len < train_emg_length and not is_test:
                    num_filtered += 1
                    continue
                
                utt_id = emg_path.stem
                speaking_mode = self.utt_id_to_spk_mode_id(utt_id)
                is_silent = speaking_mode != ste_gan.SpeakingMode.NORMAL
                if only_include_voiced and is_silent:
                    num_filtered += 1
                    continue

                if only_include_silent and not is_silent:
                    num_filtered += 1
                    continue
                
                out_paths.append(emg_path)
                out_lens.append(emg_len)
            
            return out_paths, out_lens, num_filtered
        
        
        self.emg_paths, self.emg_lengths, self.num_filtered  = filter_emg_paths(emg_paths)
        logging.info(f"Filtered {self.num_filtered} EMG paths for partition {partition} due to max. EMG length"
                     f" and/or speaking mode (only_include_voiced={only_include_voiced}).")
        
        logging.info(f"Filtered {self.num_filtered} EMG paths for partition {partition} due to max. EMG length"
                     f" and/or speaking mode (only_include_voiced={only_include_voiced}).")

        logging.info(f"Total number of EMG paths in data set {partition}: {len(self.emg_paths)}")

        # Get Utterance IDs from the EMG paths
        self.utt_ids = [emg_path.stem for emg_path in self.emg_paths]
        
        self.file_ids = [self.get_file_id_stem(emg_path) for emg_path in self.emg_paths] 

        # Load the transcripts as txt file
        self.transcripts = self.load_transcripts(self.emg_paths, self.file_ids)

        # Setup paths to speech units
        self.speech_unit_paths = [
            emg_path.parents[1] / ste_gan.DataDir.SPEECH_UNITS / f"{file_id}.pt"
            for (file_id, emg_path) in zip(self.file_ids, self.emg_paths)
        ]
        

        # Setup paths to phoneme sequences
        self.phoneme_paths = [
            emg_path.parents[1] / ste_gan.DataDir.PHONEMES / f"{file_id}.pt"
            for (file_id, emg_path) in zip(self.file_ids, self.emg_paths)
        ]
        
        # Setup EMG feature paths 
        self.emg_feat_paths = [
            self.split_dir / ste_gan.DataDir.EMG_FEATS / f"{utt_id}.pt"
            for utt_id in self.utt_ids 
        ]
        
        # Setup paths to MFCCs
        self.mfcc_paths = [
            self.split_dir / ste_gan.DataDir.MFCCS / f"{utt_id}.pt"
            for utt_id in self.utt_ids 
        ] 
        
        # Map Utterance IDs to the session IDs and speaking modes
        self.session_ids = [self.utt_id_to_session_id(utt_id) for utt_id in self.utt_ids]
        self.speaking_mode_ids = [self.utt_id_to_spk_mode_id(utt_id) for utt_id in self.utt_ids]
        
        # Setup a mapping of session IDs to an integer
        if not session_id_to_idx:
            self.session_id_to_idx = {sess_id: idx for idx, sess_id in enumerate(sorted(set(self.session_ids)))}
        else:
            self.session_id_to_idx = session_id_to_idx
        
        # Setup a mapping of speaking modes to an integer
        if not speaking_mode_id_to_idx:
            self.speaking_mode_id_to_idx = {spk_mode: idx for idx, spk_mode in enumerate(sorted(set(self.speaking_mode_ids)))}
        else:
            self.speaking_mode_id_to_idx = speaking_mode_id_to_idx
        
        # Setup a list of session indices
        self.session_indices = [torch.tensor(self.session_id_to_idx[sess_id]) for sess_id in self.session_ids]
        self.speaking_mode_indices = [torch.tensor(self.speaking_mode_id_to_idx[spk_mode_id]) for spk_mode_id in self.speaking_mode_ids]
        
        # Setup a reverse mapping of session indices / speaking mode indices to their strings
        self.session_idx_to_id = {idx: sess_id for sess_id, idx in self.session_id_to_idx.items()}
        self.speaking_mode_idx_to_id = {idx: spk_mode for spk_mode, idx in self.speaking_mode_id_to_idx.items()}
        
    def load_transcripts(self, emg_paths: List[Path], file_ids: List[str]) -> List[str]:
        """Loads the transcripts of the uttearncs.

        Args:
            utt_ids (List[str]): The list of utterance Ids which transcripts should be loaded.

        Returns:
            List[str]: A list of strings representing the transcripts in lower case.
        """
        transcripts = []
        for file_id, emg_path in zip(file_ids, emg_paths):
            txt_file = emg_path.parents[1] / ste_gan.DataDir.TRANSCRIPTIONS / f"{file_id}.txt"
            transcript = txt_file.read_text().strip().lower()
            transcripts.append(transcript)
        return transcripts
    
            
    def __len__(self):
        return len(self.utt_ids)
    
    @property
    def num_sessions(self) -> int:
        return len(self.session_idx_to_id)
    
    @property
    def num_speaking_modes(self) -> int:
        return len(self.speaking_mode_id_to_idx)
    
    @property
    def num_emg_channels(self) -> int:
        if not self.emg_paths:
            raise Exception(f"No EMG paths in the Data set -- cannot determine the number of channels.")
        return torch.load(self.emg_paths[0]).shape[-1]
    
    def utt_id_to_session_id(self,utt_id: str) -> int:
        parts = utt_id.split("__")
        return parts[0]
    
    def utt_id_to_spk_mode_id(self, utt_id: str) -> int:
        parts = utt_id.split("__")
        return parts[-1]
    
    def get_file_id_stem(self, emg_path: Path, emg_synth_prefix: str = "emg_synth__") -> str:
        if emg_synth_prefix in emg_path.parent.name:
           parts = emg_path.stem.split("__")
           return parts[1]
        else:
            return emg_path.stem
    
    def save_session_and_speaking_mode_mapping_json(
        self,
        save_dir: Path
    ):
        session_idx_to_id = self.session_idx_to_id
        speaking_mode_idx_to_id = self.speaking_mode_idx_to_id
    
        session_idx_to_id_path = save_dir / f"session_idx_to_id.json"
        speaking_mode_idx_to_id_path = save_dir / f"speaking_mode_idx_to_id.json"
    
        logging.info(f"Saving Session ID to Index to {session_idx_to_id_path}")
        logging.info(f"Saving Speaking Mode ID to Index to {speaking_mode_idx_to_id_path}")

        with open(session_idx_to_id_path, '+w') as fp:
            json.dump(session_idx_to_id, fp)
        with open(speaking_mode_idx_to_id_path, '+w') as fp:
            json.dump(speaking_mode_idx_to_id, fp)

    @staticmethod
    def check_no_data_overlap(emg_data_set_list):
        utt_id_set_list = [set(dataset.utt_ids) for dataset in emg_data_set_list]
        intersect_utt_ids = set.intersection(*utt_id_set_list)
        if intersect_utt_ids:
            raise ValueError(f"Intersecting utterance ids: {','.join(intersect_utt_ids)}")
        logging.info(f"Found no overlap in uterance IDs between data set splits.")
    
    @lru_cache(maxsize=None)
    def __getitem__(self, index) -> Dict:
        # Load the EMG signal of the utterance
        emg_path = self.emg_paths[index]
        real_emg = torch.load(emg_path)
    
        # Load Soft speech units
        units_path = self.speech_unit_paths[index]
        units = torch.load(units_path)
        
        # Load phonemes
        phonemes_path = self.phoneme_paths[index]
        phonemes = torch.load(phonemes_path)
        
        if self.return_mfccs:
            mfcc_path = self.mfcc_paths[index]
            mfccs = torch.load(mfcc_path)
        else:
            mfccs = None
            
        if self.return_emg_feats:
            emg_feat_path = self.emg_feat_paths[index]
            emg_feats = torch.load(emg_feat_path)
        else:
            emg_feats = None
        
        # Check the length of signal / feature sequences
        if self.strict:
            # Check that the length of speech units and phonemes matches
            assert len(units) == len(phonemes)
            # Check that the length of the EMG signal matches the length of phonemes and speehc units
            expected_units_len = ste_gan.HOPSIZE * len(units)
            expected_phonemes_len = ste_gan.HOPSIZE * len(phonemes)
            assert len(real_emg) == expected_units_len
            assert len(real_emg) == expected_phonemes_len
        
        # Get the pre-loaded data
        utt_id = self.utt_ids[index]
        transcription = self.transcripts[index]
        
        # Get the session ID and speaking mode
        session_id = self.session_ids[index]
        spk_mode_id = self.speaking_mode_ids[index]

        # Get index-values of the identifiers of the session and speaking mode
        session_index = self.session_indices[index]
        spk_mode_index = self.speaking_mode_indices[index]
        
        # Return a dictioanry with the data
        return {
            ste_gan.DataType.UTT_ID: utt_id,
            ste_gan.DataType.TRANSCRIPTION: transcription,
            # EMG signal
            ste_gan.DataType.REAL_EMG: real_emg,
            # Phonemes and speech units -- GT values obtained from parallel audio
            ste_gan.DataType.PHONEMES: phonemes,
            ste_gan.DataType.SPEECH_UNITS: units,
            # Session Info
            ste_gan.DataType.SESSION_ID: session_id,
            ste_gan.DataType.SESSION_INDEX: session_index,
            # Speaking mode info
            ste_gan.DataType.SPEAKING_MODE_ID: spk_mode_id,
            ste_gan.DataType.SPEAKING_MODE_INDEX: spk_mode_index,
            # Optionally return MFCCs and EMG features for the baseline model
            ste_gan.DataType.MFCCS: mfccs,
            ste_gan.DataType.EMG_FEATURES: emg_feats,
        }


if __name__ == "__main__":
    root_path = Path("data/gaddy_complete")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    
    train_data_set = EMGDataset(root_path, partition="train", strict=True,)
    
    print(f"Number of EMG utterances: {len(train_data_set)}")
    dev_data_set = EMGDataset(
        root_path, partition="valid",
        session_id_to_idx=train_data_set.session_id_to_idx,
        speaking_mode_id_to_idx=train_data_set.speaking_mode_id_to_idx,
        strict=True
    )
    test_data_set = EMGDataset(
        root_path, partition="test",
        session_id_to_idx=train_data_set.session_id_to_idx,
        speaking_mode_id_to_idx=train_data_set.speaking_mode_id_to_idx,
        strict=True,
    )
    session_id_to_idx = train_data_set.session_id_to_idx
    print(f"Num Sessions: {len(session_id_to_idx)}")
    
    def print_sample(sample: Dict):
        print(f"Utt: {sample[ste_gan.DataType.UTT_ID]}")
        print(f"EMG: {sample[ste_gan.DataType.REAL_EMG].shape}")
        print(f"Units: {sample[ste_gan.DataType.SPEECH_UNITS].shape}")
        print(f"Text: {sample[ste_gan.DataType.TRANSCRIPTION]}")
        print(f"Sess ID: {sample[ste_gan.DataType.SESSION_ID]}")
        print(f"Sess IDX: {sample[ste_gan.DataType.SESSION_INDEX]}")
        pass
    
    for data_set, partition in zip((train_data_set, dev_data_set, test_data_set),
                        ("train", "valid", "test")):
        print(f"Data set {partition}: {len(data_set)}")
    
    for data_set, partition in zip((train_data_set, dev_data_set, test_data_set),
                        ("train", "valid", "test")):
        
        for sample in reversed(data_set):
            print_sample(sample)
    
    
