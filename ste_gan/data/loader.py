"""Loaders for the EMGDataSet.
    Adapted from CarGAN repo: https://github.com/descriptinc/cargan
"""

import functools
import torch
import ste_gan
from omegaconf import DictConfig


###############################################################################
# Data loader
###############################################################################
from pathlib import Path
from ste_gan.data.emg_dataset import EMGDataset
from ste_gan.data.collate import ste_gan_collate


def loaders(
    data_root: Path, 
    strict=False,
    chunksize: int = ste_gan.CHUNK_SIZE,
    hopsize: int = ste_gan.HOPSIZE,
    train_emg_length: int = ste_gan.TRAIN_EMG_LENGTH,
    batch_size: int = ste_gan.BATCH_SIZE,
):
    """Setup data loaders"""
    train_data_set = EMGDataset(data_root, partition="train", 
                                strict=strict,
                                filter_by_length=True,
                                only_include_voiced=True,
                                train_emg_length=train_emg_length)
    val_data_set = EMGDataset(data_root, partition="valid",
                              session_id_to_idx=train_data_set.session_id_to_idx,
                              speaking_mode_id_to_idx=train_data_set.speaking_mode_id_to_idx,
                              only_include_voiced=True,
                              filter_by_length=True,
                              train_emg_length=train_emg_length,
                              strict=strict)
    test_data_set = EMGDataset(data_root, partition="test",
                               session_id_to_idx=train_data_set.session_id_to_idx,
                               speaking_mode_id_to_idx=train_data_set.speaking_mode_id_to_idx, 
                               only_include_voiced=True,
                               filter_by_length=False,
                               train_emg_length=train_emg_length,
                               strict=strict)
    
    EMGDataset.check_no_data_overlap([train_data_set, val_data_set, test_data_set])
    
    train_sampler = torch.utils.data.RandomSampler(train_data_set)
    val_sampler = torch.utils.data.SequentialSampler(val_data_set)
    test_sampler = torch.utils.data.SequentialSampler(test_data_set)

    # Create collate functions in train, eval and test mode
    collate_fn = functools.partial(
        ste_gan_collate,  
        emg_train_length=train_emg_length,
        chunksize=chunksize,
        hopsize=hopsize
    )

    train_collate_fn = collate_fn
    valid_collate_fn = functools.partial(collate_fn, partition='valid')
    test_collate_fn = functools.partial(collate_fn, partition='test')

    # Instantiate Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data_set,
        collate_fn=train_collate_fn,
        batch_size=batch_size,
        num_workers=ste_gan.NUM_WORKERS,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data_set,
        collate_fn=valid_collate_fn,
        batch_size=batch_size,
        num_workers=ste_gan.NUM_WORKERS,
        sampler=val_sampler,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_data_set,
        collate_fn=test_collate_fn,
        batch_size=1,
        num_workers=ste_gan.NUM_WORKERS,
        sampler=test_sampler,
    )

    return train_loader, val_loader, test_loader


def loaders_via_config(
    cfg: DictConfig
):
    data_root = Path(cfg.data.dataset_root)
    chunksize = cfg.train.chunk_size
    train_emg_length = chunksize # No autoregressive extra input
    strict = cfg.data.strict
    batch_size = cfg.train.batch_size
    
    return loaders(
        data_root=data_root, 
        strict=strict,
        chunksize=chunksize,
        hopsize=ste_gan.HOPSIZE,
        train_emg_length=train_emg_length,
        batch_size=batch_size
    )


if __name__ == "__main__":
    root_path = Path("/share/temp/kscheck/data/emg_gan/gaddy_voiced")
    train_loader, dev_loader, test_loader = loaders(root_path)
        
    def print_loader(loader, partition):
        num_batches = 0
        for batch_dict in loader:
            for data_type, data_batch in batch_dict.items():
                print(f"{data_type} -> {data_batch.shape}")
            num_batches += 1
        print(f"Num Batches: {num_batches} ({partition})")
    
    print_loader(train_loader, "train")
    print_loader(dev_loader, "valid")
    print_loader(test_loader, "test")

        
         