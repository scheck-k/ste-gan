"""Training script of the EMG Encoder.
    
Adapted from: https://github.com/dgaddy/silent_speech/blob/main/transduction_model.py
"""
import logging
import os
import random
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from absl import flags
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import ste_gan
from ste_gan import PHONEME_INVENTORY, DataType
from ste_gan.data.emg_dataset import EMGDataset
from ste_gan.emg_encoder import constants as enc_constants
from ste_gan.emg_encoder.utils import (
    SizeAwareSampler, align_from_distances, collate_raw, combine_fixed_length,
    create_output_dir_name, decollate_tensor,
    init_voiced_datasets_emg_encoder_training)
from ste_gan.models.emg_encoder import EMGEncoderTransformer, init_emg_encoder
from ste_gan.train_utils import load_config


# Paper max len: 256000
@torch.no_grad()
def test(model: EMGEncoderTransformer, testset, device, acoustic_model: Optional[nn.Module] = None):
    model.eval()
    if acoustic_model is not None:
        acoustic_model.eval()
    dataloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=enc_constants.BATCH_SIZE, 
        collate_fn=collate_raw
    )
    losses = []
    accuracies = []
    phoneme_confusion = np.zeros((len(PHONEME_INVENTORY), 
                                  len(PHONEME_INVENTORY)))
    seq_len = enc_constants.SEQ_LEN
    with torch.no_grad():
        for batch in dataloader:
            emg_input = combine_fixed_length([t.to(device, non_blocking=True) for t in batch[DataType.REAL_EMG]], seq_len * 8)
            pred, phoneme_pred = model(emg_input)
            loss, phon_acc = speech_unit_loss_combined(pred, phoneme_pred, batch, True, phoneme_confusion)
            losses.append(loss.item())
            accuracies.append(phon_acc)
            if enc_constants.DEBUG:
                logging.warning(f"BREAKING TESTING LOOP BECAUSE OF DEBUG")
                break

    model.train()
    return np.mean(losses), np.mean(accuracies), phoneme_confusion 

def speech_unit_loss_combined(
    speech_unit_predictions,
    phoneme_predictions,
    batch, phoneme_eval=False, phoneme_confusion=None,
):
    device = speech_unit_predictions.device
    speech_unit_lengths = batch["speech_unit_lengths"]

    speech_unit_predictions_list = decollate_tensor(speech_unit_predictions, speech_unit_lengths)
    phoneme_predictions_list = decollate_tensor(phoneme_predictions, speech_unit_lengths)

    batch_size = len(speech_unit_lengths)
    assert len(speech_unit_predictions_list) == batch_size

    correct_phones = 0
    weight_speech_unit_loss = enc_constants.LOSS_WEIGHT_SPEECH_UNITS
    weight_phoneme_loss = enc_constants.LOSS_WEIGHT_PHONEMES

    speech_unit_targets_list = [t.to(device, non_blocking=True) for t in batch[DataType.SPEECH_UNITS]]
    assert len(
        speech_unit_targets_list) == batch_size, f"Speech unit target list is not batch size {len(speech_unit_targets_list)} vs. {batch_size})"
    total_num_phone_targets = 0

    losses = []
    su_loss_norm = float(enc_constants.SU_LOSS_NORM)

    for sample_idx in range(batch_size):
        speech_unit_pred = speech_unit_predictions_list[sample_idx]
        speech_unit_target = speech_unit_targets_list[sample_idx]

        phoneme_prediction = phoneme_predictions_list[sample_idx]
        phoneme_target = batch[DataType.PHONEMES][sample_idx].to(device)
        is_silent = batch["silent"][sample_idx]

        if not is_silent:
            assert speech_unit_target.size(0) == speech_unit_pred.size(0)
            speech_unit_dists = F.pairwise_distance(speech_unit_target, speech_unit_pred, p=su_loss_norm)
            speech_unit_loss = speech_unit_dists.mean()

            phoneme_loss = F.cross_entropy(phoneme_prediction, phoneme_target, reduction='mean')
            # Total loss
            loss = (
                    (weight_speech_unit_loss * speech_unit_loss)
                    + (weight_phoneme_loss * phoneme_loss)
            )
            losses.append(loss)

            if phoneme_eval:
                pred_phone = phoneme_prediction.argmax(-1)
                correct_phones += (pred_phone == phoneme_target).sum().item()
                total_num_phone_targets += len(phoneme_target)

                for p, t in zip(pred_phone.tolist(), phoneme_target.tolist()):
                    phoneme_confusion[p, t] += 1

        else:
            # Speech Unit loss
            speech_unit_dists = torch.cdist(speech_unit_pred.unsqueeze(0), speech_unit_target.unsqueeze(0),
                                            p=su_loss_norm)
            speech_unit_costs = speech_unit_dists.squeeze(0)

            # phone_probs (seq1_len, seq2_len)
            pred_phone = F.log_softmax(phoneme_prediction, -1)
            phone_lprobs = pred_phone[:, phoneme_target]

            speech_unit_with_phone_costs = weight_speech_unit_loss * speech_unit_costs + weight_phoneme_loss * -phone_lprobs
            alignment = align_from_distances(speech_unit_with_phone_costs.T.cpu().detach().numpy())
            
            loss = speech_unit_with_phone_costs[alignment, range(len(alignment))].sum() / len(speech_unit_target)
            losses.append(loss)

            if phoneme_eval:
                alignment = align_from_distances(speech_unit_with_phone_costs.T.cpu().detach().numpy())

                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone[alignment] == phoneme_target).sum().item()
                total_num_phone_targets += len(phoneme_target)

                for p, t in zip(pred_phone[alignment].tolist(), phoneme_target.tolist()):
                    phoneme_confusion[p, t] += 1

    batch_loss = sum(losses) / batch_size
    if phoneme_eval:
        phone_acc = correct_phones / total_num_phone_targets
    else:
        phone_acc = float("nan")

    return batch_loss, phone_acc

def train_model(cfg: omegaconf.DictConfig,  trainset: EMGDataset, devset: EMGDataset, device, output_directory: Path, debug: bool = False):
    n_epochs = enc_constants.NUM_EPOCHS
    training_subset = trainset    
    dataloader = torch.utils.data.DataLoader(
        training_subset, pin_memory=(device == 'cuda'),
        collate_fn=collate_raw, num_workers=0,
        batch_sampler=SizeAwareSampler(training_subset,  
                                       enc_constants.TRAIN_BATCH_MAX_LEN))
    
    model = init_emg_encoder(cfg,device)
   
    optim = torch.optim.AdamW(
        list(model.parameters()), 
        weight_decay=enc_constants.WEIGHT_DECAY,
    )
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', 0.5, patience=enc_constants.LEARNING_RATE_PATIENCE)
    
    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

    target_lr = enc_constants.LEARNING_RATE

    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= enc_constants.LEARNING_RATE_WARMUP:
            set_lr(iteration * target_lr / enc_constants.LEARNING_RATE_WARMUP)

    seq_len = enc_constants.SEQ_LEN
    best_val_loss = float("inf")

    from datetime import datetime
    batch_idx = 0
    num_no_improvement = 0
    
    # Mixed Precision Scaler    
    scaler = torch.cuda.amp.GradScaler()
    
    writer = SummaryWriter(str(output_directory.absolute()))

    # Compile models with Torch2
    #if int(torch. __version__[0]) >= 2:
    #    logging.info(f"Compiling models...PyTorch version: {torch.__version__}")
    #    model = torch.compile(model)
    
    global_step = 0
    for epoch_idx in range(n_epochs):
        logging.info(f"{datetime.now()} Starting new epoch: {epoch_idx + 1}")
        losses = []

        for batch in tqdm(dataloader):
            optim.zero_grad()
            schedule_lr(batch_idx)
            emg_input = combine_fixed_length([t.to(device, non_blocking=True) 
                                              for t in batch[DataType.REAL_EMG]], 
                                             seq_len * 8)
        
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred, phoneme_pred = model(emg_input)
                loss, phon_acc = speech_unit_loss_combined(pred, phoneme_pred, batch)
                train_loss_item = loss.item()
                losses.append(train_loss_item)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            
            batch_idx += 1
            writer.add_scalar("train/loss", train_loss_item, global_step)
            writer.add_scalar("train_loss/phon_acc", phon_acc, global_step)
            global_step += 1

            if enc_constants.DEBUG or debug:
                logging.warning(f"BREAKING TESTING LOOP BECAUSE OF DEBUG")
                break
        
        train_loss = np.mean(losses)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            val, phoneme_acc, _ = test(model, devset, device)
        
        writer.add_scalar("val/loss", val, global_step)
        writer.add_scalar("val/phon_acc", phoneme_acc, global_step)
        
        lr_sched.step(val)
        logging.info(f"{datetime.now()} Finished: {epoch_idx + 1}")
        logging.info(
            f'finished epoch {epoch_idx + 1} - training loss: {train_loss:.4f} | Train Phoneme Acc: {phon_acc:.4f} |  validation loss: {val:.4f}  |  val. phoneme accuracy: {phoneme_acc * 100:.2f}')
        
        if val < best_val_loss:
            logging.info(f"Saving model as val loss improved.")
            torch.save(model.state_dict(), os.path.join(str(output_directory), f'best_val_loss_model.pt'))
            best_val_loss = float(val)
            num_no_improvement = 0
        else:
            num_no_improvement += 1

        torch.save(model.state_dict(), os.path.join(str(output_directory), 'last_model.pt'))

        if enc_constants.DEBUG or debug:
            logging.warning(f"BREAKING OVERALL TRAINING LOOP BECAUSE OF DEBUG")
            break

        if num_no_improvement > enc_constants.EARLY_STOP_PATIENCE:
            logging.warning(f"BREAKING TRAINING LOOP BECAUSE NO IMPROVEMENT AFTER 10 EPOCHS OF TRAINING")
            break

    return model


def main(
    cfg: omegaconf.DictConfig,
    exp_dir: Path,
    debug: bool = False,
):
    emg_dataset_root = Path(cfg.data.dataset_root)
    data_set_roots = [emg_dataset_root]
    
    output_dir_name = create_output_dir_name(
        data_set_roots,
        debug=debug, 
        emg_enc_name=cfg.emg_encoder.type + f"_voiced_only"
    )
    output_directory = exp_dir / output_dir_name
    print(f"Initializing experiment in: {output_directory}")

    output_directory.mkdir(exist_ok=True, parents=True)
    done_file = output_directory / ".done"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if done_file.exists():
        logging.warning(f"Exiting training script as '.done' file exists: {done_file.absolute()}")
        sys.exit()

    logging.getLogger().setLevel(logging.INFO)
    log_file = output_directory / "log.txt"
    fh = logging.FileHandler(str(log_file.absolute()))
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    logging.info(f"+++ STARTING TRAINING +++")

    # logging.info(subprocess.run(['git','rev-parse','HEAD'], stdout=subprocess.PIPE, universal_newlines=True).stdout)
    # logging.info(subprocess.run(['git','diff'], stdout=subprocess.PIPE, universal_newlines=True).stdout)
    logging.info(sys.argv)

    # Save configuration
    config_file = output_directory / "config.yaml"
    logging.info(f"Saving configuration file under: {config_file}")
    if not config_file.exists():
        with open(config_file, '+w') as fp:
            OmegaConf.save(config=cfg, f=fp.name)
    
    logging.info(f"Loading EMG data from data root: {emg_dataset_root}")
    trainset, devset, _ = init_voiced_datasets_emg_encoder_training(emg_dataset_root)
    
    logging.info(f'output directory: {output_directory.absolute()}')
    logging.info('output example: %s', devset[0])
    logging.info(f'train / dev split: {len(trainset)} / {len(devset)}')

    _example_sample = devset[0]
    for key in _example_sample:
        val = _example_sample[key]
        if isinstance(val, (torch.Tensor, np.ndarray)):
            logging.info(f"{key} -> {val.shape}")
        else:
            logging.info(f"{key} -> {val}")

    logging.info(f"Loaded EMG data from data root: {emg_dataset_root}")

    seed = ste_gan.RANDOM_SEED
    logging.info(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logging.info(f"Model output path: {output_directory}")

    model = train_model(cfg, trainset, devset, device, output_directory, debug=debug)

    logging.info(f"Finished training the model")
    logging.info(f"Creating a '.done' file: {done_file.absolute()}")
    with open(done_file, 'w') as fp:
        fp.write("Done training.\n")
        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ste_gan_base_gantts.yaml")
    parser.add_argument("--exp_dir", type=Path, default=Path("exp/emg_encoder"))
    parser.add_argument("--data", type=str, default="configs/data/gaddy_and_klein_corpus.yaml")
    parser.add_argument("--emg_enc_cfg", type=str, default="configs/emg_encoder/conv_transformer.yaml")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )    
    args = parser.parse_args()
    cfg = load_config(args, override_with_ste_gan_eval_args=False)
    
    main(
        cfg,
        args.exp_dir,
        args.debug,
    )
