"""Main Training script for the STE-GAN.

Adapted from: https://github.com/descriptinc/cargan/blob/master/cargan/train.py
"""
import argparse
import functools
import itertools
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import ste_gan
from ste_gan.constants import DataType
from ste_gan.data.emg_dataset import EMGDataset
from ste_gan.data.loader import loaders_via_config
from ste_gan.utils.plot_utils import plot_real_vs_fake_emg_signal_with_envelope
from ste_gan.losses.emg_encoder_loss import (EMGEncoderLoss,
                                             EMGEncoderLossOutput)
from ste_gan.losses.time_domain_loss import MultiTimeDomainFeatureLoss
from ste_gan.models.discriminator import init_emg_discriminators
from ste_gan.models.emg_encoder import load_emg_encoder
from ste_gan.models.generator import init_emg_generator
from ste_gan.train_utils import (add_eval_hyperparams_to_parser,
                                 create_ste_gan_model_name, load_config,
                                 mean_error, phoneme_accuracy,
                                 phoneme_accuracy_no_silence)
from ste_gan.utils.common import load_latest_checkpoint


def train(
    cfg: DictConfig,
    model_directory: Path, 
    checkpoint: Path, 
    torch_device: str,
    debug: bool,
    emg_enc_ckpt: Path = None
):
    ###############
    # Load models #
    ###############
    device = torch.device(torch_device)

    logging.info(f"Initializing Models")
   
    netG = init_emg_generator(cfg)
    netD = init_emg_discriminators(cfg)
    
    logging.info(f"Initializing EMG Encoder Model with EMG encoder checkpoint: {emg_enc_ckpt}")
    emg_encoder = load_emg_encoder(cfg, device, emg_enc_ckpt)
    emg_encoder.eval()
    emg_encoder.to(device)
    
    logging.info(f"Initializing Losses")
    multi_td_loss = MultiTimeDomainFeatureLoss(cfg.data.num_emg_channels).to(device)
    emg_encoder_loss = EMGEncoderLoss(emg_encoder).to(device)

    multi_td_loss = multi_td_loss.to(device)
    netG.to(device)
    netD.to(device)
    
    ######################
    # Create tensorboard #
    ######################

    logging.info(f"Writing tensorboard logs to: {model_directory}")
    writer = SummaryWriter(str(model_directory.absolute()))

    #####################
    # Create optimizers #
    #####################
    optG = ste_gan.OPTIMIZER(netG.parameters())
    optD = ste_gan.OPTIMIZER(netD.parameters())

    #############################################
    # Maybe start from previous checkpoint      #
    #############################################
    if checkpoint is not None:
        logging.info(f"Loading checkpoint: {checkpoint}")
        netG, netD, optG, optD, start_epoch, steps = load_latest_checkpoint(
            checkpoint, device, netG, netD, optG, optD
        )
    else:
        start_epoch, steps = -1, 0

    #####################
    # Create schedulers #
    #####################

    scheduler_fn = functools.partial(
        torch.optim.lr_scheduler.ExponentialLR,
        gamma=.999,
        last_epoch=start_epoch if checkpoint is not None else -1
    )
    scheduler_g = scheduler_fn(optG)
    scheduler_d = scheduler_fn(optD)

    #######################
    # Create data loaders #
    #######################
    np.random.seed(cfg.train.random_seed)
    torch.cuda.manual_seed(cfg.train.random_seed)
    torch.manual_seed(cfg.train.random_seed)
    random.seed(cfg.train.random_seed)

    logging.info("Loading Data -- this can take a while")
    data_root = Path(cfg.data.dataset_root)
    logging.info(f"Data Set root: {data_root}")

    train_loader, valid_loader, test_loader = loaders_via_config(cfg)

    # Save Session and Speaking Mode ID mappings
    train_data_set: EMGDataset = train_loader.dataset
    train_data_set.save_session_and_speaking_mode_mapping_json(model_directory)

        
    #########
    # Train #
    #########
    logging.info(f"Starting Training")
    log_start = time.time()

    best_td_loss = np.inf
    best_su_loss = np.inf
    
    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    train_data_set: EMGDataset = train_loader.dataset
    valid_data_set: EMGDataset = valid_loader.dataset

    if int(torch. __version__[0]) >= 2:
        logging.info(f"Compiling models...PyTorch version: {torch.__version__}")
        netG = torch.compile(netG)
        netD = torch.compile(netD)
        emg_encoder = torch.compile(emg_encoder)
        #multi_td_loss = torch.compile(multi_td_loss)
        emg_encoder_loss = torch.compile(emg_encoder_loss)
    else:
        logging.warning(f"Will NOT compile models. Torch version: {torch. __version__}")
    
    # Mixed Precision Scaler    
    scaler = torch.cuda.amp.GradScaler()    
    speech_feature_type = cfg.model.speech_feature_type
    
    # Start training
    for epoch in itertools.count(start_epoch):
        logging.info(f"Starting epoch {epoch+1}")
        epoch_start_time = time.time()

        train_num_phones = 0
        train_num_phones_correct = 0
        
        train_num_silence = 0
        train_num_phones_correct_no_silence = 0
    
        for iterno, (batch_dict) in enumerate(train_loader):
            netD.zero_grad()
            netG.zero_grad()
            
            x_t = batch_dict[DataType.REAL_EMG].to(device)
            speech_units_t = batch_dict[DataType.SPEECH_UNITS].to(device)
            sess_idx = batch_dict[DataType.SESSION_INDEX].to(device)
            spk_mode_idx = batch_dict[DataType.SPEAKING_MODE_INDEX].to(device) 
            phoneme_targets = batch_dict[DataType.PHONEMES].to(device)

            if speech_feature_type == DataType.SPEECH_UNITS:
                s_t1 = speech_units_t
            else:    
                s_t1 = batch_dict[speech_feature_type].to(device)

            netG.train()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.train.mixed_precision):
                x_pred_t = netG(s_t1, sess_idx, spk_mode_idx)

                # Discriminator input
                d_t, d_pred_t = x_t, x_pred_t
                d_t_dis = d_t
                d_pred_t_dis = x_pred_t

                if cfg.train.loss_adversarial:
                    D_fake_det = netD(d_pred_t_dis.detach())
                    D_real = netD(d_t_dis)
                    loss_D = 0
                    for scale in D_fake_det:
                        loss_D += F.mse_loss(scale[-1], torch.zeros_like(scale[-1]))
                    for scale in D_real:
                        loss_D += F.mse_loss(scale[-1], torch.ones_like(scale[-1]))

                    scaler.scale(loss_D).backward()
                    scaler.step(optD)
                    writer.add_scalar("train_loss/discriminator", loss_D.item(), steps)

            # Train Generator #
            ###################
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.train.mixed_precision):
                loss_G = 0
                D_fake = netD(d_pred_t_dis)
                D_real = netD(d_t_dis)

                if cfg.train.loss_adversarial:
                    for scale in D_fake:
                        loss_G += F.mse_loss(scale[-1], torch.ones_like(scale[-1]))

                # L1 error on multi-time-domain features 
                if cfg.train.loss_multi_td_error:
                    td_error = multi_td_loss(d_t, d_pred_t)
                    loss_G += cfg.train.loss_multi_td_weight * td_error
                    writer.add_scalar("train_loss/multi_td", td_error.item(), steps)

                if cfg.train.loss_speech_unit_error or cfg.train.loss_phoneme_error:
                    emg_enc_loss_output: EMGEncoderLossOutput = emg_encoder_loss(d_pred_t, speech_units_t, phoneme_targets)
                
                    if cfg.train.loss_speech_unit_error:
                        su_loss = emg_enc_loss_output.speech_unit_loss
                        loss_G += cfg.train.loss_speech_unit_weight * su_loss
                        writer.add_scalar("train_loss/speech_unit", su_loss.item(), steps)
                    
                    if cfg.train.loss_phoneme_error:
                        phoneme_loss = emg_enc_loss_output.phoneme_loss
                        loss_G += cfg.train.loss_phoneme_weight * phoneme_loss
                        writer.add_scalar("train_loss/phoneme", phoneme_loss.item(), steps)
                        
                    num_phones = emg_enc_loss_output.num_phones
                    num_phones_correct = emg_enc_loss_output.num_correct_phones
                    
                    phoneme_acc_batch = phoneme_accuracy(num_phones, num_phones_correct)
                    writer.add_scalar("train_loss/phoneme_acc_batch", phoneme_acc_batch, steps)

                    num_phones_cocrect_no_sil = emg_enc_loss_output.num_correct_phones_no_silence
                    num_silence = emg_enc_loss_output.num_silence_phones
                    
                    phoneme_acc_batch_no_silence = phoneme_accuracy_no_silence(num_phones, num_phones_cocrect_no_sil, num_silence)
                    writer.add_scalar("train_loss/phoneme_acc_batch_no_sil", phoneme_acc_batch_no_silence, steps)
                    
                    train_num_phones += num_phones
                    train_num_phones_correct += num_phones_correct
                    
                    train_num_phones_correct_no_silence += num_phones_cocrect_no_sil
                    train_num_silence += num_silence
                    
                # MSE error on waveform
                if cfg.train.loss_waveform_error:
                    wave_loss = torch.nn.functional.mse_loss(d_pred_t, d_t)
                    loss_G += cfg.train.loss_waveform_weight * wave_loss
                    writer.add_scalar("train_loss/waveform", wave_loss.item(), steps)

                # Feature matching loss
                if cfg.train.loss_feat_match_error:
                    loss_feat = 0
                    for i in range(len(D_fake)):
                        for j in range(len(D_fake[i]) - 1):
                            loss_feat += \
                                F.l1_loss(D_fake[i][j], D_real[i][j].detach())
                    loss_G += cfg.train.loss_feat_match_weight * loss_feat
                    writer.add_scalar("train_loss/feature_matching", loss_feat.item(), steps)

            scaler.scale(loss_G).backward()
            scaler.step(optG)
            scaler.update()

            ###########
            # Logging #
            ###########
            writer.add_scalar("train_loss/generator", loss_G.item(), steps)

            if steps % cfg.train.interval_log == 0:
                phoneme_acc_train = phoneme_accuracy(train_num_phones, train_num_phones_correct)
                phoneme_acc_train_no_sil = phoneme_accuracy_no_silence(train_num_phones, train_num_phones_correct_no_silence, train_num_silence)
                
                log = (
                    f"Epoch {epoch} ({iterno}/{len(train_loader)}) | Steps {steps} | "
                    f"ms/batch {1e3 * (time.time() - log_start) / cfg.train.interval_log:5.2f} | Train Loss: {loss_G.item():.4f} | Ph. Acc. (Avg.) {phoneme_acc_train:.2f} | Ph. Acc. Avg. (No Sil) {phoneme_acc_train_no_sil:.2f}"
                )
                writer.add_scalar("train_loss/phoneme_accuracy_avg", phoneme_acc_train, steps)
                writer.add_scalar("train_loss/phoneme_accuracy_avg_no_sil", phoneme_acc_train_no_sil, steps)
                logging.info(log)
                log_start = time.time()
            
            ##############
            # Validation #
            ##############

            if steps % cfg.train.interval_valid == 0:
                with torch.no_grad():
                    logging.info(f"Starting validation")
                    val_start = time.time()
                    netG.eval()

                    td_errors = []
                    su_errors = []
                    phoneme_errors = []
                    
                    wave_errors = []

                    # Phoneme accuracy including silences
                    val_num_phones = 0
                    val_num_phones_correct = 0
                    # Phoneme Accuracy computation with ignoring silences
                    val_num_silence = 0
                    val_num_phones_correct_no_silence = 0
                    
                    for i, feat_batch in enumerate(valid_loader):
                        with torch.no_grad():
                            x_t = feat_batch[DataType.REAL_EMG].to(device)
                            s_t = feat_batch[speech_feature_type].to(device)
                            if speech_feature_type == DataType.SPEECH_UNITS:
                                speech_units_t = s_t
                            else:
                                speech_units_t = feat_batch[DataType.SPEECH_UNITS].to(device)

                            spk_mode_idx = feat_batch[DataType.SPEAKING_MODE_INDEX].to(device)
                            sess_idx = feat_batch[DataType.SESSION_INDEX].to(device)
                            phoneme_targets = feat_batch[DataType.PHONEMES].to(device)
                            
                            # Maybe split signal
                            s_t1 = s_t
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                x_pred_t = netG(s_t1, sess_idx, spk_mode_idx)

                                wave_errors.append(torch.nn.functional.mse_loss(x_pred_t, x_t).item())
                                td_errors.append(multi_td_loss(x_t, x_pred_t).item())
                                emg_enc_loss_output = emg_encoder_loss(x_pred_t, speech_units_t, phoneme_targets)
                            
                                su_errors.append(emg_enc_loss_output.speech_unit_loss.item())
                                phoneme_errors.append(emg_enc_loss_output.phoneme_loss.item())
                            
                            val_num_phones += emg_enc_loss_output.num_phones
                            val_num_phones_correct += emg_enc_loss_output.num_correct_phones
                            
                            val_num_silence += emg_enc_loss_output.num_silence_phones
                            val_num_phones_correct_no_silence += emg_enc_loss_output.num_correct_phones_no_silence
                            
                    # Calculate Mean Validation errors
                    avg_val_td_error = mean_error(td_errors)
                    avg_val_phoneme_error = mean_error(phoneme_errors)
                    avg_val_wave_error = mean_error(wave_errors)
                    avg_su_error = mean_error(su_errors)
                    avg_phoneme_accuracy = phoneme_accuracy(val_num_phones, val_num_phones_correct)
                    avg_phoneme_accuracy_no_sil = phoneme_accuracy_no_silence(val_num_phones, 
                                                                              val_num_phones_correct_no_silence,
                                                                              val_num_silence)
                    
                    # Log validation errors to tensorboard
                    writer.add_scalar("val_loss/speech_unit", avg_su_error, steps)
                    writer.add_scalar("val_loss/multi_td", avg_val_td_error, steps)
                    writer.add_scalar("val_loss/phoneme", avg_val_phoneme_error, steps)
                    writer.add_scalar("val_loss/phoneme_accuracy_avg", avg_phoneme_accuracy, steps)
                    writer.add_scalar("val_loss/phoneme_accuracy_avg_no_sil", avg_phoneme_accuracy_no_sil, steps)
                    writer.add_scalar("val_loss/waveform", avg_val_wave_error, steps)
                    
                    logging.info("-" * 100)
                    logging.info("Took %5.4fs to run validation loop" % (time.time() - val_start))
                    logging.info(f"\t - Avg. Val. Speech Unit Error : {avg_su_error}")
                    logging.info(f"\t - Avg. Val. Multi-TD Val. Error: {avg_val_td_error}")
                    logging.info(f"\t - Avg. Val. Phoneme Error: {avg_val_phoneme_error}")
                    logging.info(f"\t - Avg. Val. Phoneme Accuracy: {avg_phoneme_accuracy}")
                    logging.info(f"\t - Avg. Val. Phoneme Accuracy (No Sil.): {avg_phoneme_accuracy_no_sil}")
                    logging.info(f"\t - Avg. Val. Waveform Error: {avg_val_wave_error}")
                    logging.info("-" * 100)
                
                    if avg_su_error < best_su_loss:
                        best_su_loss = avg_su_error
                        logging.info(f"Saving best model with best val. SU error @ {best_su_loss:5.4f}...")
                        torch.save(netG.state_dict(), model_directory / "best_netG.pt")
                        torch.save(netD.state_dict(), model_directory / "best_netD.pt")

                    logging.info("-" * 100)
                    logging.info("Took %5.4fs to run validation loop" % (time.time() - val_start))
                    logging.info("-" * 100)

            ########################################
            # Generate samples                     #
            ########################################
            if (steps % cfg.train.interval_sample == 0):
                logging.info("Starting to generate validation samples for plotting")
                save_start = time.time()
                netG.eval()
                for i, (sample_dict) in enumerate(valid_data_set):
                    with torch.no_grad():
                        s_t = sample_dict[speech_feature_type].unsqueeze(0).to(device)
                        sess_idx = sample_dict[DataType.SESSION_INDEX].unsqueeze(0).to(device)
                        spk_mode_idx = sample_dict[DataType.SPEAKING_MODE_INDEX].unsqueeze(0).to(device) 

                        # Generate the fake EMG signal
                        pred_emg = netG.generate(s_t, sess_idx, spk_mode_idx).squeeze(0).detach().cpu().numpy()

                        # Generate the real EMG signal
                        real_emg = sample_dict[DataType.REAL_EMG].squeeze(0).detach().cpu().numpy()
                        
                        plot_real_vs_fake_emg_signal_with_envelope(
                            real_emg_signal=real_emg,
                            fake_emg_signal=pred_emg,
                            file_id=f"Validation sample {i}",
                            save_as=None,
                            tb_summary_writer=writer,
                            tb_tag_prefix="val/envelopes_emg_real_vs_fake",
                            global_step=steps,
                            show=False
                        )
                    if i > cfg.train.num_test_samples:
                        break
                
                logging.info("-" * 100)
                logging.info("Took %5.4fs to generate samples" % (time.time() - save_start))
                logging.info("-" * 100)


            ########################################
            # Save checkpoint                      #
            ########################################

            if steps > 0 and steps % cfg.train.interval_save == 0:
                save_start = time.time()
                logging.info("Starting to save models...")
                # Save checkpoint
                torch.save(
                    netG.state_dict(),
                    model_directory / f'netG-{steps:08d}.pt')
                torch.save(
                    netD.state_dict(),
                    model_directory / f'netD-{steps:08d}.pt')
                torch.save({
                    'epoch': epoch,
                    'steps': steps,
                    'optG': optG.state_dict(),
                    'optD': optD.state_dict(),
                }, model_directory / f'checkpoint-{steps:08d}.pt')

                logging.info('-' * 100)
                logging.info('Took %5.4fs to save checkpoint' % (time.time() - save_start))
                logging.info('-' * 100)
                
            if steps >= cfg.train.max_steps:
                logging.info(f"Finished training script. Starting saving last model") 
                logging.info("Starting to save models...")
                # Save checkpoint
                torch.save(
                    netG.state_dict(),
                    model_directory / f'netG-final.pt')
                torch.save(
                    netD.state_dict(),
                    model_directory / f'netD-final.pt')
                torch.save({
                    'epoch': epoch,
                    'steps': steps,
                    'optG': optG.state_dict(),
                    'optD': optD.state_dict(),
                }, model_directory / f'checkpoint-final.pt')

                logging.info('-' * 100)
                logging.info('Took %5.4fs to save checkpoint' % (time.time() - save_start))
                logging.info('-' * 100)

                logging.info(f"Writing a .done file in {model_directory}")
                with open(model_directory / ".done", '+w') as fp:
                    fp.write(f"done: {time.time()}")
                return
                        
            steps += 1

        if cfg.train.loss_adversarial:
            scheduler_d.step()
        scheduler_g.step()
        
        epoch_end_time = time.time()
        logging.info(f"Finished training epoch {epoch}. Elapsed time: {epoch_end_time - epoch_start_time}")
        
        # Save "last" model every 5 epochs
        if epoch % 5 == 0:
            save_start = time.time()
            logging.info(f"Finished training epoch {epoch}. Starting saving last model") 
            logging.info("Starting to save models...")
            # Save checkpoint
            torch.save(
                netG.state_dict(),
                model_directory / f'netG-last.pt')
            torch.save(
                netD.state_dict(),
                model_directory / f'netD-last.pt')
            torch.save({
                'epoch': epoch,
                'steps': steps,
                'optG': optG.state_dict(),
                'optD': optD.state_dict(),
            }, model_directory / f'checkpoint-last.pt')

            logging.info('-' * 100)
            logging.info('Took %5.4fs to save checkpoint' % (time.time() - save_start))
            logging.info('-' * 100)



###############################################################################
# Entry point
###############################################################################
def main(cfg: DictConfig, continue_run: bool, debug: bool, emg_enc_ckpt: Path, **kwargs):
    dataset_root = cfg.data.dataset_root
    print(f"Data root: {dataset_root}")
    print(f"continue_run: {continue_run}")
    print(f"Debug (argparse): {debug}")
    
    if not debug and cfg.train.debug:
        print(f"WARNING: SETTING GLOBAL DEBUG FLAG")
        debug = True
    
    # Create output dir
    model_base_dir = Path(cfg.model_base_dir)
    output_directory = model_base_dir/ create_ste_gan_model_name(
        cfg, add_timestamp=False, debug=debug,
    )
    if output_directory.exists() and continue_run:
        logging.info(f"WARNING: Removing old model directory: {output_directory}")
        checkpoint = output_directory
    else:
        checkpoint = None
    output_directory.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_directory}")

    done_file = output_directory / ".done"
    if output_directory and done_file.exists():
        logging.warning(f"Exiting training script as '.done' file exists: {done_file.absolute()}")
        sys.exit()

    # Save configuration
    config_file = output_directory / "config.yaml"
    if not config_file.exists():
        with open(config_file, '+w') as fp:
            OmegaConf.save(config=cfg, f=fp.name)

    logging.info(OmegaConf.to_yaml(cfg))
    logging.getLogger().setLevel(logging.INFO)
    log_file = output_directory / "log.txt"
    fh = logging.FileHandler(str(log_file.absolute()))
    logging.getLogger().addHandler(fh) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(cfg, output_directory, checkpoint, device, debug, emg_enc_ckpt)
    

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ste_gan_base_gantts.yaml",
                        help="The main training configuration for this run.")
    parser.add_argument("--data", type=str, default="configs/data/gaddy_and_klein_corpus.yaml",
                        help="A path to a data configuration file.")
    parser.add_argument("--emg_enc_cfg", type=str, default="configs/emg_encoder/conv_transformer.yaml",
                        help="A path to an EMG encoder configuration file.")
    parser.add_argument("--emg_enc_ckpt", type=str, default="exp/emg_encoder/EMGEncoderTransformer_voiced_only__seq_len__200__data_gaddy_complete/best_val_loss_model.pt",
                        help="A path to a checkpooint of a pre-trained EMG encoder. Must correspond to the EMG encoder configuration in 'emg_enc_cfg'.")
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Optional checkpoint to start training from')
    parser.add_argument(
        '--continue_run',
        action='store_true',
        help='Whether to continue training')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Whether to run the training script in debug mode.')
    
    parser = add_eval_hyperparams_to_parser(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    args.cfg = cfg
    main(**vars(args))
