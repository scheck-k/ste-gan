import argparse
import glob
import logging
import os
from collections import OrderedDict
from typing import Dict

import matplotlib
import numpy as np
from typing import Optional

matplotlib.use("Agg")
import time
from pathlib import Path

import matplotlib.pylab as plt
import yaml
from omegaconf import DictConfig

import ste_gan


def override_constants_with_args(args):
    print(f"Overriding constants with command-line arguments")
    print(f"LOSS_SPEECH_UNIT_WEIGHT\t\t{ste_gan.LOSS_SPEECH_UNIT_WEIGHT} -> {args.weight_su}")
    print(f"LOSS_PHONEMES_WEIGHT\t\t{ste_gan.LOSS_PHONEMES_WEIGHT} -> {args.weight_phoneme}")
    print(f"LOSS_FEAT_MATCH_WEIGHT\t\t{ste_gan.LOSS_FEAT_MATCH_WEIGHT} -> {args.weight_feat_match}")
    print(f"LOSS_MULTI_TD_ERROR_WEIGHT\t\t{ste_gan.LOSS_MULTI_TD_ERROR_WEIGHT} -> {args.weight_td}")
    print(f"NUM_EMG_CHANNELS\t\t{ste_gan.NUM_EMG_CHANNELS} -> {args.num_emg_ch}")
    print(f"NUM_EMG_SESSIONS\t\t{ste_gan.NUM_EMG_SESSIONS} -> {args.num_emg_sess}")

    ste_gan.LOSS_SPEECH_UNIT_WEIGHT = args.weight_su
    ste_gan.LOSS_PHONEMES_WEIGHT = args.weight_phoneme
    ste_gan.LOSS_FEAT_MATCH_WEIGHT = args.weight_feat_match
    ste_gan.LOSS_MULTI_TD_ERROR = args.weight_td
    ste_gan.NUM_EMG_CHANNELS = args.num_emg_ch
    ste_gan.NUM_EMG_SESSIONS = args.num_emg_sess
    
    # Reset some loss application booleans
    if ste_gan.LOSS_PHONEMES_WEIGHT < 0.0001:
        ste_gan.LOSS_PHONEMES_ERROR = False
        print(f"Setting LOSS_PHONEMES_ERROR to False")
    if ste_gan.LOSS_SPEECH_UNIT_WEIGHT < 0.0001:
        ste_gan.LOSS_SPEECH_UNIT_ERROR = False
        print(f"Setting LOSS_SPEECH_UNIT_ERROR to False")


def override_config_with_eval_args(cfg: Dict, args) -> Dict:
    print(f"Overriding config with command-line arguments")
    if args.weight_su >= 0.0:
        cfg['train']['loss_speech_unit_weight'] = args.weight_su 
        print(f"cfg.train.loss_speech_unit_weight\t\t{cfg['train']['loss_speech_unit_weight']} -> {args.weight_su}")

    if args.weight_phoneme >= 0.0:
        print(f"cfg.train.loss_phoneme_weight\t\t{cfg['train']['loss_phoneme_weight']} -> {args.weight_phoneme}")
        cfg['train']['loss_phoneme_weight'] = args.weight_phoneme
        
    if args.weight_td >= 0.0:
        print(f"cfg.train.loss_multi_td_weight\t\t{cfg['train']['loss_multi_td_weight']} -> {args.weight_td}")
        cfg['train']['loss_multi_td_weight'] = args.weight_td
        
    if args.weight_feat_match >= 0.0:
        print(f"cfg.train.loss_feat_match_weight\t\t{cfg['train']['loss_feat_match_weight']} -> {args.weight_feat_match}")
        cfg['train']['loss_feat_match_weight'] = args.weight_feat_match
    
    if args.speech_feature_type.strip():
        print(f"cfg.model.speech_feature_type\t\t{cfg['model']['speech_feature_type']} -> {args.speech_feature_type}")
        cfg['model']['speech_feature_type'] = args.speech_feature_type
        
    if args.chunk_size > 0.0:
        print(f"cfg.train.chunk_size\t\t{cfg['train']['chunk_size']} -> {args.chunk_size}")
        cfg['train']['chunk_size'] = args.chunk_size
        
    if args.batch_size > 0.0:
        print(f"cfg.train.batch_size\t\t{cfg['train']['batch_size']} -> {args.batch_size}")
        cfg['train']['batch_size'] = args.batch_size
        
    if args.max_steps > 0:
        print(f"cfg.train.max_steps\t\t{cfg['train']['max_steps']} -> {args.max_steps}")
        cfg['train']['max_steps'] = args.max_steps
    
    # Reset some loss application booleans
    if cfg['train']['loss_speech_unit_weight'] < 0.001:
        cfg['train']['loss_speech_unit_error'] = False
        print(f"Setting cfg.train.loss_speech_unit_error to False")
    
    if cfg['train']['loss_phoneme_weight'] < 0.001:
        cfg['train']['loss_phoneme_error'] = False
        print(f"Setting cfg.train.loss_phoneme_error to False")

    return cfg
    
    
def mean_error(error_list) -> np.ndarray:
    return np.asarray(error_list).mean(0)


def phoneme_accuracy(num_phones: int, num_correct: int) -> float:
    return 100.0 * (num_correct / num_phones) if num_phones > 0 else float("nan")


def phoneme_accuracy_no_silence(num_phones_total: int, num_correct_no_silence: int, num_silence: int) -> float:
    num_phones = num_phones_total - num_silence
    return phoneme_accuracy(num_phones, num_correct_no_silence)

        
def create_ste_gan_model_name(
    cfg: DictConfig,
    add_timestamp: bool = True,
    debug: bool = False,
    note: str = ""
):
    """
    Creates a model name based on the hyperparameters used to train a STE-GAN model.
    """
    if note:
        note += "_"
    
    dataset_name = cfg.data.name
    td_weight = cfg.train.loss_multi_td_weight
    
    su_weight = cfg.train.loss_speech_unit_weight
    ph_weight = cfg.train.loss_phoneme_weight
    waveform_weight = cfg.train.loss_waveform_weight
    fm_weight = cfg.train.loss_feat_match_weight

    use_adv_str = "with_adv_loss" if cfg.train.loss_adversarial else "no_adv_loss"

    debug_str = "" if not debug else "DEBUG_"
    timestr = time.strftime("%Y%m%d-%H%M%S") + "_"
    timestamp_str = "" if debug or not add_timestamp else timestr

    emg_generator_name = cfg.model.type
    speech_feat_type = cfg.model.speech_feature_type
    small_dis = "small_dis" if cfg.model.discriminator_small else "full_dis" 

    return f"{note}{debug_str}{timestamp_str}{dataset_name}_{emg_generator_name}_{speech_feat_type}_{small_dis}_chunk_{cfg.train.chunk_size}_{use_adv_str}_fmw_{fm_weight}_tdw_{td_weight}_suw_{su_weight}_phw_{ph_weight}_wv_{waveform_weight}"


def add_eval_hyperparams_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Loss Weights
    parser.add_argument(
        "--weight_su",
        type=float,
        default=-1.0,
        help='Weight of the speech unit loss of the EMG encoder (a value smaller 0.0 means this setting is ignored)'
    )
    parser.add_argument(
        "--weight_phoneme",
        type=float,
        default=-1.0,
        help='Weight of the phoneme loss of the EMG encoder. (a value smaller than 0.0 means this argument is ignored)'
    )
    parser.add_argument(
        "--weight_td",
        type=float,
        default=-1.0,
        help='Weight of the the Multi-Time-Domain loss. (a value smaller than 0.0 means this argument is ignored)'
    )
    parser.add_argument(
        "--weight_feat_match",
        type=float,
        default=-1.0,
        help='Weight of the Feature matching loss. (a value smaller than 0.0 means this argument is ignored)'
    )
    parser.add_argument(
        "--speech_feature_type",
        type=str,
        default="",
        help="A DataType which denotes the speech feature used as EMG generator input. Leave blank to use the speech feature in the training configuration."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=-1,
        help="The number of EMG samples used for training. (a value smaller than 0.0 means this argument is ignored)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help="The batch size used for training. (a value smaller than 0.0 means this argument is ignored)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="The maximum number of training steps. (a value smaller than 0.0 means this argument is ignored)"
    )
    return parser


def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    prefix = '_orig_mod.' 
    for sd_key in state_dict.keys():
        if prefix in sd_key:
            new_state_dict[sd_key.replace(prefix, "")] = state_dict[sd_key]
        else:
            new_state_dict[sd_key] = state_dict[sd_key]
    return new_state_dict


def load_config(args: argparse.Namespace, override_with_ste_gan_eval_args: bool = True) -> DictConfig:
    """Loads the config for training a STE-GAN model.

    Args:
        args (argparse.Namespace): The arpgarse arguments. The "config" field should be a path to the config file.
        override_with_ste_gan_eval_args (bool, optional): 
        Whether the configuration should be overwritten with additional command-line arguments (if they are set). Defaults to True.

    Returns:
        DictConfig: _description_
    """

    with open(args.config) as fp:
        print(f"Loading {args.config}")
        base_cfg = yaml.load(fp, Loader=yaml.FullLoader)
    
    with open(args.data) as fp:
        print(f"Loading {args.data}")
        data_cfg = yaml.load(fp, Loader=yaml.FullLoader) 
    
    if args.emg_enc_cfg:
        with open(args.emg_enc_cfg) as fp:
            emg_enc_cfg = yaml.load(fp, Loader=yaml.FullLoader) 
            base_cfg["emg_encoder"] = emg_enc_cfg
    
    base_cfg["data"] = data_cfg
    
    if override_with_ste_gan_eval_args:
        base_cfg = override_config_with_eval_args(base_cfg, args)
    
    cfg = DictConfig(base_cfg)
    return cfg
