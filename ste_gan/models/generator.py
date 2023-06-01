"""Implements the EMGGenerator architecture of the paper.
The model is based on the generator of the CarGAN repository: https://github.com/descriptinc/cargan
"""

from typing import Dict

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

import ste_gan
from ste_gan.constants import DataType
from ste_gan.layers.conv import GBlock, WNConv1d


class EMGGenerator(nn.Module):
    """Base class for the EMG Generator

    """
    
    def __init__(
        self,
        speech_feature_type: str,
        speech_input_dim: int,
        num_sessions: int,
        num_output_channels: int,
    ) -> None:
        """Constructor of the EMG generator

        Args:
            speech_feature_type (str): A constant depicting the type of speech feature (see the DataType class in constants.py)
            speech_input_dim (int): The dimensionality of the speech feature inputs.
            num_sessions (int): The number of EMG session.
            num_output_channels (int): The number of EMG channels.
        """
        super().__init__()
        self.speech_feature_type = speech_feature_type
        self.speech_input_dim = speech_input_dim
        self.num_output_channels = num_output_channels
        self.num_sessions = num_sessions
        
    def forward(self, speech_unit_sequence: Tensor, session_ids: Tensor, speaking_mode_ids: Tensor, ar=None, ):
        raise NotImplementedError("Must implement the EMGGenerator interface.")
    
    @torch.inference_mode()
    def generate(self, speech_unit_sequence: Tensor, session_ids: Tensor, speaking_mode_ids: Tensor) -> Tensor:
        return self(speech_unit_sequence, session_ids, speaking_mode_ids)
        
    @torch.inference_mode()
    def generate_from_data_dict(self, data_dict: Dict, device: torch.device) -> Tensor:
        """An auxilary method which generates an EMG signal given a data dictionary from the EMGDataset class.

        Args:
            data_dict (Dict): The data dictionary mapping DataTypes to tensors.
            device (torch.device): The device on which we perform the inference

        Returns:
            Tensor: The generated EMG signal without a batch dimension.
        """
        s_t = data_dict[self.speech_feature_type].to(device)
        sess_idx = data_dict[DataType.SESSION_INDEX].to(device)
        spk_mode_idx = data_dict[DataType.SPEAKING_MODE_INDEX].to(device) 

        # Not a data Loader i.e. no batch dim --> unsqueeze inputs
        if len(s_t.shape) == 2:
            s_t = s_t.unsqueeze(0)
            sess_idx = sess_idx.unsqueeze(0)
            spk_mode_idx = spk_mode_idx.unsqueeze(0)
        
        # Generate the fake EMG signal
        pred_emg = self.generate(s_t, sess_idx, spk_mode_idx).squeeze(0).detach().cpu()
        return pred_emg


class EMGGeneratorGanTTS(EMGGenerator):
    """The EMGGenerator as used in the paper.
    It is based on the GanTTS architecture, which is also used in CarGAN

    """
    def __init__(
        self,
        speech_feature_type: str,
        speech_input_dim: int,
        num_sessions: int,
        num_emg_channels: int,
        use_speaking_mode_embedding: bool = False,
        use_session_embeddings: bool = True,
        num_speaking_modes: int = 3,
        embedding_dim: int = 64,
        channels: int = 768,
    ):
        super().__init__(speech_feature_type=speech_feature_type, speech_input_dim=speech_input_dim,
                         num_sessions=num_sessions, num_output_channels=num_emg_channels)

        if use_session_embeddings:
            self.session_embeddings = nn.Embedding(num_sessions, embedding_dim)
        else:
            self.session_embeddings = None
        self.use_session_embeddings = use_session_embeddings
        
        if use_speaking_mode_embedding:
            self.speaking_mode_embeddings = nn.Embedding(num_speaking_modes, embedding_dim)
        else:
            self.speaking_mode_embeddings = None
        self.use_speaking_mode_embedding = use_speaking_mode_embedding
        
        self.input_size = self.speech_input_dim + (
            (use_session_embeddings * embedding_dim)
            + (use_speaking_mode_embedding * embedding_dim)
        )
        
        # If we have speech units, we upsample at the almost last layer
        upsample_last = 2 if self.speech_feature_type == DataType.SPEECH_UNITS else 1

        self.gblocks = nn.Sequential(
            WNConv1d(self.input_size, channels, kernel_size=1),
            # Processing Gblocks
            GBlock(channels, channels),
            GBlock(channels, channels),
            # Upsample Gblocks
            GBlock(channels, channels // 2, upsample=2), # 50 Hz -> 100 Hz
            GBlock(channels // 2, channels // 2, upsample=2), # 100 Hz -> 200 Hz
            GBlock(channels // 2, channels // 2, upsample=2), # 200 Hz -> 400 Hz
            GBlock(channels // 2, channels // 4, upsample=upsample_last), # 400 Hz -> 800 Hz
            # Processing Gblocks
            GBlock(channels // 4, channels // 4),
            GBlock(channels // 4, channels // 4),
        )

        self.last_conv = nn.Sequential(
            nn.ReLU(),
            WNConv1d(channels // 4, num_emg_channels,
                     kernel_size=3, padding=1)
        )

    
    def forward(self, speech_unit_sequence: Tensor, session_ids: Tensor, speaking_mode_ids: Tensor):
        x = speech_unit_sequence
        num_su_frames = speech_unit_sequence.shape[1]
        if self.use_session_embeddings:
            session_embs = self.session_embeddings(session_ids)
            session_embs = repeat(session_embs, 'b d -> b t d', t=num_su_frames)
            x = torch.cat((x, session_embs), dim=-1)

        if self.use_speaking_mode_embedding:
            speaking_mode_embs = self.speaking_mode_embeddings(speaking_mode_ids)
            speaking_mode_embs = repeat(speaking_mode_embs, 'b d -> b t d', t=num_su_frames)
            x = torch.cat((x, speaking_mode_embs), dim=-1)
        
        # Conv Processing - Channel first.
        x = rearrange(x, 'b t c -> b c t') 
        x = self.gblocks(x)
        x = self.last_conv(x)
        x = rearrange(x, 'b c t -> b t c')
        
        # Output activation; Scale signal between -1.0 to 1.0
        x = torch.tanh(x)

        return x


def init_emg_generator(cfg: omegaconf.OmegaConf, emg_generator_type: str = "") -> EMGGenerator:
    """A factory method to initialize the EMGGenerator class.

    Args:
        cfg (omegaconf.OmegaConf): A config containing the parameters of the model.
        emg_generator_type (str, optional): An optional string of the class of the EMGGenerator. Defaults to "".

    Returns:
        EMGGenerator: The initialized EMGGenerator model.
    """
    speech_feature_type = cfg.model.speech_feature_type
    if speech_feature_type == DataType.SPEECH_UNITS:
        speech_input_dim: int = ste_gan.SPEECH_UNITS_FEAT_SIZE
    elif speech_feature_type == DataType.MFCCS:
        speech_input_dim: int = ste_gan.NUM_MFCCS
    else:
        raise ValueError(f"Unrecognized speech feature type: {speech_feature_type}")
    
    num_emg_channels = cfg.data.num_emg_channels
    num_sessions = cfg.data.num_emg_sessions
    if not emg_generator_type:
        emg_generator_type = cfg.model.type

    params = dict(num_emg_channels=num_emg_channels, num_sessions=num_sessions,
                  speech_feature_type=speech_feature_type,speech_input_dim=speech_input_dim )
    extra_params = cfg.model.params if "params" in cfg.model else {}
        
    if emg_generator_type == "EMGGeneratorGanTTS":
        return EMGGeneratorGanTTS(**params, **extra_params)
    else:
        raise ValueError(f"Unrecognized EMG generator type: {emg_generator_type}")
    