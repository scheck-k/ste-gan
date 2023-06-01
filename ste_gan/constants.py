"""Constants which are used for the STE-GAN project.
Adapted from: https://github.com/descriptinc/cargan/blob/master/cargan/constants.py
"""
from pathlib import Path
import torch
import functools
###############################################################################
# Training parameters
###############################################################################

EMG_SAMPLE_RATE = 800

# Batch size (per gpu)
BATCH_SIZE = 32

# Training chunk size in EMG signal samples
CHUNK_SIZE = 2048

# Maximum length of a training example
MAX_LENGTH = 10 * EMG_SAMPLE_RATE

# Maximum number of training steps
MAX_STEPS = 50_000 

# Dimensionality of soft speech units
SPEECH_UNITS_FEAT_SIZE = 256

# Number of MFCCs used in comparison experiments 
NUM_MFCCS = 25

# Size of embeddings for session ID / speaking mode
EMBEDDING_DIM_SIZE = 64

# Number of channels of the EMG data set of Gaddy and Klei
NUM_EMG_CHANNELS = 8

# Number of sessions of the used EMG data set of Gaddy and Klei
NUM_EMG_SESSIONS = 17

# Hop Size of Soft Speech Units
# Soft speech units have frequeny of 50Hz -> 20ms frame shift / "hop"
SPEECH_UNIT_HOPSIZE_SECONDS = 0.02

# Hopsize of soft speech units
HOPSIZE = int(EMG_SAMPLE_RATE * SPEECH_UNIT_HOPSIZE_SECONDS)

# Number of input channels
NUM_FEATURES = SPEECH_UNITS_FEAT_SIZE

# Number of input channels to the discriminator
NUM_DISCRIM_FEATURES = NUM_EMG_CHANNELS

# Number of data loading worker threads
NUM_WORKERS = 2

# The optimizer to use for training
OPTIMIZER = functools.partial(torch.optim.AdamW, lr=2e-4, betas=(.8, .99))

# Seed for all random number generators
RANDOM_SEED = 0

# Number of samples in a training example
TRAIN_EMG_LENGTH = CHUNK_SIZE 

# Number of frames in a training example
TRAIN_FEATURE_LENGTH = TRAIN_EMG_LENGTH // HOPSIZE

###############################################################################
# Training parameters (loss)
###############################################################################

# Loss function to use for adversarial loss
# Options: 'hinge', 'mse', None
LOSS_ADVERSARIAL = 'mse'

# Whether to use feature matching loss
LOSS_FEAT_MATCH = True

# Feature matching loss weight
LOSS_FEAT_MATCH_WEIGHT = 7.

# Whether to use mel error loss
LOSS_MEL_ERROR = True

# Whether to use TD
LOSS_MULTI_TD_ERROR = True

# Weight of the Multi-TD Loss
LOSS_MULTI_TD_ERROR_WEIGHT = 15.0

# Loss of the speech units
LOSS_SPEECH_UNIT_ERROR = True

# Speech Unit loss weight
LOSS_SPEECH_UNIT_WEIGHT = 1.0 

# Loss of phoneme accuracy
LOSS_PHONEMES_ERROR = True

# Loss weight of the phoneme accuracy 
LOSS_PHONEMES_WEIGHT = 1.0

# Whether to use L2 waveform loss
LOSS_WAVEFORM_ERROR = False

# Waveform error loss weight
LOSS_WAVEFORM_ERROR_WEIGHT = 1

###############################################################################
# Training parameters (logging)
###############################################################################

# Number of steps between logging
INTERVAL_LOG = 50

# Number of steps between generating samples
INTERVAL_SAMPLE = 1_000

# Interval between computing quick evaluation metrics on the data sets.
INTERVAL_EMG_SIGNAL_METRICS = 5_000

# Interval between computing longer evaluation metrics on the data sets (e.g. ASR data set).
INTERVAL_EMG_SYNTH_METRICS = 10_000

# Number of steps between saving
INTERVAL_SAVE = 25_000

# Number of steps between validation
INTERVAL_VALID = 500

# Number of steps between logging waveform MSE
INTERVAL_WAVEFORM = 500

# Interval between plotting
INTERVAL_PLOT = 1000

# Number of samples to put on tensorboard
NUM_TEST_SAMPLES = 10

### EVALUATION
MIN_NUM_STEPS_ASR_EVAL = 25_000

ASR_EVAL_MORE_TEMPERATURE_MIN_STEPS = 25_000


# Used to generate phoneme indices from TextGrid files 
# ARPABet phonemes, as in: https://github.com/dgaddy/silent_speech/blob/main/read_emg.py
PHONEME_INVENTORY = ['aa','ae','ah','ao','aw','ax','axr','ay','b','ch','d','dh','dx','eh','el','em','en','er','ey','f','g','hh','hv','ih','iy','jh','k','l','m','n','nx','ng','ow','oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh','sil']

SILENCE_PHONEME_INDEX = PHONEME_INVENTORY.index("sil")

# Names of TD Features.
EMG_TD_FEAT_NAMES = [
    "Mean Lowp",
    "Power Lowp",
    "Power High",
    "ZCR High.",
    "Mean High",
    "Hilbert Env"
]

# Global boolean whether DEBUG mode is on. 
# Can also be set in argparse for some training scripst
DEBUG = False

PHONEME_INVENTORY =  [
    'aa','ae','ah','ao','aw','ax','axr','ay','b','ch','d','dh','dx',
    'eh','el','em','en','er','ey','f','g','hh','hv','ih','iy','jh',
    'k','l','m','n','nx','ng','ow','oy','p','r','s','sh','t','th',
    'uh','uw','v','w','y','z','zh','sil'
]

NUM_PHONEMS = len(PHONEME_INVENTORY)

class DataDir:
    """Constants for the directories in which the dat samples will be saved."""
    EMG = "emg"
    TRANSCRIPTIONS = "transcriptions"
    PHONEMES = "phonemes"
    SPEECH_UNITS = "units"
    MFCCS = "mfccs"
    EMG_FEATS = "emg_feats"
    ACOUSTIC_FEATS = "acoustic_feats"
    AUDIO = "audio"
    

class SpeakingMode:
    SILENT = "silent"
    NORMAL = "normal"


class DataType:
    """
    A class which holds the constants for data types for one utterance.
    """
    
    # The ID of the utterance.
    UTT_ID = "UTT_ID"
    
    # The real EMG signal
    REAL_EMG = "REAL_EMG"

    # EMG Features (e.g. TD features + Hilbert)
    EMG_FEATURES = "EMG_FEATS"
    
    # Mel-frequency cepstral coefficients.
    MFCCS = "MFCCS"

    # The ground-truth transcription of the utterance.
    TRANSCRIPTION = "TRANSCRIPTION"

    # A sequence of phoneme Ids with the same length as the speech untis
    PHONEMES = "PHONEMES"

    # The Identifier of the speech units. Normally, soft speech units are used.
    SPEECH_UNITS = "SPEECH_UNITS"
    
    # The index of the session
    SESSION_INDEX = "SESSION_INDEX"

    # A string identifier of a session
    SESSION_ID = "SESSION_ID"

    # An ID for the type of speaking
    SPEAKING_MODE_ID = "SPEAKING_MODE"
    

    SPEAKING_MODE_INDEX = "SPEAKING_MODE_IDX"

    # The generated EMG signal
    FAKE_EMG = "FAKE_EMG"
    # The predicted phonemes given the fake EMG signal
    PRED_PHONEMES = "PRED_PHONEMES"
    # The predicted speech unit sequence given the fake EMG signal
    PRED_SPEECH_UNITS = "PRED_SPEECH_UNITS"
    # The predicted transcript given the speech synthesis of the fake EMG signal
    PRED_TRANSCRIPT = "PRED_TRANSCRIPT"
    # Synthesized audio from the predict soft-speech units & a pre-trained acoustic model & vocoder
    PRED_AUDIO_SYNTH = "PRED_SYNTH_AUDIO"
    