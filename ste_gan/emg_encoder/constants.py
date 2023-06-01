from pathlib import Path

OUTPUT_BASE_DIR = Path("/share/temp/kscheck/models/ste-gan-encoders-test-with-fake-data")

# If we are in debug Mode (e.g., break training loops early)
DEBUG = False

# Sequence length for allocating / deallocating tensors
SEQ_LEN = 200

# Number of utterances in a batch; 
BATCH_SIZE = 16

LEARNING_RATE = 3e-4

EMG_SIGNAL_TO_SPEECH_UNITS = 16

LEARNING_RATE_PATIENCE = 5

LEARNING_RATE_WARMUP = 500

WEIGHT_DECAY = 1e-5

TRAIN_BATCH_MAX_LEN = 128000

NUM_EPOCHS = 160

EARLY_STOP_PATIENCE = 10

LOSS_WEIGHT_SPEECH_UNITS = 0.5

LOSS_WEIGHT_PHONEMES = 0.5

# Norm of the distance of speech unit loss
SU_LOSS_NORM = 2.0