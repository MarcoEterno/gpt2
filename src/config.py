import torch

# Training parameters
BATCH_SIZE = 32
TRAIN_NUM_EPOCHS = 3
SAVE_CHECKPOINT_EVERY_N_MINUTES = 5

# Model parameters
CONTEXT_LENGTH = 16  # 32
EMBEDDING_SIZE = 512  # 128. calculated with optimal parameters scaling. gpt2 has 768
POSITIONAL_ENCODING_SCALAR = 10_000
POSITIONAL_ENCODING_COEFFICIENT = 300
NUM_HEADS = 8  # 8
NUM_DECODERS = 12  # 12
VOCAB_SIZE = 50257

# paths
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR = "logs"


# Device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {DEVICE} device")
