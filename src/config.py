import torch

BATCH_SIZE = 32
TRAIN_NUM_EPOCHS = 3
SAVE_CHECKPOINT_EVERY_N_MINUTES = 1

CONTEXT_LENGTH = 32  # 32
EMBEDDING_SIZE = 32  # 128. calculated with optimal parameters scaling
POSITIONAL_ENCODING_SCALAR = 10_000
POSITIONAL_ENCODING_COEFFICIENT = 300
NUM_HEADS = 2  # 8
NUM_DECODERS = 6  # 12
VOCAB_SIZE = 50257

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {DEVICE} device")
