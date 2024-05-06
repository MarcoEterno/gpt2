import torch

BATCH_SIZE = 2
TRAIN_NUM_EPOCHS = 10

CONTEXT_LENGTH = 9  # 32
EMBEDDING_SIZE = 4  # 128. calculated with optimal parameters scaling
POSITIONAL_ENCODING_SCALAR = 10_000
POSITIONAL_ENCODING_COEFFICIENT = 300
NUM_HEADS = 2
NUM_DECODERS = 12
VOCAB_SIZE = 50257

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {DEVICE} device")
