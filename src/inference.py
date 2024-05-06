import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE
from src.config import CONTEXT_LENGTH
from src.config import DEVICE
from src.config import EMBEDDING_SIZE
from src.config import NUM_DECODERS
from src.config import NUM_HEADS
from src.config import POSITIONAL_ENCODING_COEFFICIENT
from src.config import POSITIONAL_ENCODING_SCALAR
from src.config import VOCAB_SIZE
from src.utils.dataset import train_ds
from src.model.gpt2 import GPT2
from src.utils.tokenizer import tokenize, tokenizer

from train import save_checkpoint, load_checkpoint


if __name__ == "__main__":
    epoch_to_start_from = 0
    dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=tokenize)
    model = GPT2(
        vocabulary_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        context_length=CONTEXT_LENGTH,
        positional_encoding_scalar=POSITIONAL_ENCODING_SCALAR,
        positional_encoding_coefficient=POSITIONAL_ENCODING_COEFFICIENT,
        batch_size=BATCH_SIZE,
        num_heads=NUM_HEADS,
        num_decoders=NUM_DECODERS
    ).to(DEVICE)
    if epoch_to_start_from > 0:
        load_checkpoint(model, epoch=epoch_to_start_from)
    print(model)

    for batch in dataloader:
        print(batch)
        for i, sentence in enumerate(batch['input_ids'].to("cpu")):
            print(f"Sentence {i}:", tokenizer.decode(sentence))

        logits = model(batch, 0)
        probs = F.softmax(logits, dim=1)
        print(probs)
        token = torch.argmax(probs, dim=1)  # generalize to top_k

        for i, single_token in enumerate(token.to("cpu")):
            print(f"Token {i}:", tokenizer.decode(single_token))

        break
