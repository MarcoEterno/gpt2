import torch
from torch.nn import functional as F

from src.config import BATCH_SIZE
from src.config import CONTEXT_LENGTH
from src.config import DEVICE
from src.config import EMBEDDING_SIZE
from src.config import NUM_DECODERS
from src.config import NUM_HEADS
from src.config import POSITIONAL_ENCODING_COEFFICIENT
from src.config import POSITIONAL_ENCODING_SCALAR
from src.config import VOCAB_SIZE
from src.dataloader import test_dataloader
from src.model.gpt2 import GPT2
from src.model.tokenizer import tokenizer
from src.checkpoint_management import load_model, load_model_old
from src.model.model_inspector import ModelInspector

if __name__ == "__main__":
    start_epoch = 0
    start_batch_number = 3
    model_parameters = f"_{BATCH_SIZE}_{CONTEXT_LENGTH}_{VOCAB_SIZE}_{EMBEDDING_SIZE}_{NUM_DECODERS}_{NUM_HEADS}"

    #logging.basicConfig(level=logging.INFO)
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

    if start_epoch > 0 or start_batch_number > 0:
        model, optimizer, start_epoch, start_batch_number = load_model_old(model, start_epoch, start_batch_number)

    model_inspector = ModelInspector(model)
    model_inspector.print_parameters()

    batch = next(iter(test_dataloader))
    print(batch['input_ids'].shape)

    number_of_sentences_to_infer = 1
    #print the inference of the model for the first batch
    for batch_number in range(0, number_of_sentences_to_infer):
        for i, sentence in enumerate(batch['input_ids'].to("cpu")):
            logits = model(batch, i)
            probs = F.softmax(logits, dim=1)
            print(f"Sentence {i}:", tokenizer.decode(sentence[0:i]))
            print(f"Token {i}:", tokenizer.decode(torch.argmax(probs, dim=1)[batch_number]))
            print(f"Probability {i}:", probs[batch_number].max().item())













    """
    for batch in test_dataloader:
        print(batch)
        for i, sentence in enumerate(batch['input_ids'].to("cpu")):
            print(f"Sentence {i}:", tokenizer.decode(sentence))

        logits = model(batch, 0)
        probs = F.softmax(logits, dim=1)
        print(probs)
        token = torch.argmax(probs, dim=1)  # generalize to top_k
        for i, single_token in enumerate(token.to("cpu")):
            print(f"Token {i}:", tokenizer.decode(single_token))

        break"""