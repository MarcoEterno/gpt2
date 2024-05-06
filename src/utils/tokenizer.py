from transformers import GPT2TokenizerFast

from src.config import DEVICE

from src.config import CONTEXT_LENGTH

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

VOCABULARY_SIZE = tokenizer.vocab_size


def tokenize(batch):
    tokens = tokenizer(
        text=[page['text'] for page in batch],
        padding="max_length",
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    }


if __name__ == "__main__":
    print(tokenizer("Hello there! General Kenobi"))
