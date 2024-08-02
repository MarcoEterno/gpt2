from src.model.gpt2 import GPT2
from src.model.tokenizer import tokenizer



if __name__ == '__main__':
    # instantiate the tokenizer
    text = "Hello there! General Kenobi"
    tokens = tokenizer(text)
    print(tokens)
    print(tokenizer.decode(tokens['input_ids']))
    assert tokenizer.decode(tokens['input_ids']) == text

    # instantiate the model
    model = GPT2(
        vocabulary_size=50257,
        embedding_size=768,
        context_length=1024,
        positional_encoding_scalar=10000,
        positional_encoding_coefficient=10000,
        batch_size=1,
        num_heads=12,
        num_decoders=12
    )

    # forward pass
    x = tokens['input_ids']
    print(x)
    i = 0
    print(print(model.forward(x, i).shape), model(x, i))