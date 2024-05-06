from gpt2 import GPT2
from src.config import *


class ModelInspector:
    def __init__(self, model: GPT2):
        self.model = model

    def inspect(self):
        print(self.model)
        print(self.model.embedding)
        print(self.model.pos_encoder)
        for decoder in self.model.decoders:
            print(decoder)

    def print_parameters(self):
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            print(name, param.numel())
        print(f"Total number of parameters: {total_params}")

    def print_named_modules(self):
        for name, module in self.model.named_modules():
            print(name)


if __name__ == "__main__":
    from gpt2 import GPT2
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
    model_inspector = ModelInspector(model)
    #model_inspector.inspect()
    model_inspector.print_parameters()
    #model_inspector.print_named_modules()
    print(sum(p.numel() for p in model.parameters()))