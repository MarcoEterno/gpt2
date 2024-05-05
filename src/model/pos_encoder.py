import torch
from torch import nn


class PositionalEncoder(nn.Module):
    def __init__(
            self,
            vocabulary_size: int,
            embedding_size: int,
            context_length: int,
            positional_encoding_scalar: int,
            positional_encoding_coefficient: int,
            batch_size: int
    ):
        super().__init__()

        assert embedding_size % 2 == 0, "EMBEDDING_SIZE must be even (positional encoder)"

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_tokens = context_length
        self.positional_encoding_scalar = positional_encoding_scalar
        self.positional_encoding_coefficient = positional_encoding_coefficient
        self.batch_size = batch_size

        self.positional_encoding_vector = nn.Parameter(
            self.compute_positional_encoding_vector(), requires_grad=False)

    def compute_positional_encoding_vector(self):
        v1 = torch.arange(self.num_tokens)
        v2 = torch.pow(self.positional_encoding_scalar,
                       - 2 * torch.arange(self.embedding_size // 2) / self.embedding_size)

        prod = torch.outer(v1, v2)

        sin = torch.sin(prod)
        cos = torch.cos(prod)

        data = torch.stack([sin, cos], dim=2).view(self.num_tokens, self.embedding_size)

        return data.repeat(self.batch_size, 1, 1)

    def forward(self, embedded_token):
        return embedded_token + self.positional_encoding_vector / self.positional_encoding_coefficient

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])

        named_modules = list(named_modules)

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr += f'({name}): tensor({str(tuple(p[1].shape))}, ' \
                               f'requires_grad={str(p[1].requires_grad)})\n'

        return string_repr

