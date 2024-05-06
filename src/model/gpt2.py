import torch
from torch import nn

from src.config import DEVICE
from src.model.decoder import Decoder
from src.model.embedding import Embedding
from src.model.pos_encoder import PositionalEncoder


class GPT2(nn.Module):
    def __init__(
            self,
            vocabulary_size: int,
            embedding_size: int,
            context_length: int,
            positional_encoding_scalar: int,
            positional_encoding_coefficient: int,
            batch_size: int,
            num_heads: int,
            num_decoders: int,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.context_length = context_length
        self.batch_size = batch_size
        self.num_heads = num_heads

        self.embedding = Embedding(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size
        ).to(DEVICE)

        self.pos_encoder = PositionalEncoder(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            context_length=context_length,
            positional_encoding_scalar=positional_encoding_scalar,
            positional_encoding_coefficient=positional_encoding_coefficient,
            batch_size=batch_size,
        ).to(DEVICE)

        self.decoders = nn.ModuleList([
            Decoder(
                embedding_size=embedding_size,
                context_length=context_length,
                batch_size=batch_size,
                num_heads=num_heads
            ).to(DEVICE) for _ in range(num_decoders)
        ])


    def forward(self, x, *args):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for decoder in self.decoders:
            x = decoder(x)

        # get only "next" token
        x = x.select(dim=1, index=-1)

        # get each token probability
        x = torch.matmul(x, self.embedding.token_embeddings.T)

        return x
