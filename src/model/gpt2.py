import torch
from torch import nn

from src.config import DEVICE
from src.model.decoder import Decoder
from src.model.embedding import Embedding
from src.model.pos_encoder import PositionalEncoder

#TODO: implement MoE and MoD

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


    def forward(self, x, i):
        """
        does a forward pass on the network
        :param x: input tensor
        :param i: index of the token to predict
        :return:
        """
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for decoder in self.decoders:
            x = decoder(x)

        # get only "next" token
        x = x.select(dim=1, index=i)

        # get each token probability
        x = torch.matmul(x, self.embedding.token_embeddings.T)

        return x

if __name__ == "__main__":

    model = GPT2(
        vocabulary_size=50257,
        embedding_size=768,
        context_length=1024,
        positional_encoding_scalar=10000,
        positional_encoding_coefficient=10000,
        batch_size=1,
        num_heads=12,
        num_decoders=12
    ).to(DEVICE)

    print(f"{sum(p.numel() for p in model.parameters()):,}")

