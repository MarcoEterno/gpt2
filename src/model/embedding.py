import torch
from torch import nn
from torch.nn import functional as F


class Embedding(nn.Module):
    def __init__(self, vocabulary_size, embedding_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.token_embeddings = nn.Parameter(torch.randn(vocabulary_size, embedding_size), requires_grad=True)
        # self.layer = nn.Linear(vocabulary_size, embedding_size)

    def forward(self, batch):
        """
        Forward pass of the model

        Args:
            batch (dict): dictionary with the input_ids of the batch

        Returns:
            torch.Tensor: the embeddings of the input_ids
        """
        x = batch['input_ids']
        one_hot_encoded_x = F.one_hot(x, num_classes=self.vocabulary_size).float()
        return torch.matmul(one_hot_encoded_x, self.token_embeddings)
        # return self.layer(one_hot_encoded_x)
