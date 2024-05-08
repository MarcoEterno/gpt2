import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    # TODO: IMPLEMENT GQA, MQA.
    def __init__(
            self,
            embedding_size: int,
            context_length: int,
            batch_size: int,
            num_heads: int
    ):
        super().__init__()
        assert embedding_size % num_heads == 0, \
            "EMBEDDING_SIZE must be a multiple of NUM_HEADS (multi-head attention)"

        # parameters
        self.embedding_size = embedding_size
        self.num_tokens = context_length
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_feature = self.embedding_size // self.num_heads
        self.dim_model = embedding_size
        self.qkv_matrix_dim = (num_heads, embedding_size, embedding_size // num_heads)

        # self attention layer
        self.q_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)
        self.k_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)
        self.v_matrix = nn.Parameter(torch.randn(*self.qkv_matrix_dim), requires_grad=True)

        self.feature_reduction_matrix = nn.Parameter(
            torch.randn(batch_size, embedding_size, self.dim_model), requires_grad=True)

        # feed forward layer
        self.feed_forward_layer = nn.Linear(self.dim_model, self.embedding_size)
        self.activation_function = nn.ReLU()

        self.layer_norm_1 = nn.LayerNorm([context_length, embedding_size])
        self.layer_norm_2 = nn.LayerNorm([context_length, embedding_size])

    def forward(self, x):
        z = self.masked_self_attention(x)
        z1 = self.layer_norm_1(x + z)

        z2 = self.feed_forward(z1)
        return self.layer_norm_2(z1 + z2)

    def get_qkv(self, x, matrix):
        """qkv stands for q, k or v

        Following convention:
        - batch_size = 3
        - num_heads = 8
        - num_tokens = 32
        - embedding_size = 512
        - head_feature = embedding_size / num_heads = 64
        """

        qkv = torch.tensordot(x, matrix, dims=([2], [1]))  # 3, 32, 8, 64
        qkv = torch.swapaxes(qkv, 1, 2)  # 3, 8, 32, 64
        qkv = torch.reshape(qkv, (self.batch_size * self.num_heads, self.num_tokens, self.head_feature))  # 24, 32, 64

        return qkv  # 24, 32, 64

    def qkv_product(self, q, k, v):
        """Takes q, k, v and returns z

        This is used both in self_attention and in encoder_decoder_attention.
        """
        kt = torch.transpose(k, 1, 2)  # 24, 64, 32
        z = torch.bmm(q, kt)

        # masked self-attention
        ones = torch.ones_like(z)
        mask = - 1e9 * torch.triu(ones, diagonal=1) + ones
        z = torch.mul(z, mask)

        z = F.softmax(torch.div(z, self.head_feature ** 0.5), dim=2)  # 24, 32, 32
        z = torch.bmm(z, v)  # 24, 32, 64
        z = torch.reshape(z, (self.batch_size, self.num_heads, self.num_tokens, self.head_feature))  # 3, 8, 32, 64
        z = torch.swapaxes(z, 1, 2)  # 3, 32, 8, 64
        z = torch.reshape(z, (self.batch_size, self.num_tokens, self.embedding_size))  # 3, 32, 512

        return z

    def masked_self_attention(self, x):
        q = self.get_qkv(x, self.q_matrix)
        k = self.get_qkv(x, self.k_matrix)
        v = self.get_qkv(x, self.v_matrix)

        z = self.qkv_product(q, k, v)

        return torch.bmm(z, self.feature_reduction_matrix)  # 3, 32, 512

    def feed_forward(self, x):
        z = self.feed_forward_layer(x)
        return self.activation_function(z)

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
