import seaborn as sns
import torch
from matplotlib import pyplot as plt

from src.config import CONTEXT_LENGTH, POSITIONAL_ENCODING_SCALAR, EMBEDDING_SIZE


if __name__ == '__main__':
    v1 = torch.arange(CONTEXT_LENGTH)
    v2 = torch.pow(POSITIONAL_ENCODING_SCALAR, - 2 * torch.arange(EMBEDDING_SIZE // 2) / EMBEDDING_SIZE)

    prod = torch.outer(v1, v2)

    sin = torch.sin(prod)
    cos = torch.cos(prod)

    data = torch.stack([sin, cos], dim=2).view(CONTEXT_LENGTH, EMBEDDING_SIZE)
    # data = torch.concat([sin, cos], dim=1)

    sns.heatmap(data, cmap="viridis")
    plt.show()
