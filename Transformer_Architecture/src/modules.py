import torch.nn as nn
import torch
from .configs import (
    FeedForwardConfig,
    AttConfig,
    MultiAttConfig,
    EncoderLayerConfig,
    EncoderConfig,
)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network of the Paper "Attention Is All You Need".
    Consists of two linear transformations with a ReLU activation in between.
    """

    def __init__(self, config: FeedForwardConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.ff1 = nn.Linear(
            in_features=config.in_features, out_features=config.out_features
        )
        self.ff2 = nn.Linear(config.out_features, config.in_features)
        self.act_fct = config.act_fct

    def forward(self, X: torch.tensor) -> torch.tensor:
        return self.ff2(self.act_fct(self.ff1(X)))


class AttentionHead(nn.Module):
    """
    Scaled Dot-Procut Attention Head of the Paper "Attention is All You Need".
    Consists of learned linear representations key, query, value, which are matrix-multiplied in a special order.
    This enables the module to predict based on a LEARNED representation of the ENTIRE previous context.
    """

    def __init__(self, config: AttConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.K = nn.Linear(config.in_features, config.hidden_dim)
        self.Q = nn.Linear(config.in_features, config.hidden_dim)
        self.V = nn.Linear(config.in_features, config.hidden_dim)
        self.d_k = torch.tensor(config.hidden_dim)  # TODO: check if this makes sense

    def forward(self, X: torch.tensor) -> torch.tensor:
        B, T, C = X.shape  # Batch size, sample size, token repr size

        K, Q, V = self.K(X), self.Q(X), self.V(X)
        W = Q @ K.transpose(-2, -1)  # (B,T,C) @ (B,C,T) -> (B,T,T)
        W = torch.div(W, torch.sqrt(self.d_k))
        W = nn.functional.softmax(W, dim=-1)
        value_repr = W @ V  #  (B,T,T) @ (B,T,C) -> (B,T,C)
        return value_repr


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention of the Paper "Attention is All You Need".
    Consists of multiple sub-dimensional attention heads whose output is concatenated and feed through a final linear layer.
    """

    def __init__(self, config: MultiAttConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.heads = [
            AttentionHead(config=config.head_config) for _ in range(config.n_heads)
        ]
        self.proj = nn.Linear(config.proj_in_features, config.proj_out_features)

    def forward(self, X: torch.tensor) -> torch.tensor:
        result = torch.zeros_like(X)
        for i, head in enumerate(self.heads):
            sub_res = head(X)
            result[
                :,
                i
                * self.config.head_config.hidden_dim : (i + 1)
                * self.config.head_config.hidden_dim,
            ] = sub_res
        return self.proj(result)


class EncoderLayer(nn.Module):
    """
    Encoder layer of a an transformer style Encdoer of the Paper "Attention is All You Need".
    A layer consists of multi-head self-attention mechanism and a positionwise fully connected feed-forward network.
    """

    def __init__(self, config: EncoderLayerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.multihead_att = MultiHeadAttention(config.multi_att_config)
        self.ff = FeedForward(config.ff_config)
        self.norm = nn.LayerNorm(config.norm_shape)

    def forward(self, X: torch.tensor) -> torch.tensor:
        residual = X
        att_output = self.multihead_att(X)
        interm = self.norm(att_output + residual)

        ff_output = self.ff(interm)
        res = self.norm(ff_output + interm)
        return res


class Encoder(nn.Module):
    """
    Encoder of a transformer of the Paper "Attention is All You Need".
    Consists of a series of layers each holding a multi-head Attention module and a feed-forward network.
    """

    def __init__(self, config: EncoderConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.layers = nn.ModuleList(
            EncoderLayer(config.encoder_layer_config) for _ in range(config.num_layers)
        )

    def forward(self, X: torch.tensor) -> torch.tensor:
        for i, layer in enumerate(self.layers):
            X = layer(X)
        return X


class MaskedAttentionHead(AttentionHead):
    """
    Masked Attention Head of a transformer of the Paper "Attention is All You Need".
    """

    def __init__(self, config: AttConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, X: torch.tensor) -> torch.tensor:
        B, T, C = X.shape  # Batch size, sample size, token repr size

        K, Q, V = self.K(X), self.Q(X), self.V(X)
        W = Q @ K.transpose(-2, -1)  # (B,T,C) @ (B,C,T) -> (B,T,T)
        W = torch.div(W, torch.sqrt(self.d_k))

        # Masked attention
        W = W.masked_fill(torch.tril(torch.ones(T, T) == 0, float("-inf")))  # (B,T,T)
        W = nn.functional.softmax(W, dim=-1)
        value_repr = W @ V  # (B,T,T) @ (B,T,C) -> (B,T,C)
        return value_repr


# class Embedding(nn.Module):
#     """
#     Learned Embedding Layer from the Paper "Attention is All You Need".
#     Converts input tokens into vectors (numerical representation) for the model as input.
#     """

#     def __init__(self, vocab_size: int, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.emb = nn.Embedding(
#             num_embeddings=vocab_size, embedding_dim=512, padding_idx=0
#         )

#     def forward(self, X: torch.tensor) -> torch.tensor:
#         return self.emb(X)
