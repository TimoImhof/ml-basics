import torch.nn as nn
import torch
from .configs import (
    FeedForwardConfig,
    AttConfig,
    MultiAttConfig,
    EncoderLayerConfig,
    EncoderConfig,
    DecoderLayerConfig,
    DecoderConfig,
    TransformerConfig,
)


class FeedForward(nn.Module):

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

    def __init__(self, config: AttConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.K = nn.Linear(config.in_features, config.hidden_dim)
        self.Q = nn.Linear(config.in_features, config.hidden_dim)
        self.V = nn.Linear(config.in_features, config.hidden_dim)
        self.d_k = torch.tensor(config.hidden_dim)  # TODO: check if this makes sense

    def forward(self, X: torch.tensor, **kwargs) -> torch.tensor:
        B, T, C = X.shape  # Batch size, sample size, token repr size
        if "enc_hidden_states" in kwargs:
            enc_hidden_states = kwargs["enc_hidden_states"]
            K, Q = self.K(enc_hidden_states), self.Q(enc_hidden_states)
        else:
            K, Q = self.K(X), self.Q(X)
        V = self.V(X)
        W = Q @ K.transpose(-2, -1)  # (B,T,C) @ (B,C,T) -> (B,T,T)
        W = torch.div(W, torch.sqrt(self.d_k))
        W = nn.functional.softmax(W, dim=-1)
        value_repr = W @ V  #  (B,T,T) @ (B,T,C) -> (B,T,C)
        return value_repr


class MultiHeadAttention(nn.Module):

    def __init__(self, config: MultiAttConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.heads = [
            AttentionHead(config=config.head_config) for _ in range(config.n_heads)
        ]
        self.proj = nn.Linear(config.proj_in_features, config.proj_out_features)

    def forward(self, X: torch.tensor, **kwargs) -> torch.tensor:
        result = torch.zeros_like(X)
        for i, head in enumerate(self.heads):
            sub_res = head(X, **kwargs)
            result[
                :,
                i
                * self.config.head_config.hidden_dim : (i + 1)
                * self.config.head_config.hidden_dim,
            ] = sub_res
        return self.proj(result)


class EncoderLayer(nn.Module):

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

    def __init__(self, *args, **kwargs):
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


class MaskedMultiHeadAttention(MaskedAttentionHead):

    def __init__(self, config: MultiAttConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.heads = [
            MaskedAttentionHead(config=config.head_config)
            for _ in range(config.n_heads)
        ]


class DecoderLayer(nn.Module):

    def __init__(self, config: DecoderLayerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.multihead_att = MultiHeadAttention(config.multi_att_config)
        self.multihead_mask_att = MaskedMultiHeadAttention(config.multi_att_config)
        self.ff = FeedForward(config.ff_config)
        self.norm = nn.LayerNorm(config.norm_shape)

    def forward(self, X: torch.tensor, **kwargs) -> torch.tensor:
        m_att_out = self.multihead_mask_att(X)
        m_interm = self.norm(m_att_out + X)

        att_out = self.multihead_att(X, **kwargs)
        interm = self.norm(att_out + m_interm)

        ff_out = self.ff(interm)
        final = self.norm(ff_out + interm)
        return final


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.layers = nn.ModuleList(
            DecoderLayer(config.decoder_layer_config) for _ in range(config.num_layers)
        )

    def forward(self, X: torch.tensor, **kwargs) -> torch.tensor:
        for i, layer in enumerate(self.layers):
            X = layer(X, **kwargs)
        return X


class Transformer(nn.Module):

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.encoder = Encoder(config.encoder_config)
        self.decoder = Decoder(config.decoder_config)
        # TODO self.embedding =


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
