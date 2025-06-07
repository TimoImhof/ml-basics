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
    EmbeddingConfig,
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
        self.scale = config.hidden_dim**-0.5

    def forward(
        self, K_in: torch.tensor, V_in: torch.tensor, Q_in: torch.tensor, **kwargs
    ) -> torch.tensor:
        K, Q, V = self.K(K_in), self.Q(Q_in), self.V(V_in)
        W = Q @ K.transpose(-2, -1) * self.scale  # (B,T,C) @ (B,C,T) -> (B,T,T)
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

    def forward(
        self, K_in: torch.tensor, V_in: torch.tensor, Q_in: torch.tensor, **kwargs
    ) -> torch.tensor:
        result = torch.zeros_like(K_in)
        for i, head in enumerate(self.heads):
            sub_res = head(K_in, V_in, Q_in, **kwargs)
            result[
                :,
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
        att_output = self.multihead_att(X, X, X)
        interm = self.norm(att_output + X)
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
        for _, layer in enumerate(self.layers):
            X = layer(X)
        return X


class MaskedAttentionHead(AttentionHead):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, K_in: torch.tensor, V_in: torch.tensor, Q_in: torch.tensor, **kwargs
    ) -> torch.tensor:
        _, T, _ = K_in.shape

        K, Q, V = self.K(K_in), self.Q(Q_in), self.V(V_in)
        W = Q @ K.transpose(-2, -1) * self.scale  # (B,T,C) @ (B,C,T) -> (B,T,T)

        # Masked attention
        W = W.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))  # (B,T,T)
        W = nn.functional.softmax(W, dim=-1)
        value_repr = W @ V  # (B,T,T) @ (B,T,C) -> (B,T,C)
        return value_repr


class MaskedMultiHeadAttention(MultiHeadAttention):

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

    def forward(
        self, X: torch.tensor, enc_hidden_state: torch.tensor, **kwargs
    ) -> torch.tensor:
        m_att_out = self.multihead_mask_att(X, X, X)
        m_interm = self.norm(m_att_out + X)

        att_out = self.multihead_att(
            K_in=enc_hidden_state, V_in=enc_hidden_state, Q_in=m_interm, **kwargs
        )
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

    def forward(
        self, X: torch.tensor, end_hidden_state: torch.tensor, **kwargs
    ) -> torch.tensor:
        for _, layer in enumerate(self.layers):
            X = layer(X, end_hidden_state, **kwargs)
        return X


class Embedding(nn.Module):

    def __init__(self, config: EmbeddingConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim,
            padding_idx=config.padding_idx,
        )
        self.pos_n = config.pos_n
        self.hidden_dim = config.hidden_dim
        self.scale = config.hidden_dim**0.5

    def forward(self, X: torch.tensor, **kwargs) -> torch.tensor:
        B, T = X.shape  # Batch size, sample size

        word_emb = self.word_embedding(X) * self.scale
        pos_emb = torch.zeros(T, self.hidden_dim)

        token_positions = torch.arange(T, dtype=torch.float).unsqueeze(1)
        token_enc_indices = torch.arange(0, self.hidden_dim, 2, dtype=torch.float)

        pos_emb[:, 0::2] = torch.sin(
            token_positions / (self.pos_n ** (token_enc_indices / self.hidden_dim))
        )
        pos_emb[:, 1::2] = torch.cos(
            token_positions / (self.pos_n ** (token_enc_indices / self.hidden_dim))
        )

        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B,T,C)
        return word_emb + pos_emb


class Transformer(nn.Module):

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.encoder = Encoder(config.encoder_config)
        self.decoder = Decoder(config.decoder_config)
        self.embedding = Embedding(config.embedding_config)

    def forward(self, X_enc: torch.tensor, X_dec: torch.tensor) -> torch.tensor:
        emb_X_enc = self.embedding(X_enc)
        emb_X_dec = self.embedding(X_dec)
        enc_X = self.encoder(emb_X_enc)
        dec_X = self.decoder(emb_X_dec, enc_X)
        return dec_X
