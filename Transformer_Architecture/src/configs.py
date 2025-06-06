from dataclasses import dataclass, field
import torch.nn as nn


@dataclass
class FeedForwardConfig:
    in_features: int = 512
    out_features: int = 2048
    act_fct: nn.Module = field(default_factory=nn.ReLU)


@dataclass
class AttConfig:
    in_features: int = 512
    hidden_dim: int = 64


@dataclass
class MultiAttConfig:
    n_heads: int = 8
    head_config: AttConfig = field(default_factory=AttConfig)
    proj_in_features: int = 512
    proj_out_features: int = 512


@dataclass
class EncoderLayerConfig:
    multi_att_config: MultiAttConfig = field(default_factory=MultiAttConfig)
    ff_config: FeedForwardConfig = field(default_factory=FeedForwardConfig)
    norm_shape: int = 512


@dataclass
class EncoderConfig:
    num_layers: int = 6
    encoder_layer_config: EncoderLayerConfig = field(default_factory=EncoderLayerConfig)


@dataclass
class DecoderLayerConfig(EncoderLayerConfig):
    pass


@dataclass
class DecoderConfig:
    num_layers: int = 6
    decoder_layer_config: DecoderLayerConfig = field(default_factory=DecoderLayerConfig)


@dataclass
class EmbeddingConfig:
    vocab_size: int = 32768
    hidden_dim: int = 512
    padding_idx: int = 0
    pos_n: int = 10000


@dataclass
class TransformerConfig:
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_config: DecoderConfig = field(default_factory=DecoderConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
