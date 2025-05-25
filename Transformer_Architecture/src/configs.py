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
