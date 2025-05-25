from src.configs import EncoderConfig
import torch
from src.modules import Encoder

X = torch.rand(size=[32, 512])

config = EncoderConfig()
encoder = Encoder(config)

out = encoder(X)
print(out.shape)
