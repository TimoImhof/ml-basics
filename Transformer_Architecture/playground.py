from src.configs import TransformerConfig
import torch
from src.modules import Encoder, Transformer

X = torch.rand(size=[32, 512])



model = Transformer(config=TransformerConfig())
print(model)
y = model.forward(X)
print(y.shape)