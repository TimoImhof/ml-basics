from src.configs import TransformerConfig
import torch
from src.modules import Encoder, Transformer

X = torch.randint(low=0, high=32768, size=[1, 32])


model = Transformer(config=TransformerConfig())
print(model)
y = model(X, X)
print(y.shape)
