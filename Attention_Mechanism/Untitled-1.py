# %%
# Hyperparameters
n_embd = 32
batch_size = 16  # from part 1
block_size = 32

# %%
# Let's modify our simple bigram model to a simple attention model

import torch.nn as nn
import torch
import torch.nn.functional as F
    
class Head(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        
        self.key_repr = nn.Linear(n_embd, block_size, bias=False)
        self.query_repr = nn.Linear(n_embd, block_size, bias=False)
        self.value_repr = nn.Linear(n_embd, block_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        
        # create learned representations
        k = self.key_repr(x)
        q = self.query_repr(x)
        v = self.value_repr(x)
        
        # compute attention scores ('affinities between tokens')
        W = q @ k.transpose(-2,-1)  # (B,T,C) @ (B,C,T) -> (B,T,T)
        W *= C ** -0.5  # scaling of the dot product to keep softmax from saturating too much
        W = W.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T) # TODO: explain [:T, :T]
        W = F.softmax(W, dim=-1) # (B,T,T)
        
        # compute weighted aggregation of values
        out = W @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out        

# %%
class SimpleAttentionModel(nn.Module):
    
    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.token_embeddings_table = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd=n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, inputs: torch.tensor, targets: torch.tensor = None):
        B, T = inputs.shape
        
        # inputs and targets are both (B,T) tensors of integers
        tok_emb = self.token_embeddings_table(inputs)
        pos_emb = self.position_embeddings_table(torch.arange(T)) # (T,C)
        
        x = tok_emb + pos_emb # (B,T,C)
        x = self.sa_head(x)  # apply one head of self-attention (B,T,C)
        logits = self.lm_head(x)  # (B,T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view (B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, inputs, max_new_tokens):
        # Generate new tokens one at a time, using only the last token to predict the next
        
        for _ in range(max_new_tokens):
            
            # make sure the inputs are at max block_size number of tokens
            # necessary because our position embedding can only encode position information for up to block_size tokens
            inputs = inputs[:, -block_size:]
            #print(f"Generate: inputs.shape = {inputs.shape}")
            
            logits, _ = self(inputs)  # shape: (B,T,C)
            # For generation, we only need the predictions from the last position
            probs = F.softmax(logits[:, -1, :], dim=-1)  # shape: (B,C)
            
            # Sample from the probability distribution to get the next token
            inputs_next = torch.multinomial(probs, num_samples=1)  # shape: (B,1)
            
            # Append the new token to our sequence
            inputs = torch.cat((inputs, inputs_next), dim=1)  # shape: (B,T+1)
        return inputs

# %%
# Read dataset
with open(r'Attention_Mechanism/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# How we encode text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Dataset encoding
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

# Generating train and test split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# %%
model = SimpleAttentionModel(n_embd=n_embd, vocab_size=vocab_size)

inputs = torch.zeros((1,1), dtype=torch.long)
decoded_output = decode(model.generate(inputs=inputs, max_new_tokens=500)[0].tolist())
print(len(decoded_output))
print(decoded_output)

# %%

def get_batch(split):
    # from part 1
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i + block_size] for i in idxs])
    y = torch.stack([data[i+1:i+block_size+1] for i in idxs])
    return x,y  

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # We use a relatively large learning rate, because the model is fairly small

# Train
for steps in range(10000): # increase number of steps for good results...

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# %%
inputs = torch.zeros((1,1), dtype=torch.long)
decoded_output = decode(model.generate(inputs=inputs, max_new_tokens=500)[0].tolist())
print(len(decoded_output))
print(decoded_output)

# %%



