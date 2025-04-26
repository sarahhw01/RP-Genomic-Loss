"""
Code from tutorial found on the following page: https://medium.com/@hhpatil001/transformers-from-scratch-in-simple-python-part-i-b290760c1040
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from torch import nn
from math import sqrt

# Tokenize an example text
tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNA_bert_6')
text = "ACGTAGCTAGCTAGGCTAACGTAACCGT"
#print(tokenizer(text, add_special_tokens=False, return_tensors='pt'))
inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt')

# get configuration of the BERT model
config = AutoConfig.from_pretrained('zhihan1996/DNA_bert_6')
#print(config)

# create dense embeddings
# create lookup table
token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
#print(token_embeddings)
# generate embeddings by feeding in the input IDs
input_embeds = token_embeddings(inputs.input_ids)
#print(input_embeds.size()) # This is a tensor of shape [batch_size, seq_len, hidden_dim]

# Calculate Attention weights
query = key = value = input_embeds

# Calculate the self-attention: for each token check which other words should 'get attention', important to get context for each token
# One head attention: focuses on one aspect of similarity
def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    # torch.bmm is batch matrix - matrix multiplication. Basically a dot product.
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k) 
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

# Multihead attention: having several heads allows the model to focus on several aspects at once
# Single Head attention layer:
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim) -> None:
        super().__init__()
        # initialize three independent linear layers, apply matrix multiplication to the embedding vectors to get tensores of shape [batch_size, seq_len, head_dim]
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
    
    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return(attn_outputs)

# Multihead attention layer: takes input tensor (hidden_state) and applies multiple attention heads independently, concatenates their outputs and 
# passes them through a final linear layer
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

multihead_attn = MultiHeadAttention(config)
attn_output = multihead_attn(input_embeds)
#print(attn_output.size())

# Implement feed forward layer
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return(x)

feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_output)
print(ff_outputs.size())

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, hidden_state):
        # Multi-head attention with residual connection
        attn_output = self.attention(hidden_state)
        hidden_state = self.attention_norm(hidden_state + attn_output)

        # Feed forward network with residual connection
        ffn_output = self.feed_forward(hidden_state)
        hidden_state = self.ffn_norm(hidden_state + ffn_output)
        
        return hidden_state

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens + positions
        hidden_state = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden_state = self.dropout(hidden_state)
        
        for layer in self.layers:
            hidden_state = layer(hidden_state)

        return hidden_state

class TransformerForClassification(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
    
    def forward(self, input_ids):
        hidden_state = self.encoder(input_ids)  # [batch_size, seq_length, hidden_dim]
        pooled_output = hidden_state[:, 0]  # usually use [CLS] token (index 0)
        logits = self.classifier(pooled_output)
        return logits
    
# Define model
model = TransformerForClassification(config, num_classes=2)

# Optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

# Dummy data
inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt')
labels = torch.tensor([1])  # batch_size=1, label=1

# Move model and data to device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
labels = labels.to(device)

# Training loop
for epoch in range(100):
    model.train()  # Set model to training mode
    
    optimizer.zero_grad()  # Clear gradients

    # Forward pass
    logits = model(inputs['input_ids'])
    
    # Compute loss
    loss = loss_fn(logits, labels)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()

    # Print loss for each epoch
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

