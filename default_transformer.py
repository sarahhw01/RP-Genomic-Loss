"""
Code from tutorial found on the following page: https://medium.com/@hhpatil001/transformers-from-scratch-in-simple-python-part-i-b290760c1040
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from torch import nn
from math import sqrt

# Tokenize an example text
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text = 'I love data science.'
#print(tokenizer(text, add_special_tokens=False, return_tensors='pt'))
inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt')

# get configuration of the BERT model
config = AutoConfig.from_pretrained('bert-base-uncased')
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